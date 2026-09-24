"""The automatic miner, off the request path.

A mining run is an embed of the whole pool, one Gemini call to group it (~0.9 s
of network) and a backtest per candidate — seconds, and all of it behind
``LayaEngine._lock``. Running it inside ``POST /api/chat`` made every
``MINER_BATCH``-th miss pay for it, and the delay grew with the pool: measured
on the real checkpoint (JEB-1509) the embed alone is 430 ms at 20 cases and
2473 ms at 100, against 110 ms for one router ``predict``.

So the trigger hands the run to this worker instead: one daemon thread, and a
**one-slot** queue in front of it. One slot, not N, because a second run would
re-read the same pool and race the first one to the same proposals — exactly
what :func:`backend.miner.run.mine_once` already refuses with its own lock. A
trigger that arrives while a run is in flight and one already queued is dropped,
not buffered: the pool it would read is still there on the next miss.

What this does **not** fix is the engine lock — a backtest running in the
background still holds it, so a chat arriving mid-run waits for the current
forward pass. Bounding the pool (``MINER_POOL_WINDOW``, see
:func:`backend.miner.case.load_pool`) is what bounds that wait.

``POST /api/mine`` does not come through here. It is the manual run and its
caller wants the proposal count back, so it stays synchronous.
"""

from __future__ import annotations

import logging
import queue
import threading

from .run import mine_once

log = logging.getLogger(__name__)

#: The only two things that ever go through the queue.
_RUN = "run"
_STOP = "stop"

#: How long :func:`stop_worker` waits for a run already in flight. A mining run
#: is bounded by the Gemini timeouts, not by us; the thread is a daemon, so a
#: run that outlasts this does not hold the process open either.
STOP_TIMEOUT_S = 10.0

_state_lock = threading.Lock()


class _Worker:
    def __init__(self) -> None:
        self.queue: queue.Queue[str] = queue.Queue(maxsize=1)
        #: Clear from the moment a run is queued until the queue is empty again
        #: and the run has returned. Tests wait on it; nothing else reads it.
        self.idle = threading.Event()
        self.idle.set()
        self.thread = threading.Thread(target=self._loop, name="pixel-miner", daemon=True)

    def _loop(self) -> None:
        while True:
            item = self.queue.get()
            if item == _STOP:
                return
            try:
                result = mine_once()
                log.info("miner worker: run finished, %d proposal(s)", result.proposals)
            except Exception:
                # A failed run must not take the worker with it: the next miss
                # would then find nothing behind the trigger and mining would be
                # silently off until a restart.
                log.exception("miner worker: the run failed")
            finally:
                with _state_lock:
                    if self.queue.empty():
                        self.idle.set()


_worker: _Worker | None = None


def request_mine() -> bool:
    """Ask for a mining run in the background.

    Returns ``False`` when a run is already queued — the trigger is dropped on
    purpose, see the module docstring. Never blocks, never raises: the caller is
    a request that has already produced its answer.
    """
    global _worker
    with _state_lock:
        if _worker is None:
            _worker = _Worker()
            _worker.thread.start()
        worker = _worker
        # Cleared under the lock and set again by the run's own `finally`, so a
        # trigger and a finishing run cannot leave it lying about being idle.
        worker.idle.clear()
        try:
            worker.queue.put_nowait(_RUN)
        except queue.Full:
            log.info("miner worker: a run is already queued")
            return False
    return True


def stop_worker(timeout: float = STOP_TIMEOUT_S) -> None:
    """Stop the worker and wait for a run in flight. Idempotent."""
    global _worker
    with _state_lock:
        worker, _worker = _worker, None
    if worker is None:
        return
    try:
        # A queued-but-unstarted run is still ahead of this in the queue and
        # will be mined before the stop is seen. That is one extra run at
        # shutdown, which is cheaper than a special case that drops it.
        worker.queue.put(_STOP, timeout=timeout)
    except queue.Full:
        log.warning("miner worker: the stop signal did not fit the queue")
    worker.thread.join(timeout)
    if worker.thread.is_alive():
        log.warning("miner worker: still running after %.1fs — left as a daemon", timeout)


def wait_idle(timeout: float = STOP_TIMEOUT_S) -> bool:
    """Block until no run is queued or in flight. For tests and shutdown only."""
    with _state_lock:
        worker = _worker
    return True if worker is None else worker.idle.wait(timeout)
