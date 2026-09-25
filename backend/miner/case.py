"""One row of ``teacher_log``, read back as something the miner can use.

``teacher_log`` stores the command and the robot's state inside ``state_json``
(see :func:`backend.teacher.log.log_case`), so reading the pool is a parse, not
a plain SELECT. Anything that does not parse is skipped and logged rather than
taking the whole run down — the pool is untrusted input by construction, since
half of it was written by a language model.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..state import START_ENERGY, START_FACE, START_FULLNESS, START_MOOD, RobotState

log = logging.getLogger(__name__)

#: The newest N unmined cases one run may look at. 40 is eight `MINER_BATCH`
#: triggers of memory and about a second of embed on the real checkpoint.
DEFAULT_POOL_WINDOW = 40


@dataclass(frozen=True)
class Case:
    """One command the router missed, plus the plan the teacher answered with."""

    id: int
    user_text: str
    state: RobotState
    actions: list[dict[str, Any]]

    @property
    def action_names(self) -> set[str]:
        """What the teacher did, ignoring order and arguments.

        Two plans that say different sentences are still the same behaviour, and
        the teacher never phrases a reply the same way twice — so the backtest
        compares these sets and not the plans.
        """
        return {step["action"] for step in self.actions if isinstance(step, dict)}


def _state(payload: dict[str, Any]) -> RobotState:
    raw = payload.get("state")
    raw = raw if isinstance(raw, dict) else {}
    return RobotState(
        mood=float(raw.get("mood", START_MOOD)),
        energy=float(raw.get("energy", START_ENERGY)),
        fullness=float(raw.get("fullness", START_FULLNESS)),
        face=str(raw.get("face", START_FACE)),
    )


def _parse(row: sqlite3.Row) -> Case | None:
    try:
        payload = json.loads(row["state_json"] or "{}")
        actions = json.loads(row["actions_json"] or "[]")
    except (TypeError, ValueError):
        log.warning("teacher_log row %s is not readable JSON", row["id"])
        return None
    if not isinstance(payload, dict) or not isinstance(actions, list):
        return None

    # A failed teacher call is logged too, and it is a useful signal — but its
    # plan is FALLBACK_PLAN, i.e. "curious face, I did not understand". Mining a
    # skill out of those would teach Pixel to not understand on purpose.
    if payload.get("error"):
        return None

    # And the same holds for a call that *succeeded* and declined. `handled` is
    # false when the teacher answered "я не умею заказывать еду" — a real plan, a
    # real reply, and the worst possible thing to learn: a mined refusal answers
    # instantly, from Laya, for ever, and the command never reaches the teacher
    # again. Measured before this check existed (JEB-1547): four of five phrasings
    # of "покажи фокус" came back as refusals and the drafted skill said "я не
    # умею показывать фокусы" from Laya in 300 ms.
    #
    # `is False`, not falsy: a row written before this field existed has no
    # `handled` at all, and a missing verdict must not read as a refusal.
    if payload.get("handled") is False:
        return None

    text = str(payload.get("user_text") or "").strip()
    if not text or not actions:
        return None
    return Case(id=int(row["id"]), user_text=text, state=_state(payload), actions=actions)


def pool_window() -> int:
    """How many of the newest mineable cases one run drafts from."""
    return max(1, int(os.environ.get("MINER_POOL_WINDOW", DEFAULT_POOL_WINDOW)))


def load_pool(conn: sqlite3.Connection, *, bounded: bool = True) -> list[Case]:
    """The cases waiting to be mined, oldest first — the newest window of them.

    The window is what stops a run getting slower forever. ``mined = 1`` is set
    only on a case that was published (``backend/miner/run.py``), so a cluster
    the backtest refused, and every one-off miss, stays in the pool for good —
    and a run reads the whole pool twice, once to group it and once as the
    backtest's control set. The grouping alone is 430 ms of embed at 20 cases
    and 2473 ms at 100 on the fallback path, against 110 ms for one router pass
    (measured on the real checkpoint, JEB-1509). Unbounded, every run takes
    ``LayaEngine._lock`` more times than the last one, and running it on a
    background thread hides that rather than fixing it.

    It is also the bound on the worst wait a concurrent chat can take, and that
    is this fallback path specifically: ``engine.embed`` takes the engine lock
    **once for the whole batch**, so a chat that arrives inside it waits the
    whole embed — measured at 1092 ms at p50 for a window of 40 and 572 ms for
    20, linear in the window (JEB-1599,
    ``scripts/bench_router_under_mining.py``). When the teacher grouper answers
    instead, nothing embeds and the window is a run-length knob only: there the
    chat waits one backtest forward pass (260 ms) whatever the window is.

    A case that falls out of the window is not deleted and not marked mined — it
    is simply too old to still be the pattern worth a skill. Mining it later is
    one new phrasing away, since that puts it back inside the window.
    ``MINER_POOL_WINDOW`` overrides the bound; ``bounded=False`` asks for the
    whole pool, which only :func:`pool_size` wants.
    """
    sql = "SELECT id, state_json, actions_json FROM teacher_log WHERE mined = 0 ORDER BY id DESC"
    rows = (
        conn.execute(f"{sql} LIMIT ?", (pool_window(),)).fetchall()
        if bounded
        else conn.execute(sql).fetchall()
    )
    return [case for case in (_parse(row) for row in reversed(rows)) if case is not None]


def pool_ids(conn: sqlite3.Connection) -> set[int]:
    """Every mineable case id, window or no window.

    What the attempts ledger is swept against (:func:`backend.miner.attempts.forget`):
    a signature dies when its cases *left the pool*, and a case that is merely
    outside this run's window has not left it — it is still unmined, and one new
    phrasing puts it back in range with whatever budget it had spent.
    """
    return {case.id for case in load_pool(conn, bounded=False)}


def is_mineable(conn: sqlite3.Connection, case_id: int) -> bool:
    """Did this one row land in the pool?

    The mining trigger counts *arrivals*, not rows — see
    :func:`backend.miner.run.mining_due` for why that distinction is what keeps
    it from firing on every message. Which rows count is decided by
    :func:`_parse`, asked about one row here rather than restated as a second
    predicate that could disagree with it.
    """
    row = conn.execute(
        "SELECT id, state_json, actions_json FROM teacher_log WHERE id = ? AND mined = 0",
        (case_id,),
    ).fetchone()
    return row is not None and _parse(row) is not None


def pool_size(conn: sqlite3.Connection) -> int:
    """How many cases the miner could actually use — what the trigger watches.

    ``len(load_pool(conn))`` and not ``COUNT(*) WHERE mined = 0``, so that "pool
    size" has exactly one definition. The rows :func:`_parse` drops are never
    marked ``mined``, so a count of raw rows drifts: three unusable rows and the
    every-``MINER_BATCH``-th trigger fires on two mineable cases — below
    ``MINER_MIN_CLUSTER``, so the run finds nothing, and the offset never
    recovers. With refusals now dropped too (above) that drift would only grow.

    The cost is parsing tens of rows per teacher call instead of a ``COUNT``. The
    mining run this decides reads and parses the same rows anyway.

    Counted over the **whole** pool, not the window a run drafts from: the
    trigger is ``size % MINER_BATCH == 0``, and a size pinned at the window would
    satisfy that on every single arrival once the pool grew past it — a run per
    miss, which is the opposite of what the window is for.
    """
    return len(load_pool(conn, bounded=False))
