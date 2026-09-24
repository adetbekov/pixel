"""The automatic miner is off the request path, and the pool it reads is bounded.

The thing under test is a *timing* guarantee, so the fake mining run here blocks
on an event the test controls: an answer that comes back while the run is still
parked is an answer that did not wait for it. No sleeps, no thresholds.
"""

import json
import threading

import httpx
import pytest

from backend.brain.engine import set_engine
from backend.main import app
from backend.miner import MineResult, request_mine, stop_worker, wait_idle
from backend.miner import worker as miner_worker
from backend.miner.case import (
    DEFAULT_POOL_WINDOW,
    load_pool,
    pool_ids,
    pool_size,
    pool_window,
)

from .fakes import FakeEngine
from .trick_cluster import (
    TRICK_COMMANDS,
    TRICK_EMBEDDINGS,
    TRICK_ROUTES,
    draft_json,
    fill_pool,
)

TEACHER_REPLY = json.dumps(
    {
        "reply": "Тада!",
        # A declined answer never enters the pool, so it would never trigger the
        # miner either — and this test is about the trigger.
        "handled": True,
        "actions": [
            {"action": "spin"},
            {"action": "set_face", "face": "happy"},
            {"action": "say", "text": "Тада!"},
        ],
    },
    ensure_ascii=False,
)


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(seeded):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.fixture()
def miner_engine():
    """The engine the miner and the app share, scripted for the trick cluster."""
    fake = FakeEngine(routes=TRICK_ROUTES, embeddings=TRICK_EMBEDDINGS)
    set_engine(fake)
    yield fake
    set_engine(None)


@pytest.fixture()
def parked_run(monkeypatch):
    """A mining run that starts and then stops dead until the test frees it."""

    class Parked:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()
            self.runs = 0

        def __call__(self) -> MineResult:
            self.runs += 1
            self.entered.set()
            assert self.release.wait(5), "the parked run was never released"
            return MineResult(started=True, proposals=0)

    parked = Parked()
    monkeypatch.setattr(miner_worker, "mine_once", parked)
    yield parked
    parked.release.set()
    stop_worker()


@pytest.mark.anyio
async def test_chat_answers_without_waiting_for_the_miner(
    client, seeded, engine, teacher, parked_run, monkeypatch
):
    """The acceptance criterion: a miss that triggers the miner still answers now."""
    monkeypatch.setenv("MINER_BATCH", "1")
    teacher(TEACHER_REPLY)

    response = await client.post("/api/chat", json={"text": "покажи фокус"})

    assert response.status_code == 200
    assert response.json()["engine"] == "gemini"
    # The run was triggered...
    assert parked_run.entered.wait(5)
    # ...and it is still parked, so the answer above cannot have waited for it.
    assert not parked_run.release.is_set()


def test_a_trigger_arriving_during_a_run_is_dropped_not_queued(parked_run):
    """One slot: a second run would re-read the same pool and race the first."""
    assert request_mine() is True
    assert parked_run.entered.wait(5)

    # One in flight plus one waiting fills the queue; everything after is dropped.
    assert request_mine() is True
    assert request_mine() is False
    assert request_mine() is False

    parked_run.release.set()
    assert wait_idle(5)
    assert parked_run.runs == 2


@pytest.mark.anyio
async def test_post_mine_stays_synchronous(client, seeded, miner_engine, generator, parked_run):
    """The manual run answers with its count, so it cannot go through the worker."""
    generator(draft_json())
    fill_pool(seeded)

    body = (await client.post("/api/mine")).json()

    assert body == {"started": True, "proposals": 1}
    assert parked_run.runs == 0


def test_the_pool_is_read_newest_first_and_bounded(seeded, monkeypatch):
    """Otherwise every run embeds a bigger pool than the last one, forever."""
    monkeypatch.setenv("MINER_POOL_WINDOW", "3")
    fill_pool(seeded, TRICK_COMMANDS)

    pool = load_pool(seeded)

    assert [case.user_text for case in pool] == TRICK_COMMANDS[2:]
    # Still oldest-first inside the window — the clustering reads it in order.
    assert [case.id for case in pool] == sorted(case.id for case in pool)
    # And nothing was marked mined: those cases are out of the window, not gone.
    row = seeded.execute("SELECT COUNT(*) AS n FROM teacher_log WHERE mined = 0").fetchone()
    assert row["n"] == len(TRICK_COMMANDS)


def test_the_trigger_and_the_ledger_still_see_the_whole_pool(seeded, monkeypatch):
    """Both read past the window, and for different reasons.

    `pool_size` because `size % MINER_BATCH == 0` would otherwise be true on
    every arrival once the pool passed the window — a run per miss. `pool_ids`
    because a case the window left behind has not left the pool, and the attempts
    ledger drops the budget of any signature whose cases have.
    """
    monkeypatch.setenv("MINER_POOL_WINDOW", "3")
    ids = fill_pool(seeded, TRICK_COMMANDS)

    assert len(load_pool(seeded)) == 3
    assert pool_size(seeded) == len(TRICK_COMMANDS)
    assert pool_ids(seeded) == set(ids)


def test_the_window_has_a_default_and_refuses_nonsense(monkeypatch):
    assert pool_window() == DEFAULT_POOL_WINDOW
    monkeypatch.setenv("MINER_POOL_WINDOW", "0")
    assert pool_window() == 1
