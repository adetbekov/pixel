"""Stage 5: a 👎 has consequences.

The point of these tests is not that a column gets written — it is that a bad
skill actually stops answering, and that everything the miner needs to try again
comes back with it.
"""

import json
import uuid

import httpx
import pytest

from backend import db, feedback
from backend.brain.engine import set_engine
from backend.brain.router import RouterMiss, route
from backend.brain.skill import load_skills
from backend.main import app
from backend.state import iso, utcnow

from .fakes import FakeEngine
from .trick_cluster import TRICK_EMBEDDINGS, TRICK_ROUTES, draft_json, fill_pool


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(conn):
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


def rate(conn, skill_id: str, votes: list[int]) -> list[int]:
    """Log one interaction per vote for `skill_id`, then return their ids.

    The votes are applied through `feedback.record`, i.e. the same path
    `/api/feedback` takes, so the auto-disable rule is what is under test rather
    than a hand-written UPDATE.

    `ts` is *now*, exactly as `_log_interaction` writes it, because the health
    query only counts a skill's current incarnation (`i.ts >= s.created_at`) —
    a hand-picked constant would sit before the skill was created and count as
    somebody else's life.
    """
    ids = []
    for value in votes:
        cursor = conn.execute(
            "INSERT INTO interactions (ts, user_text, engine, skill_id, confidence,"
            " latency_ms, actions_json, reply_text, feedback)"
            " VALUES (?, 'танцуй', 'laya', ?, 0.9, 12, '[]', 'ок', NULL)",
            (iso(utcnow()), skill_id),
        )
        ids.append(int(cursor.lastrowid))
    conn.commit()
    for interaction_id, value in zip(ids, votes, strict=True):
        feedback.record(conn, interaction_id=interaction_id, value=value)
    return ids


def status_of(conn, skill_id: str) -> str:
    return conn.execute("SELECT status FROM skills WHERE id = ?", (skill_id,)).fetchone()["status"]


def test_four_dislikes_in_ten_disable_the_skill(seeded):
    rate(seeded, "greet", [1] * 6 + [-1] * 4)

    row = seeded.execute(
        "SELECT status, disabled_at, disabled_reason FROM skills WHERE id = 'greet'"
    ).fetchone()
    assert row["status"] == "disabled"
    assert row["disabled_reason"] == "dislike_rate"
    assert row["disabled_at"]


def test_two_dislikes_in_three_do_not_disable_the_skill(seeded):
    """Under `SKILL_MIN_RATED` the ratio is meaningless — 2/3 is 67% and stays on.

    Without this floor the first 👎 on a freshly accepted skill would kill it
    before it ever had a chance to be judged.
    """
    rate(seeded, "greet", [1, -1, -1])
    assert status_of(seeded, "greet") == "active"


def test_exactly_the_limit_is_not_over_it(seeded):
    """3/10 is 30%, and the rule is *strictly* greater than the limit."""
    rate(seeded, "greet", [1] * 7 + [-1] * 3)
    assert status_of(seeded, "greet") == "active"


def test_a_flipped_vote_is_recounted_not_accumulated(seeded):
    """Re-rating an interaction overwrites it; the verdict follows the table."""
    ids = rate(seeded, "greet", [1] * 7 + [-1] * 2)  # 2/9 = 22%, under the limit
    assert status_of(seeded, "greet") == "active"

    # The same interaction rated 👎 twice is still one dislike: 2/9 stays under.
    # An accumulating counter would read 3 here and disable the skill.
    feedback.record(seeded, interaction_id=ids[-1], value=-1)
    assert status_of(seeded, "greet") == "active"

    # A third *distinct* dislike crosses it — 3/9 = 33%.
    feedback.record(seeded, interaction_id=ids[0], value=-1)
    assert status_of(seeded, "greet") == "disabled"

    # And flipping one back turns the skill's own counters around, even though
    # the skill stays off until a human re-enables it.
    feedback.record(seeded, interaction_id=ids[0], value=1)
    row = seeded.execute(
        "SELECT SUM(feedback IS -1) AS dislikes FROM interactions WHERE skill_id = 'greet'"
    ).fetchone()
    assert row["dislikes"] == 2


def test_a_disabled_skill_leaves_the_router_immediately(seeded, engine):
    """The acceptance criterion: the same command must now miss, not hit.

    `load_skills(only_active=True)` is what `/api/chat` calls per request, so
    this is the router's real view and not a stand-in for it.
    """
    engine.pick, engine.confidence = "greet", 0.95
    from backend.state import read_state

    state = read_state(seeded)
    assert not isinstance(route(engine, load_skills(seeded, only_active=True), "привет", state),
                          RouterMiss)

    rate(seeded, "greet", [1] * 6 + [-1] * 4)

    skills = load_skills(seeded, only_active=True)
    assert "greet" not in {skill.id for skill in skills}
    assert isinstance(route(engine, skills, "привет", state), RouterMiss)


def test_a_gemini_dislike_disables_nothing(seeded):
    """`skill_id IS NULL` — there is no skill to blame, and no crash either."""
    cursor = seeded.execute(
        "INSERT INTO interactions (ts, user_text, engine, skill_id, confidence, latency_ms,"
        " actions_json, reply_text, feedback)"
        " VALUES (?, 'спой', 'gemini', NULL, 0.2, 900, '[]', 'ок', NULL)",
        (iso(utcnow()),),
    )
    seeded.commit()
    assert feedback.record(seeded, interaction_id=int(cursor.lastrowid), value=-1) is True
    assert status_of(seeded, "greet") == "active"


def test_an_unknown_interaction_is_not_recorded(seeded):
    assert feedback.record(seeded, interaction_id=4242, value=-1) is False


# ─── Returning the mined cases to the pool ──────────────────────────────────


def accepted_proposal(conn, skill_id: str, case_ids: list[int]) -> str:
    """Reconstruct what stage 4 leaves behind: an accepted proposal and its cases."""
    proposal_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO skill_proposals (id, skill_json, match_rate, sample_ids, status, created_at)"
        " VALUES (?, ?, 0.9, ?, 'accepted', '2026-09-23T09:00:00Z')",
        (proposal_id, json.dumps({"id": skill_id}), json.dumps(case_ids)),
    )
    for case_id in case_ids:
        conn.execute(
            "INSERT INTO teacher_log (id, interaction_id, state_json, raw_response, actions_json,"
            " mined, cluster_id) VALUES (?, NULL, '{}', '', '[]', 1, ?)",
            (case_id, proposal_id),
        )
    conn.commit()
    return proposal_id


def test_disabling_returns_the_cases_to_the_pool(seeded):
    """The miner gets its raw material back and can propose a better wording."""
    seeded.execute(
        "INSERT INTO skills (id, json, status, origin, created_at)"
        " VALUES ('dance', '{\"id\": \"dance\"}', 'active', 'mined', '2026-09-23T09:00:00Z')"
    )
    accepted_proposal(seeded, "dance", [11, 12, 13])
    rate(seeded, "dance", [1] * 6 + [-1] * 4)

    assert status_of(seeded, "dance") == "disabled"
    rows = seeded.execute(
        "SELECT mined, cluster_id FROM teacher_log WHERE id IN (11, 12, 13)"
    ).fetchall()
    assert [(row["mined"], row["cluster_id"]) for row in rows] == [(0, None)] * 3


def test_disabling_lifts_the_rejected_signature_of_the_same_cluster(seeded):
    """Cases back in the pool but still blocked is a silent hole — check it.

    `backend/miner/run.py` skips a cluster whose case set matches a *rejected*
    proposal. If an earlier rejection of exactly these cases stayed 'rejected',
    the pool would refill and no proposal would ever appear again.
    """
    from backend.miner.run import _rejected_signatures

    seeded.execute(
        "INSERT INTO skills (id, json, status, origin, created_at)"
        " VALUES ('dance', '{\"id\": \"dance\"}', 'active', 'mined', '2026-09-23T09:00:00Z')"
    )
    seeded.execute(
        "INSERT INTO skill_proposals (id, skill_json, match_rate, sample_ids, status, created_at)"
        " VALUES ('old', '{\"id\": \"dance_v0\"}', 0.7, '[13, 11, 12]', 'rejected',"
        " '2026-09-23T08:00:00Z')"
    )
    accepted_proposal(seeded, "dance", [11, 12, 13])
    assert (11, 12, 13) in _rejected_signatures(seeded)

    rate(seeded, "dance", [1] * 6 + [-1] * 4)

    assert (11, 12, 13) not in _rejected_signatures(seeded)
    # The row is retired, not deleted: the history of the decision is kept.
    assert seeded.execute(
        "SELECT status FROM skill_proposals WHERE id = 'old'"
    ).fetchone()["status"] == "retired"


def test_an_unrelated_rejection_survives(seeded):
    """Only the disabled skill's own cluster is unblocked, not every rejection."""
    from backend.miner.run import _rejected_signatures

    seeded.execute(
        "INSERT INTO skills (id, json, status, origin, created_at)"
        " VALUES ('dance', '{\"id\": \"dance\"}', 'active', 'mined', '2026-09-23T09:00:00Z')"
    )
    seeded.execute(
        "INSERT INTO skill_proposals (id, skill_json, match_rate, sample_ids, status, created_at)"
        " VALUES ('other', '{\"id\": \"sing\"}', 0.7, '[91, 92]', 'rejected',"
        " '2026-09-23T08:00:00Z')"
    )
    accepted_proposal(seeded, "dance", [11, 12, 13])
    rate(seeded, "dance", [1] * 6 + [-1] * 4)

    assert (91, 92) in _rejected_signatures(seeded)


def test_a_seed_skill_gets_no_exemption(seeded):
    """No carve-out: a starter skill the user keeps disliking is just as wrong."""
    rate(seeded, "feed", [1] * 5 + [-1] * 5)
    assert status_of(seeded, "feed") == "disabled"
    # Nothing to return to the pool — a seed skill was never a proposal.
    assert seeded.execute("SELECT COUNT(*) AS n FROM teacher_log").fetchone()["n"] == 0


# ─── The retry: a disabled skill can actually be mined again ────────────────


@pytest.mark.anyio
async def test_a_disabled_skill_can_be_mined_and_accepted_again(
    client, seeded, miner_engine, generator
):
    """The whole point of returning the cases: a *proposal* appears, not just rows.

    Two locks had to come off together, and this is the test that would have
    caught either one left on: the rejected signature (lifted in
    `backend/feedback.py`) and the skill id (`_taken_ids` in
    `backend/miner/run.py`). With the id still held, the retry dies in a
    `log.warning` on every run; with the id freed but `accept_proposal` still
    doing a bare INSERT, accepting the retry is a 500 on the primary key.
    """
    generator(draft_json())
    fill_pool(seeded)

    assert (await client.post("/api/mine")).json()["proposals"] == 1
    first = (await client.get("/api/proposals")).json()[0]
    assert first["skill"]["id"] == "show_trick"
    assert (await client.post(f"/api/proposals/{first['id']}/accept")).status_code == 200
    assert status_of(seeded, "show_trick") == "active"

    # The user turns out to hate it: 4 dislikes in 10 ratings.
    rate(seeded, "show_trick", [1] * 6 + [-1] * 4)
    assert status_of(seeded, "show_trick") == "disabled"
    assert seeded.execute(
        "SELECT COUNT(*) AS n FROM teacher_log WHERE mined = 0"
    ).fetchone()["n"] == 5

    # The miner proposes the same cluster again, under the same obvious id.
    assert (await client.post("/api/mine")).json()["proposals"] == 1
    retry = (await client.get("/api/proposals")).json()[0]
    assert retry["skill"]["id"] == "show_trick"
    assert retry["id"] != first["id"]

    assert (await client.post(f"/api/proposals/{retry['id']}/accept")).status_code == 200
    row = seeded.execute(
        "SELECT status, disabled_at, disabled_reason FROM skills WHERE id = 'show_trick'"
    ).fetchone()
    assert row["status"] == "active"
    # The card must not still read "отключён из-за дизлайков" — it is a new skill.
    assert row["disabled_at"] is None
    assert row["disabled_reason"] is None

    # The card opens on a clean slate: the previous life's 10 uses and 4 dislikes
    # belong to a skill that no longer exists, even though it had this id.
    card = next(
        item for item in (await client.get("/api/skills")).json() if item["id"] == "show_trick"
    )
    assert (card["status"], card["disabled_reason"]) == ("active", None)
    assert (card["uses"], card["likes"], card["dislikes"]) == (0, 0, 0)

    # And it routes again, in the same process.
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert (body["engine"], body["skill_id"]) == ("laya", "show_trick")

    # The gate that matters: the first rating of the new life must not be judged
    # against the old one. Counting all 11 ratings gives 4/11 = 36% > 30% at
    # `rated >= 5`, so an inherited history kills the retry on its first 👍.
    assert (
        await client.post(
            "/api/feedback", json={"interaction_id": body["interaction_id"], "value": 1}
        )
    ).status_code == 200
    assert status_of(seeded, "show_trick") == "active"

    card = next(
        item for item in (await client.get("/api/skills")).json() if item["id"] == "show_trick"
    )
    assert (card["uses"], card["likes"], card["dislikes"]) == (1, 1, 0)

    # A 👎 of the new life is judged on the new life alone — one rating is under
    # `SKILL_MIN_RATED`, so it does not disable anything either.
    second = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    await client.post("/api/feedback", json={"interaction_id": second["interaction_id"], "value": -1})
    assert status_of(seeded, "show_trick") == "active"


def test_ratings_from_before_this_incarnation_are_not_counted(seeded):
    """The boundary on its own: `created_at` moving forward resets the verdict.

    `skills` and `interactions` have no foreign key, so the old rows keep the id
    of the skill that is gone. Only `created_at` separates the two lives.
    """
    rate(seeded, "greet", [1] * 6 + [-1] * 4)
    assert status_of(seeded, "greet") == "disabled"

    # Re-accepting rewrites `created_at` — which is exactly what `accept_proposal`
    # does through INSERT OR REPLACE.
    seeded.execute(
        "UPDATE skills SET status = 'active', created_at = ?, disabled_at = NULL,"
        " disabled_reason = NULL WHERE id = 'greet'",
        (iso(utcnow()),),
    )
    seeded.commit()

    assert feedback.review_skill(seeded, "greet") is False
    rate(seeded, "greet", [1])
    assert status_of(seeded, "greet") == "active"

    # And the new life is judged normally once it has ratings of its own.
    rate(seeded, "greet", [1] * 5 + [-1] * 4)
    assert status_of(seeded, "greet") == "disabled"


@pytest.mark.anyio
async def test_an_active_skill_still_blocks_its_id(client, seeded, miner_engine, generator):
    """Only a *disabled* id is free. A live skill is still a 409 and still taken."""
    from backend.miner.run import _taken_ids

    generator(draft_json())
    fill_pool(seeded)
    assert (await client.post("/api/mine")).json()["proposals"] == 1
    proposal = (await client.get("/api/proposals")).json()[0]
    await client.post(f"/api/proposals/{proposal['id']}/accept")

    assert "show_trick" in _taken_ids(seeded)
    assert "greet" in _taken_ids(seeded)

    # A second proposal for the same id cannot slip past the accept endpoint.
    seeded.execute(
        "INSERT INTO skill_proposals (id, skill_json, match_rate, sample_ids, status, created_at)"
        " VALUES ('dup', ?, 0.9, '[]', 'pending', '2026-09-23T09:00:00Z')",
        (json.dumps(proposal["skill"]),),
    )
    seeded.commit()
    assert (await client.post("/api/proposals/dup/accept")).status_code == 409


# ─── The endpoint and the panel ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_feedback_endpoint_disables_and_the_panel_explains_why(client, seeded, engine):
    engine.pick, engine.confidence = "greet", 0.95
    for index in range(10):
        interaction_id = (await client.post("/api/chat", json={"text": "привет"})).json()[
            "interaction_id"
        ]
        value = -1 if index < 4 else 1
        assert (
            await client.post(
                "/api/feedback", json={"interaction_id": interaction_id, "value": value}
            )
        ).status_code == 200

    cards = {card["id"]: card for card in (await client.get("/api/skills")).json()}
    assert cards["greet"]["status"] == "disabled"
    assert cards["greet"]["disabled_reason"] == "dislike_rate"
    assert cards["greet"]["dislikes"] == 4


@pytest.mark.anyio
async def test_history_survives_a_reload(client, seeded, engine):
    """The vote is stored, and `/api/history` is what puts it back on screen."""
    engine.pick, engine.confidence = "greet", 0.95
    first = (await client.post("/api/chat", json={"text": "привет"})).json()["interaction_id"]
    await client.post("/api/action", json={"name": "feed"})
    await client.post("/api/feedback", json={"interaction_id": first, "value": -1})

    history = (await client.get("/api/history")).json()
    assert [item["interaction_id"] for item in history] == sorted(
        item["interaction_id"] for item in history
    )
    by_id = {item["interaction_id"]: item for item in history}
    assert by_id[first]["feedback"] == -1
    assert by_id[first]["engine"] == "laya"
    assert by_id[first]["user_text"] == "привет"


@pytest.mark.anyio
async def test_history_is_bounded(client, conn):
    for _ in range(3):
        await client.post("/api/action", json={"name": "pet"})
    assert len((await client.get("/api/history?limit=2")).json()) == 2
    # An absurd limit is clamped, not honoured, and not a 422.
    assert (await client.get("/api/history?limit=100000")).status_code == 200


def test_the_migration_is_idempotent(conn):
    """`connect()` already ran it once; running it again must be a no-op."""
    db.migrate(conn)
    db.migrate(conn)
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(skills)")}
    assert {"disabled_at", "disabled_reason"} <= columns


def test_the_thresholds_come_from_the_environment(seeded, monkeypatch):
    monkeypatch.setenv("SKILL_MIN_RATED", "2")
    monkeypatch.setenv("SKILL_DISLIKE_LIMIT", "0.4")
    assert feedback.min_rated() == 2
    assert feedback.dislike_limit() == 0.4

    rate(seeded, "greet", [1, -1])  # 1/2 = 50% over 40%, and 2 ratings is enough
    assert status_of(seeded, "greet") == "disabled"
