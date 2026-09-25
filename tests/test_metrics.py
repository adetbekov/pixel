"""Stage 5: the numbers behind the learning panel.

`laya_share` is the project's headline metric, so the two ways to fake it —
counting button clicks, and dividing by zero — are what these tests pin.
"""

import httpx
import pytest

from backend import metrics
from backend.main import app
from backend.state import iso, utcnow

FIELDS = {
    "laya_share",
    "laya_share_24h",
    "avg_latency_laya_ms",
    "avg_latency_gemini_ms",
    "skills_active",
    "skills_disabled",
    "total_commands",
    "gemini_calls_24h",
    "teacher_calls_24h",
    "clusters_stuck",
}


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(conn):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


def log(conn, engine: str, *, latency_ms: int = 10, ts: str | None = None) -> int:
    """A logged interaction. Default `ts` is *now*, so the row is inside the window."""
    ts = ts or iso(utcnow())
    cursor = conn.execute(
        "INSERT INTO interactions (ts, user_text, engine, skill_id, confidence, latency_ms,"
        " actions_json, reply_text, feedback) VALUES (?, 'команда', ?, NULL, NULL, ?, '[]', '', NULL)",
        (ts, engine, latency_ms),
    )
    conn.commit()
    return int(cursor.lastrowid)


def test_an_empty_database_does_not_divide_by_zero(conn):
    body = metrics.collect(conn)
    assert set(body) == FIELDS
    assert body["laya_share"] == 0.0
    assert body["laya_share_24h"] == 0.0
    assert body["avg_latency_laya_ms"] == 0.0
    assert body["avg_latency_gemini_ms"] == 0.0
    assert body["total_commands"] == 0


def test_button_clicks_are_not_commands(conn):
    """Nine clicks and one Gemini call is a 0% share, not 90%."""
    for _ in range(9):
        log(conn, "button")
    log(conn, "gemini")

    body = metrics.collect(conn)
    assert body["total_commands"] == 1
    assert body["laya_share"] == 0.0

    log(conn, "laya")
    body = metrics.collect(conn)
    assert body["total_commands"] == 2
    assert body["laya_share"] == 0.5


def test_the_share_is_laya_over_laya_plus_gemini(conn):
    for _ in range(3):
        log(conn, "laya")
    log(conn, "gemini")
    assert metrics.collect(conn)["laya_share"] == 0.75


def test_average_latency_is_per_path(conn):
    log(conn, "laya", latency_ms=20)
    log(conn, "laya", latency_ms=40)
    log(conn, "gemini", latency_ms=1000)

    body = metrics.collect(conn)
    assert body["avg_latency_laya_ms"] == 30.0
    assert body["avg_latency_gemini_ms"] == 1000.0


def test_average_latency_ignores_rows_older_than_the_window(conn):
    """JEB-1574: one batch of 100 s rows must not poison the average forever.

    The panel says "how fast is the robot", so the averages read over the same
    24h as the counts next to them — otherwise history from a fixed timeout bug
    keeps the card showing 98 s while live calls answer in 1.6 s.
    """
    old = "2020-01-01T00:00:00Z"
    log(conn, "gemini", latency_ms=100_000, ts=old)
    log(conn, "laya", latency_ms=100_000, ts=old)

    body = metrics.collect(conn)
    assert body["avg_latency_gemini_ms"] == 0.0
    assert body["avg_latency_laya_ms"] == 0.0

    log(conn, "gemini", latency_ms=1600)
    log(conn, "laya", latency_ms=40)

    body = metrics.collect(conn)
    assert body["avg_latency_gemini_ms"] == 1600.0
    assert body["avg_latency_laya_ms"] == 40.0
    # The lifetime counts still see the old rows — only the averages are windowed.
    assert body["total_commands"] == 4


def test_the_24h_window_ignores_older_rows(conn):
    """The lifetime share hides today's trend; the windowed one is why it exists."""
    old = "2020-01-01T00:00:00Z"
    now = iso(utcnow())
    log(conn, "gemini", ts=old)
    log(conn, "gemini", ts=old)
    log(conn, "laya", ts=now)
    log(conn, "laya", ts=now)
    log(conn, "gemini", ts=now)

    body = metrics.collect(conn)
    assert body["laya_share"] == 0.4  # 2 of 5 over all time
    assert body["laya_share_24h"] == 0.667  # 2 of 3 today
    assert body["gemini_calls_24h"] == 1


def test_disabled_skills_are_counted_separately(seeded):
    assert metrics.collect(seeded)["skills_active"] == 4
    assert metrics.collect(seeded)["skills_disabled"] == 0

    seeded.execute("UPDATE skills SET status = 'disabled' WHERE id = 'greet'")
    seeded.commit()

    body = metrics.collect(seeded)
    assert body["skills_active"] == 3
    assert body["skills_disabled"] == 1


@pytest.mark.anyio
async def test_the_endpoint_returns_every_field(client, seeded, engine):
    engine.pick, engine.confidence = "greet", 0.95
    await client.post("/api/action", json={"name": "feed"})
    await client.post("/api/chat", json={"text": "привет"})

    body = (await client.get("/api/metrics")).json()
    assert set(body) == FIELDS
    # One button click and one routed command: the click is not in the share.
    assert body["total_commands"] == 1
    assert body["laya_share"] == 1.0
    assert body["skills_active"] == 4


@pytest.mark.anyio
async def test_accepting_a_skill_raises_the_share(client, seeded, engine):
    """The learning curve, end to end: a command that cost Gemini now costs Laya.

    Gemini is simulated by logging its interaction directly — the teacher's own
    path has its own tests, and what is under test here is the metric.
    """
    log(seeded, "gemini", latency_ms=900)
    assert (await client.get("/api/metrics")).json()["laya_share"] == 0.0

    engine.pick, engine.confidence = "greet", 0.95
    await client.post("/api/chat", json={"text": "привет"})
    assert (await client.get("/api/metrics")).json()["laya_share"] == 0.5
