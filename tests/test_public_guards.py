"""The three guards JEB-1623 put in front of a publicly reachable Pixel.

`https://pixel.yeldos.dev` answers without authentication, and twenty requests
are enough to spend the day's free-tier Gemini bucket. These tests pin the half
the app owns: a per-IP rate limit that stops short of the engine, a daily teacher
budget counted over the right window, and a token on the one endpoint a visitor
never needs.

The Access List in Nginx Proxy Manager is the real boundary and is out of scope
here — none of this authenticates anybody.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from backend.api import ADMIN_HEADER
from backend.main import app
from backend.ratelimit import RateLimiter, client_key
from backend.teacher import CAP_REPLY, CAP_STATUS
from backend.teacher.budget import (
    DEFAULT_DAILY_CAP,
    RESET_HOUR_UTC,
    bucket_start,
    calls_used,
    daily_cap,
)

GOOD_PLAN = json.dumps(
    {
        "reply": "Тада!",
        "handled": True,
        "actions": [{"action": "spin", "args": {}}],
    }
)


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(conn):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.fixture()
def missing(engine):
    """Laya reports a miss, which is the only door into the teacher."""
    engine.pick, engine.confidence = "unknown", 0.38
    return engine


def teacher_rows(conn) -> list:
    return conn.execute("SELECT * FROM teacher_log ORDER BY id").fetchall()


def reason(row) -> str:
    return json.loads(row["state_json"])["error"] or ""


# --- the rate limiter, in isolation -----------------------------------------


class Clock:
    """A monotonic clock a test advances by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_a_window_lets_the_limit_through_and_stops_the_next_one():
    limiter = RateLimiter(lambda: 3, window_s=60.0, clock=Clock())
    assert [limiter.take("ip") for _ in range(3)] == [None, None, None]
    assert limiter.take("ip") == 60


def test_retry_after_counts_down_to_the_oldest_hit_expiring():
    clock = Clock()
    limiter = RateLimiter(lambda: 1, window_s=60.0, clock=clock)
    assert limiter.take("ip") is None
    clock.now += 45
    # 15 s of the first hit's minute are left, and that is when a slot opens.
    assert limiter.take("ip") == 15
    clock.now += 16
    assert limiter.take("ip") is None


def test_each_key_gets_its_own_window():
    limiter = RateLimiter(lambda: 1, window_s=60.0, clock=Clock())
    assert limiter.take("1.1.1.1") is None
    assert limiter.take("2.2.2.2") is None
    assert limiter.take("1.1.1.1") is not None


def test_a_limit_of_zero_closes_the_endpoint():
    limiter = RateLimiter(lambda: 0, window_s=60.0, clock=Clock())
    assert limiter.take("ip") == 60


def test_the_limit_is_read_per_call():
    limit = [1]
    limiter = RateLimiter(lambda: limit[0], window_s=60.0, clock=Clock())
    assert limiter.take("ip") is None
    assert limiter.take("ip") is not None
    limit[0] = 5
    assert limiter.take("ip") is None


def test_expired_keys_are_collected():
    clock = Clock()
    limiter = RateLimiter(lambda: 5, window_s=60.0, clock=clock)
    for index in range(RateLimiter._SWEEP_AT + 1):
        limiter.take(f"10.0.0.{index}")
    clock.now += 61
    limiter.take("10.0.0.0")
    assert len(limiter._hits) == 1, "a per-IP map that only grows is a leak"


class FakeRequest:
    def __init__(self, headers: dict[str, str], host: str | None = "5.5.5.5") -> None:
        self.headers = {key.lower(): value for key, value in headers.items()}
        self.client = None if host is None else type("Client", (), {"host": host})()


@pytest.mark.parametrize(
    ("headers", "host", "expected"),
    [
        ({}, "5.5.5.5", "5.5.5.5"),
        ({"X-Forwarded-For": "1.2.3.4"}, "5.5.5.5", "1.2.3.4"),
        # A chain: the client is the first hop, the rest are proxies.
        ({"X-Forwarded-For": "1.2.3.4, 10.0.0.1, 10.0.0.2"}, "5.5.5.5", "1.2.3.4"),
        ({"X-Forwarded-For": "  1.2.3.4  "}, "5.5.5.5", "1.2.3.4"),
        # An empty header must not become one shared bucket for everybody.
        ({"X-Forwarded-For": ""}, "5.5.5.5", "5.5.5.5"),
        ({}, None, "unknown"),
    ],
)
def test_the_client_key_comes_from_the_first_forwarded_hop(headers, host, expected):
    assert client_key(FakeRequest(headers, host)) == expected


# --- the rate limiter, over HTTP --------------------------------------------


@pytest.mark.anyio
async def test_chat_over_the_limit_is_429_and_never_reaches_the_engine(
    client, seeded, engine, monkeypatch
):
    monkeypatch.setenv("CHAT_RATE_LIMIT", "2")
    engine.pick, engine.confidence = "greet", 0.95

    for _ in range(2):
        assert (await client.post("/api/chat", json={"text": "привет"})).status_code == 200
    engine.calls.clear()

    response = await client.post("/api/chat", json={"text": "привет"})

    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1
    # The whole point of checking first: past this line the request would take
    # the engine's single lock and possibly spend a unit of the daily bucket.
    assert engine.calls == []


@pytest.mark.anyio
async def test_a_second_address_is_not_blocked_by_the_first(client, seeded, engine, monkeypatch):
    monkeypatch.setenv("CHAT_RATE_LIMIT", "1")
    engine.pick, engine.confidence = "greet", 0.95
    ask = {"text": "привет"}

    first = await client.post("/api/chat", json=ask, headers={"X-Forwarded-For": "1.1.1.1"})
    blocked = await client.post("/api/chat", json=ask, headers={"X-Forwarded-For": "1.1.1.1"})
    other = await client.post("/api/chat", json=ask, headers={"X-Forwarded-For": "2.2.2.2"})

    assert first.status_code == 200
    assert blocked.status_code == 429
    assert other.status_code == 200


@pytest.mark.anyio
async def test_mine_has_its_own_tighter_limit(client, seeded, monkeypatch):
    monkeypatch.setenv("MINE_RATE_LIMIT", "1")
    assert (await client.post("/api/mine")).status_code == 200
    response = await client.post("/api/mine")
    assert response.status_code == 429
    assert "retry-after" in response.headers


@pytest.mark.anyio
async def test_the_default_limits_do_not_get_in_a_user_s_way(client, seeded, engine):
    # 20 a minute is the shipped default; a human typing commands must never see
    # a 429, and a test that asserts the guard exists must not assert it bites.
    engine.pick, engine.confidence = "greet", 0.95
    for _ in range(20):
        assert (await client.post("/api/chat", json={"text": "привет"})).status_code == 200


# --- the admin token on /api/mine -------------------------------------------


@pytest.mark.anyio
async def test_mine_is_open_when_no_token_is_configured(client, seeded, monkeypatch):
    monkeypatch.delenv("MINE_REQUIRE_TOKEN", raising=False)
    assert (await client.post("/api/mine")).status_code == 200


@pytest.mark.anyio
async def test_mine_without_the_header_is_403(client, seeded, monkeypatch):
    monkeypatch.setenv("MINE_REQUIRE_TOKEN", "s3cret")
    response = await client.post("/api/mine")
    assert response.status_code == 403


@pytest.mark.anyio
async def test_mine_with_the_wrong_header_is_403(client, seeded, monkeypatch):
    monkeypatch.setenv("MINE_REQUIRE_TOKEN", "s3cret")
    response = await client.post("/api/mine", headers={ADMIN_HEADER: "guess"})
    assert response.status_code == 403


@pytest.mark.anyio
async def test_mine_with_the_right_header_runs(client, seeded, monkeypatch):
    monkeypatch.setenv("MINE_REQUIRE_TOKEN", "s3cret")
    response = await client.post("/api/mine", headers={ADMIN_HEADER: "s3cret"})
    assert response.status_code == 200


@pytest.mark.anyio
async def test_a_non_latin_header_is_refused_not_crashed(client, seeded, monkeypatch):
    """JEB-1624: `compare_digest` on `str` raises on anything non-ASCII.

    The endpoint is public and unauthenticated, so any visitor could turn the
    guard into a `500` by sending a header the guard was supposed to just refuse.
    """
    monkeypatch.setenv("MINE_REQUIRE_TOKEN", "s3cret")
    response = await client.post("/api/mine", headers={ADMIN_HEADER: "тест".encode()})
    assert response.status_code == 403


@pytest.mark.anyio
async def test_a_non_latin_secret_still_lets_the_right_header_through(
    client, seeded, monkeypatch
):
    # The same trap from the other side: a non-latin MINE_REQUIRE_TOKEN used to
    # make *every* request a 500, the correct one included.
    monkeypatch.setenv("MINE_REQUIRE_TOKEN", "секрет")
    assert (await client.post("/api/mine", headers={ADMIN_HEADER: "секрет".encode()})).status_code == 200
    wrong = await client.post("/api/mine", headers={ADMIN_HEADER: "пароль".encode()})
    assert wrong.status_code == 403


# --- the daily teacher cap ---------------------------------------------------


def test_the_bucket_starts_at_the_free_tier_reset_hour():
    assert RESET_HOUR_UTC == 7, "midnight Pacific, which is when Google resets the tier"
    after = datetime(2026, 9, 25, 7, 0, tzinfo=UTC)
    assert bucket_start(after) == after
    assert bucket_start(after + timedelta(hours=16)) == after
    # One second before the reset still belongs to yesterday's bucket.
    assert bucket_start(after - timedelta(seconds=1)) == after - timedelta(days=1)


@pytest.mark.anyio
async def test_the_cap_counts_a_bucket_not_a_rolling_24h(client, seeded, missing, teacher):
    """JEB-1600's exact trap: a rolling window spans two buckets.

    A call from 20:00Z yesterday is inside the last 24 hours and outside today's
    bucket. Counting it would refuse a call the API would have accepted, and a
    busy evening would leave the next morning locked.
    """
    teacher(GOOD_PLAN)
    await client.post("/api/chat", json={"text": "покажи фокус"})

    row = seeded.execute("SELECT id, ts FROM interactions ORDER BY id DESC LIMIT 1").fetchone()
    stamped = datetime.fromisoformat(row["ts"])
    yesterday_evening = bucket_start(stamped) - timedelta(hours=3)
    seeded.execute(
        "UPDATE interactions SET ts = ? WHERE id = ?",
        (yesterday_evening.isoformat(), row["id"]),
    )
    seeded.commit()

    assert calls_used(seeded, stamped) == 0
    # ...and it is inside the rolling 24 hours the old number used, which is the
    # difference this test exists for.
    assert yesterday_evening > stamped - timedelta(hours=24)


@pytest.mark.anyio
async def test_past_the_cap_the_robot_says_so_and_never_calls_gemini(
    client, seeded, missing, teacher, monkeypatch
):
    monkeypatch.setenv("TEACHER_DAILY_CAP", "2")
    gemini = teacher(GOOD_PLAN)

    for _ in range(2):
        assert (await client.post("/api/chat", json={"text": "покажи фокус"})).json()[
            "reply"
        ] == "Тада!"
    assert len(gemini.calls) == 2

    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert body["reply"] == CAP_REPLY
    assert body["teacher_status"] == CAP_STATUS
    # Still `gemini`: the miss was routed to the teacher and the metrics count it
    # there, exactly as a 429 does (JEB-1603).
    assert body["engine"] == "gemini"
    assert len(gemini.calls) == 2, "a capped request must not reach the API"


@pytest.mark.anyio
async def test_the_cap_writes_a_reason_a_429_cannot_be_confused_with(
    client, seeded, missing, teacher, monkeypatch
):
    monkeypatch.setenv("TEACHER_DAILY_CAP", "1")
    teacher(GOOD_PLAN)
    await client.post("/api/chat", json={"text": "покажи фокус"})
    await client.post("/api/chat", json={"text": "спой песню"})

    rows = teacher_rows(seeded)
    assert len(rows) == 2, "the capped command is still logged — the miner wants to see it"
    assert reason(rows[0]) == ""
    assert CAP_STATUS in reason(rows[1])
    assert "429" not in reason(rows[1])
    assert "RESOURCE_EXHAUSTED" not in reason(rows[1])
    # And it is not mineable: a budget stop is not a skill.
    assert json.loads(rows[1]["state_json"])["handled"] is False


@pytest.mark.anyio
async def test_a_capped_row_is_not_charged_to_the_bucket(
    client, seeded, missing, teacher, monkeypatch
):
    # No request left the process for it, so counting it would drift this number
    # away from the one Google is keeping.
    monkeypatch.setenv("TEACHER_DAILY_CAP", "1")
    teacher(GOOD_PLAN)
    for _ in range(3):
        await client.post("/api/chat", json={"text": "покажи фокус"})

    assert len(teacher_rows(seeded)) == 3
    assert calls_used(seeded) == 1


@pytest.mark.anyio
async def test_under_the_cap_nothing_changes(client, seeded, missing, teacher):
    teacher(GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert body["reply"] == "Тада!"
    assert body["teacher_status"] is None


def test_the_shipped_cap_leaves_the_live_contract_gate_its_two_calls():
    # The free tier gives 20 per (project, model); the nightly gate (JEB-1552)
    # makes two live calls and must not be locked out by a day of ordinary use.
    assert DEFAULT_DAILY_CAP == 18
    assert daily_cap() == DEFAULT_DAILY_CAP
