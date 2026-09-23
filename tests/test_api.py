"""Contract tests — every one of the nine frozen endpoints, driven in-process."""

import httpx
import pytest

from backend.main import app
from backend.state import read_state, utcnow, write_state

STATE_KEYS = {"mood", "energy", "fullness", "face"}
REPLY_KEYS = {
    "interaction_id",
    "reply",
    "actions",
    "engine",
    "skill_id",
    "confidence",
    "latency_ms",
    "state",
}


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
async def client(conn):
    # The `conn` fixture already pointed the module-level connection at a tmp
    # database, so the app's lifespan is deliberately not run here.
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


def assert_reply(body: dict, engine: str) -> None:
    assert set(body) == REPLY_KEYS
    assert body["engine"] == engine
    assert isinstance(body["interaction_id"], int)
    assert isinstance(body["latency_ms"], int)
    assert set(body["state"]) == STATE_KEYS
    for action in body["actions"]:
        assert set(action) == {"action", "args"}


@pytest.mark.anyio
async def test_get_state(client):
    response = await client.get("/api/state")
    assert response.status_code == 200
    assert set(response.json()) == STATE_KEYS


@pytest.mark.anyio
async def test_feed_raises_fullness(client):
    before = (await client.get("/api/state")).json()["fullness"]
    response = await client.post("/api/action", json={"name": "feed"})
    assert response.status_code == 200
    body = response.json()
    assert_reply(body, "button")
    assert body["actions"]
    assert body["state"]["fullness"] > before
    assert body["state"]["face"] == "happy"


@pytest.mark.anyio
async def test_play_refuses_when_energy_is_low(client, conn):
    state = read_state(conn)
    state.energy = 10
    write_state(conn, state, utcnow())

    body = (await client.post("/api/action", json={"name": "play"})).json()
    assert [a["action"] for a in body["actions"]] == ["set_face", "say"]
    assert body["state"]["face"] == "sleepy"
    assert body["reply"] == "Я устал, давай позже"


@pytest.mark.anyio
async def test_unknown_button_is_rejected(client):
    assert (await client.post("/api/action", json={"name": "launch_rocket"})).status_code == 422


@pytest.mark.anyio
async def test_chat_stub(client):
    body = (await client.post("/api/chat", json={"text": "станцуй"})).json()
    assert_reply(body, "button")
    assert body["skill_id"] is None
    assert body["confidence"] is None


@pytest.mark.anyio
async def test_feedback(client):
    interaction_id = (await client.post("/api/action", json={"name": "pet"})).json()[
        "interaction_id"
    ]
    response = await client.post(
        "/api/feedback", json={"interaction_id": interaction_id, "value": 1}
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.anyio
async def test_feedback_unknown_interaction_is_404(client):
    response = await client.post("/api/feedback", json={"interaction_id": 999, "value": -1})
    assert response.status_code == 404


@pytest.mark.anyio
async def test_feedback_rejects_other_values(client):
    response = await client.post("/api/feedback", json={"interaction_id": 1, "value": 0})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_stage_later_endpoints_are_empty_but_present(client):
    assert (await client.get("/api/skills")).json() == []
    assert (await client.get("/api/proposals")).json() == []
    assert (await client.post("/api/proposals/abc/accept")).json() == {"ok": True}
    assert (await client.post("/api/proposals/abc/reject")).json() == {"ok": True}
    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}


@pytest.mark.anyio
async def test_metrics(client):
    await client.post("/api/action", json={"name": "feed"})
    body = (await client.get("/api/metrics")).json()
    assert set(body) == {
        "laya_share",
        "avg_latency_laya_ms",
        "avg_latency_gemini_ms",
        "skills_active",
        "total_commands",
    }
    assert body["total_commands"] == 1
    # No Laya and no Gemini yet — the share must not divide by zero.
    assert body["laya_share"] == 0.0
