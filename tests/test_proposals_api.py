"""Stage 4 end to end: pool -> cluster -> skill -> backtest -> proposal -> accept.

Everything here runs against `FakeEngine` and a scripted Gemini client, so it is
green without an API key and without weights. What it does *not* fake is the
pipeline: the real clustering, the real draft schema, the real backtest and the
real endpoints.
"""

import inspect
import json

import httpx
import pytest

from backend.brain.engine import set_engine
from backend.brain.skill import load_skills
from backend.main import app
from backend.miner import MineResult, mine_once, set_generator

from .fakes import FakeEngine
from .trick_cluster import (
    LATER_TRICK,
    TRICK_COMMANDS,
    TRICK_EMBEDDINGS,
    TRICK_ROUTES,
    draft_json,
    fill_pool,
)

PROPOSAL_KEYS = {"id", "skill", "match_rate", "sample_ids", "status", "created_at"}


@pytest.fixture()
def anyio_backend():
    return "asyncio"


@pytest.fixture()
def miner_engine():
    """The engine the miner and the app share, scripted for the trick cluster."""
    fake = FakeEngine(routes=TRICK_ROUTES, embeddings=TRICK_EMBEDDINGS)
    set_engine(fake)
    yield fake
    set_engine(None)


@pytest.fixture()
async def client(seeded):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


async def mined_proposal(client) -> dict:
    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 1}
    proposals = (await client.get("/api/proposals")).json()
    assert len(proposals) == 1
    return proposals[0]


@pytest.mark.anyio
async def test_five_similar_cases_become_exactly_one_proposal(
    client, seeded, miner_engine, generator
):
    generator(draft_json())
    fill_pool(seeded)

    proposal = await mined_proposal(client)
    assert set(proposal) == PROPOSAL_KEYS
    assert proposal["skill"]["id"] == "show_trick"
    assert proposal["skill"]["examples"] == TRICK_COMMANDS
    assert proposal["match_rate"] >= 0.8
    assert proposal["status"] == "pending"
    assert len(proposal["sample_ids"]) == 5


@pytest.mark.anyio
async def test_a_cluster_of_two_is_not_proposed(client, seeded, miner_engine, generator):
    generator(draft_json())
    fill_pool(seeded, TRICK_COMMANDS[:2])

    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    assert (await client.get("/api/proposals")).json() == []
    # And the cases stay in the pool — they ripen when more arrive.
    assert (
        seeded.execute("SELECT COUNT(*) AS n FROM teacher_log WHERE mined = 0").fetchone()["n"] == 2
    )


@pytest.mark.anyio
async def test_the_proposal_takes_its_cases_out_of_the_pool(
    client, seeded, miner_engine, generator
):
    generator(draft_json())
    fill_pool(seeded)
    proposal = await mined_proposal(client)

    rows = seeded.execute("SELECT mined, cluster_id FROM teacher_log").fetchall()
    assert {(row["mined"], row["cluster_id"]) for row in rows} == {(1, proposal["id"])}
    # Nothing left to mine, so a second run proposes nothing.
    assert (await client.post("/api/mine")).json()["proposals"] == 0


@pytest.mark.anyio
async def test_accepting_routes_the_command_to_the_new_skill_without_a_restart(
    client, seeded, miner_engine, generator, teacher
):
    generator(draft_json())
    teacher(
        json.dumps(
            {
                "reply": "Тада!",
                "actions": [
                    {"action": "spin"},
                    {"action": "set_face", "face": "happy"},
                    {"action": "say", "text": "Тада!"},
                ],
            },
            ensure_ascii=False,
        )
    )
    fill_pool(seeded)
    proposal = await mined_proposal(client)

    # Before: the skill does not exist, so the router misses and Gemini answers.
    before = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert before["engine"] == "gemini"

    assert (await client.post(f"/api/proposals/{proposal['id']}/accept")).status_code == 200

    # After: same process, same command, no Gemini.
    after = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert after["engine"] == "laya"
    assert after["skill_id"] == "show_trick"
    assert [step["action"] for step in after["actions"]] == ["spin", "set_face", "say"]


@pytest.mark.anyio
async def test_an_accepted_skill_lands_in_the_library_at_the_end(
    client, seeded, miner_engine, generator
):
    generator(draft_json())
    fill_pool(seeded)
    proposal = await mined_proposal(client)
    await client.post(f"/api/proposals/{proposal['id']}/accept")

    assert [skill.id for skill in load_skills(seeded)] == [
        "greet",
        "feed",
        "play",
        "sleep",
        "show_trick",
    ]
    card = next(
        row for row in (await client.get("/api/skills")).json() if row["id"] == "show_trick"
    )
    assert card["origin"] == "mined"
    assert card["status"] == "active"
    assert (await client.get("/api/proposals")).json() == []


@pytest.mark.anyio
async def test_the_starter_commands_still_reach_their_own_skills_after_accepting(
    client, seeded, miner_engine, generator
):
    generator(draft_json())
    fill_pool(seeded)
    proposal = await mined_proposal(client)
    await client.post(f"/api/proposals/{proposal['id']}/accept")

    for command, skill_id in (
        ("привет", "greet"),
        ("покорми", "feed"),
        ("поиграй", "play"),
        ("пора спать", "sleep"),
    ):
        body = (await client.post("/api/chat", json={"text": command})).json()
        assert (body["engine"], body["skill_id"]) == ("laya", skill_id)
        assert body["confidence"] >= 0.6


@pytest.mark.anyio
async def test_a_candidate_that_breaks_an_active_skill_is_never_proposed(client, seeded, generator):
    """The regression check, through the endpoint: nothing reaches the user."""
    thief = FakeEngine(
        routes={**TRICK_ROUTES, "покорми": "show_trick"}, embeddings=TRICK_EMBEDDINGS
    )
    set_engine(thief)
    try:
        generator(draft_json())
        fill_pool(seeded)
        assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
        assert (await client.get("/api/proposals")).json() == []
    finally:
        set_engine(None)


@pytest.mark.anyio
async def test_rejecting_returns_the_cases_and_does_not_propose_them_again(
    client, seeded, miner_engine, generator
):
    generator(draft_json())
    fill_pool(seeded)
    proposal = await mined_proposal(client)

    assert (await client.post(f"/api/proposals/{proposal['id']}/reject")).status_code == 200
    assert (await client.get("/api/proposals")).json() == []
    rows = seeded.execute("SELECT mined, cluster_id FROM teacher_log").fetchall()
    assert {(row["mined"], row["cluster_id"]) for row in rows} == {(0, None)}

    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    assert (await client.get("/api/proposals")).json() == []


@pytest.mark.anyio
async def test_a_rejected_cluster_is_mined_again_once_it_has_grown(
    client, seeded, miner_engine, generator
):
    """The signature is the exact case set, so new evidence reopens the question."""
    generator(draft_json())
    fill_pool(seeded)
    proposal = await mined_proposal(client)
    await client.post(f"/api/proposals/{proposal['id']}/reject")

    fill_pool(seeded, [LATER_TRICK])
    assert (await client.post("/api/mine")).json()["proposals"] == 1


@pytest.mark.anyio
async def test_accepting_or_rejecting_twice_is_a_404(client, seeded, miner_engine, generator):
    generator(draft_json())
    fill_pool(seeded)
    proposal = await mined_proposal(client)

    assert (await client.post(f"/api/proposals/{proposal['id']}/accept")).status_code == 200
    assert (await client.post(f"/api/proposals/{proposal['id']}/accept")).status_code == 404
    assert (await client.post(f"/api/proposals/{proposal['id']}/reject")).status_code == 404
    assert (await client.post("/api/proposals/nope/reject")).status_code == 404


@pytest.mark.anyio
async def test_a_candidate_colliding_with_an_existing_skill_is_not_proposed(
    client, seeded, miner_engine, generator
):
    generator(draft_json(id="feed"))
    fill_pool(seeded)
    assert (await client.post("/api/mine")).json()["proposals"] == 0


@pytest.mark.anyio
async def test_an_invented_primitive_never_reaches_a_proposal(
    client, seeded, miner_engine, generator
):
    fake = generator(
        draft_json(
            rules=[
                {
                    "when_state": "",
                    "when_band": "",
                    "actions": [{"action": "launch_rocket"}, {"action": "say", "text": "Пуск!"}],
                }
            ]
        )
    )
    fill_pool(seeded)
    assert (await client.post("/api/mine")).json()["proposals"] == 0
    assert (await client.get("/api/proposals")).json() == []
    # One try plus one retry, then the cluster waits for the next run.
    assert len(fake.calls) == 2


@pytest.mark.anyio
async def test_a_second_attempt_is_given_the_reason_the_first_failed(
    client, seeded, miner_engine, generator
):
    fake = generator(draft_json(description="ц" * 200), draft_json())
    fill_pool(seeded)
    assert (await client.post("/api/mine")).json()["proposals"] == 1
    assert "Предыдущий ответ не прошёл проверку" in fake.calls[1]["contents"]


@pytest.mark.anyio
async def test_the_generator_is_asked_with_the_miner_model_and_the_skill_schema(
    client, seeded, miner_engine, generator
):
    fake = generator(draft_json())
    fill_pool(seeded)
    await client.post("/api/mine")

    call = fake.calls[0]
    assert call["model"] == "fake-model"
    assert call["config"]["response_mime_type"] == "application/json"
    assert set(call["config"]["response_schema"]["properties"]) == {
        "id",
        "name",
        "description",
        "examples",
        "rules",
    }
    # The cluster and the teacher's plans both go in — the plans are what make
    # the generated rules resemble what the teacher actually did.
    for command in TRICK_COMMANDS:
        assert command in call["contents"]
    assert "spin(), set_face(happy)" in call["contents"]


def test_the_sdk_still_has_the_generate_content_surface_we_call():
    """`FakeGeminiClient` takes any keyword, so only the real SDK can say whether
    `GeminiSkillGenerator._call` is calling anything that exists.

    The miner is on `models.generate_content` and not the teacher's
    `interactions.create` because `models/gemini-2.5-flash-lite` fences its JSON
    on the latter. A renamed config field would ride along silently, the answer
    would come back unschema'd, and every draft would die in `_parse` with only
    a `log.warning` behind it. Needs no network and no real key.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key="not-a-real-key-and-never-sent")
    parameters = inspect.signature(type(client.models).generate_content).parameters
    assert {"model", "contents", "config"} <= set(parameters)

    config_fields = set(types.GenerateContentConfig.model_fields)
    assert {
        "system_instruction",
        "response_mime_type",
        "response_schema",
        "http_options",
    } <= config_fields
    # The timeout lives here on this path, in milliseconds.
    assert "timeout" in types.HttpOptions.model_fields
    assert hasattr(types.GenerateContentResponse, "text")


@pytest.mark.anyio
async def test_mining_runs_itself_every_batch_th_case(
    client, seeded, miner_engine, generator, teacher, monkeypatch
):
    """No button pressed: the fifth miss triggers the run that proposes."""
    monkeypatch.setenv("MINER_BATCH", "5")
    generator(draft_json())
    teacher(json.dumps({"reply": "Тада!", "actions": [{"action": "spin"}]}, ensure_ascii=False))

    fill_pool(seeded, TRICK_COMMANDS[:4])
    assert (await client.get("/api/proposals")).json() == []

    # The fifth case arrives the way a real one does — through /api/chat.
    await client.post("/api/chat", json={"text": TRICK_COMMANDS[4]})
    assert len((await client.get("/api/proposals")).json()) == 1


@pytest.mark.anyio
async def test_mine_on_an_empty_pool_is_200_not_500(client):
    assert (await client.post("/api/mine")).status_code == 200
    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    assert (await client.get("/api/proposals")).json() == []


@pytest.mark.anyio
async def test_mine_without_an_api_key_is_honest_about_doing_nothing(client, seeded, miner_engine):
    """No generator installed is exactly the key-less deployment."""
    fill_pool(seeded)
    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}


@pytest.mark.anyio
async def test_a_concurrent_run_is_refused_not_queued(client, seeded, miner_engine, generator):
    seen = []

    class Reentrant:
        def group(self, texts):
            return []

        def propose(self, cases, skills):
            # Re-entering from inside a run is what the Lock has to refuse.
            seen.append(mine_once())

    generator(draft_json())
    set_generator(Reentrant())
    fill_pool(seeded)

    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    assert seen == [MineResult(started=False, proposals=0)]
