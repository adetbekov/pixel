"""Stage 4 end to end: pool -> cluster -> skill -> backtest -> proposal -> accept.

Everything here runs against `FakeEngine` and a scripted Gemini client, so it is
green without an API key and without weights. What it does *not* fake is the
pipeline: the real clustering, the real draft schema, the real backtest and the
real endpoints.
"""

import inspect
import json
import math
import sqlite3

import httpx
import pytest

from backend import db
from backend.brain.engine import set_engine
from backend.brain.skill import load_skills
from backend.main import app
from backend.miner import MineResult, mine_once, set_generator
from backend.miner.generate import TIMEOUT_S, GeminiSkillGenerator
from backend.teacher.client import MIN_SERVER_DEADLINE_S

from .fakes import FakeEngine, FakeGeminiClient
from .trick_cluster import (
    LATER_TRICK,
    SEED_CONTROLS,
    TRICK_COMMANDS,
    TRICK_EMBEDDINGS,
    TRICK_ROUTES,
    draft_json,
    fill_pool,
)

PROPOSAL_KEYS = {
    "id",
    "skill",
    "match_rate",
    "generalization",
    "sample_ids",
    "status",
    "created_at",
}


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
async def test_the_card_gets_both_numbers_and_they_can_disagree(client, seeded, generator):
    """`generalization` is the number that varies, so the endpoint has to carry it.

    The head here answers three of the five trick phrases and misses the other
    two, while every one of them is listed in the draft's `examples` — which is
    the live shape (JEB-1562): `match_rate` 1.00 because step 0 answers, and a
    `generalization` well under it because an unlisted phrasing would miss.
    """
    shallow = FakeEngine(
        routes={**dict.fromkeys(TRICK_COMMANDS[:3], "show_trick"), **SEED_CONTROLS},
        embeddings=TRICK_EMBEDDINGS,
    )
    set_engine(shallow)
    try:
        generator(draft_json())
        fill_pool(seeded)
        proposal = await mined_proposal(client)
    finally:
        set_engine(None)

    assert proposal["match_rate"] == pytest.approx(1.0)
    assert proposal["generalization"] == pytest.approx(0.6)


@pytest.mark.anyio
async def test_a_proposal_mined_before_the_column_existed_reads_as_unknown(client, seeded):
    """Legacy rows have no `generalization`; `null` is the answer, not a zero."""
    seeded.execute(
        "INSERT INTO skill_proposals (id, skill_json, match_rate, sample_ids, status, created_at)"
        " VALUES ('old', '{\"id\": \"show_trick\"}', 1.0, '[1]', 'pending', '2026-09-01T00:00:00Z')"
    )
    seeded.commit()

    proposals = (await client.get("/api/proposals")).json()
    assert [row["generalization"] for row in proposals] == [None]


def test_an_old_db_gains_the_generalization_column(tmp_path):
    """The column is a migration, so a `pixel.db` from an earlier run upgrades."""
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE skill_proposals (id TEXT PRIMARY KEY, skill_json TEXT, match_rate REAL,"
        " sample_ids TEXT, status TEXT, created_at TEXT)"
    )
    old.execute(
        "INSERT INTO skill_proposals (id, skill_json, match_rate, sample_ids, status, created_at)"
        " VALUES ('old', '{}', 1.0, '[1]', 'pending', '2026-09-01T00:00:00Z')"
    )
    old.commit()
    old.close()

    conn = db.connect(path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(skill_proposals)")}
        assert "generalization" in columns
        row = conn.execute("SELECT generalization FROM skill_proposals").fetchone()
        assert row["generalization"] is None
    finally:
        conn.close()


@pytest.mark.anyio
async def test_a_cluster_of_two_is_not_proposed(client, seeded, miner_engine, generator, caplog):
    generator(draft_json())
    fill_pool(seeded, TRICK_COMMANDS[:2])

    with caplog.at_level("INFO", logger="backend.miner.run"):
        assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    assert (await client.get("/api/proposals")).json() == []
    # And the cases stay in the pool — they ripen when more arrive.
    assert (
        seeded.execute("SELECT COUNT(*) AS n FROM teacher_log WHERE mined = 0").fetchone()["n"] == 2
    )
    # Said out loud, because this is the commonest way a run ends at zero and an
    # empty log is indistinguishable from a broken miner (JEB-1593).
    assert "the pool holds 2 cases, fewer than MINER_MIN_CLUSTER (3)" in caplog.text


@pytest.mark.anyio
async def test_a_pool_the_grouper_split_too_small_says_so(
    client, seeded, miner_engine, generator, caplog
):
    """The JEB-1593 shape: enough cases, and every group under the bar.

    Measured on the live grouper, "сделай сальто" / "умеешь сальто?" / "сальто
    назад" came back as three groups of 2, 2 and 1 — a run that answers
    `{"started": true, "proposals": 0}`, writes nothing to the log, and cannot be
    told apart from a draft that was rejected.
    """
    generator(draft_json(), grouping=[[0, 1], [2, 3], [4]])
    fill_pool(seeded)

    with caplog.at_level("INFO", logger="backend.miner.run"):
        assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    ids = [row["id"] for row in seeded.execute("SELECT id FROM teacher_log ORDER BY id")]
    assert f"cluster {[ids[0], ids[1]]} is smaller than MINER_MIN_CLUSTER (2 < 3)" in caplog.text
    assert f"cluster {[ids[4]]} is smaller than MINER_MIN_CLUSTER (1 < 3)" in caplog.text


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
                "handled": True,
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
    config = call["config"]
    assert call["model"] == "fake-model"
    assert config["response_mime_type"] == "application/json"
    assert set(config["response_schema"]["properties"]) == {
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


@pytest.mark.anyio
async def test_the_miner_budget_reaches_the_api_in_milliseconds_and_as_a_floor(
    client, seeded, miner_engine, generator
):
    """The two numbers JEB-1540 exists to get right.

    `http_options.timeout` is milliseconds on this path — passing `TIMEOUT_S`
    raw would cut every miner call to 30 ms. And `X-Server-Timeout` is a floor,
    not `MIN_SERVER_DEADLINE_S`: the miner's budget is 30 s, three times the
    API's minimum, so pinning the header at 10 would have the server abandon a
    call httpx is still waiting out — the one place the teacher's constant would
    have been wrong here.
    """
    fake = generator(draft_json())
    fill_pool(seeded)
    await client.post("/api/mine")

    http_options = fake.calls[0]["config"]["http_options"]
    assert http_options["timeout"] == int(TIMEOUT_S * 1000)
    assert TIMEOUT_S > MIN_SERVER_DEADLINE_S, "otherwise this test proves nothing"
    assert http_options["headers"]["X-Server-Timeout"] == str(math.ceil(TIMEOUT_S))


def test_the_sdk_still_has_the_surface_the_miner_calls():
    """That the *real* SDK takes what `GeminiSkillGenerator._call` passes it.

    `FakeGeminiClient` accepts any keyword, so every other test here proves only
    that the code *sends* the argument. `config` is a plain dict, so a renamed
    field would not even raise — it would ride along to the API, the draft would
    come back fenced, `propose` would return `None`, and the miner would go
    silent with CI still green. That is exactly the defect JEB-1513 found in the
    teacher and this task's sibling of it. Constructing the client needs no
    network and no real key.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key="not-a-real-key-and-never-sent")
    parameters = inspect.signature(type(client.models).generate_content).parameters
    assert {"model", "contents", "config"} <= set(parameters)

    assert {
        "system_instruction",
        "response_mime_type",
        "response_schema",
        "http_options",
    } <= set(types.GenerateContentConfig.model_fields)
    assert {"timeout", "headers"} <= set(types.HttpOptions.model_fields)
    assert hasattr(types.GenerateContentResponse, "text")

    # The floor only holds because the SDK fills `X-Server-Timeout` from the
    # timeout *only* when we have not set it ourselves.
    from google.genai._api_client import populate_server_timeout_header

    headers = {"X-Server-Timeout": "30"}
    populate_server_timeout_header(headers, 30.0)
    assert headers["X-Server-Timeout"] == "30"


@pytest.mark.anyio
async def test_a_fenced_draft_is_not_parsed(client, seeded, miner_engine, generator, caplog):
    """The fences stay a failure here too, deliberately.

    ` ```json ` around an otherwise valid draft is what `interactions.create`
    returned on `models/gemini-2.5-flash-lite`, and it is the *whole* visible
    symptom of the wrong call shape on this path: offline there is no timeout
    and no error page, only a cluster that quietly goes back in the pool and a
    `/api/proposals` that stays empty forever. So a fenced answer must still be
    retried and still end without a proposal — stripping the fence in `_parse`
    would hide a regression back onto `interactions.create` instead of failing
    on it, which is why this test exists rather than a lenient parser.

    The mirror of `tests/test_teacher.py::test_a_fenced_response_is_not_parsed`.
    """
    fake = generator(f"```json\n{draft_json()}\n```")
    fill_pool(seeded)

    with caplog.at_level("WARNING", logger="backend.miner.generate"):
        assert (await client.post("/api/mine")).json()["proposals"] == 0
    assert (await client.get("/api/proposals")).json() == []

    assert len(fake.calls) == 2, "a fenced draft is retried like any other bad response"
    assert "did not match the schema" in caplog.text


def test_the_grouping_call_goes_out_the_same_way():
    """`group()` shares `_call`, so it shared the defect — and hides it better.

    Grouping is the miner's first call and its failures are silent: `group`
    swallows the parse error and returns `[]`, which reads as "no clusters"
    rather than as a failure, and the run then falls back to the local vectors.
    Pin both halves — the shape that goes out, and that a bare-JSON answer comes
    back parsed.
    """
    fake = FakeGeminiClient(grouping=[[0, 1], [2]])
    generator = GeminiSkillGenerator(client=fake, model="fake-model")

    assert generator.group(["покажи фокус", "сделай фокус", "станцуй"]) == [[0, 1], [2]]

    call = fake.grouping_calls[0]
    assert call["model"] == "fake-model"
    assert call["config"]["response_mime_type"] == "application/json"
    assert "groups" in call["config"]["response_schema"]["properties"]
    assert "покажи фокус" in call["contents"]


def test_a_fenced_grouping_is_not_parsed():
    """And the fence is a failure on this path too — an empty, silent one."""
    fake = FakeGeminiClient(grouping='```json\n{"groups": [[0, 1]]}\n```')
    generator = GeminiSkillGenerator(client=fake, model="fake-model")

    assert generator.group(["покажи фокус", "сделай фокус"]) == []


@pytest.mark.anyio
async def test_the_same_draft_unfenced_is_proposed(client, seeded, miner_engine, generator):
    """The control for the test above: the fence is the only thing wrong."""
    generator(draft_json())
    fill_pool(seeded)
    assert (await client.post("/api/mine")).json()["proposals"] == 1


@pytest.mark.anyio
async def test_mining_runs_itself_every_batch_th_case(
    client, seeded, miner_engine, generator, teacher, monkeypatch
):
    """No button pressed: the fifth miss triggers the run that proposes."""
    monkeypatch.setenv("MINER_BATCH", "5")
    generator(draft_json())
    teacher(
        json.dumps(
            {"reply": "Тада!", "handled": True, "actions": [{"action": "spin"}]},
            ensure_ascii=False,
        )
    )

    fill_pool(seeded, TRICK_COMMANDS[:4])
    assert (await client.get("/api/proposals")).json() == []

    # The fifth case arrives the way a real one does — through /api/chat.
    await client.post("/api/chat", json={"text": TRICK_COMMANDS[4]})
    assert len((await client.get("/api/proposals")).json()) == 1


@pytest.mark.anyio
async def test_a_refusal_does_not_re_trigger_a_parked_pool(
    client, seeded, miner_engine, generator, teacher, monkeypatch
):
    """The pool sits on a multiple of MINER_BATCH and a refusal arrives.

    `mined = 1` is set only when a proposal is saved, so a cluster that fails its
    backtest stays in the pool for good — and a refusal does not move the pool at
    all. On a pool parked at exactly MINER_BATCH, a trigger that read the level
    instead of the arrival would fire on *every* later "какая погода": one
    synchronous `generator.propose` round trip per declined message, inside a
    request the user is waiting on (JEB-1547 review).
    """
    monkeypatch.setenv("MINER_BATCH", "5")
    fake = generator(draft_json())
    teacher(
        json.dumps(
            {"reply": "Я не умею предсказывать погоду", "handled": False, "actions": []},
            ensure_ascii=False,
        )
    )

    fill_pool(seeded, TRICK_COMMANDS)
    assert fake.calls == [], "nothing has run the miner yet"

    for text in ("какая погода", "закажи пиццу", "который час"):
        await client.post("/api/chat", json={"text": text})

    assert fake.calls == [], "a refusal must not pay for a mining run"
    assert (await client.get("/api/proposals")).json() == []


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


@pytest.mark.anyio
async def test_the_run_asks_the_teacher_to_group_before_anything_else(
    client, seeded, miner_engine, generator
):
    """JEB-1548: one grouping call per run, and the vectors are not consulted."""
    fake = generator(draft_json(), grouping=[[0, 1, 2, 3, 4]])
    fill_pool(seeded)

    proposal = await mined_proposal(client)
    assert len(fake.grouping_calls) == 1
    assert TRICK_COMMANDS[0] in fake.grouping_calls[0]["contents"]
    assert miner_engine.embedded == []
    assert len(proposal["sample_ids"]) == 5


@pytest.mark.anyio
async def test_a_command_the_teacher_left_out_kills_an_over_broad_candidate(
    client, seeded, miner_engine, generator
):
    """The rest of the pool is the control set no `examples[0]` check can be."""
    generator(draft_json(), grouping=[[0, 1, 2, 3, 4], [5]])
    fill_pool(seeded, [*TRICK_COMMANDS, "покажи сальто"])
    miner_engine.routes = {**TRICK_ROUTES, "покажи сальто": "show_trick"}

    assert (await client.post("/api/mine")).json() == {"started": True, "proposals": 0}
    assert (await client.get("/api/proposals")).json() == []
    # Rejected, not mined: the cases stay in the pool for a narrower draft later.
    assert (
        seeded.execute("SELECT COUNT(*) AS n FROM teacher_log WHERE mined = 0").fetchone()["n"] == 6
    )
