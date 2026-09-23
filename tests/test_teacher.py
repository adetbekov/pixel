"""Stage 3: the teacher, driven end to end against a scripted Gemini client.

Nothing here touches the network. What is actually being tested is the promise
:mod:`backend.teacher.client` makes — that *whatever* comes back from the model,
the caller gets a plan made of library actions and the case lands in
``teacher_log``.
"""

import inspect
import json
import typing

import httpx
import pytest

from backend.actions import ACTIONS, MAX_SAY_LEN
from backend.api import MAX_CHAT_TEXT
from backend.brain.skill import load_skills
from backend.main import app
from backend.state import read_state
from backend.teacher import FALLBACK_PLAN, GeminiTeacher
from backend.teacher.client import (
    FALLBACK_REPLY,
    TIMEOUT_S,
    TOTAL_DEADLINE_S,
)
from backend.teacher.prompt import MAX_TEXT_LEN, build_input
from backend.teacher.schema import TeacherPlan

GOOD_PLAN = json.dumps(
    {
        "reply": "Смотри, что я умею!",
        "actions": [
            {"action": "set_face", "face": "happy"},
            {"action": "spin"},
            {"action": "jump"},
            {"action": "say", "text": "Тада!"},
        ],
    },
    ensure_ascii=False,
)

HACKED_PLAN = json.dumps(
    {"reply": "Сейчас!", "actions": [{"action": "hack_nasa"}, {"action": "jump"}]},
    ensure_ascii=False,
)

BLANK_REPLY_PLAN = json.dumps({"reply": "   ", "actions": [{"action": "jump"}]})


class TimingOutClient:
    """A client whose every call burns its whole budget and then times out.

    Drives `GeminiTeacher`'s injected clock, so the deadline arithmetic is
    exercised at full speed instead of in real seconds.
    """

    def __init__(self, overshoot: float = 0.0) -> None:
        self.now = 0.0
        self.budgets: list[float] = []
        self.interactions = self
        self._overshoot = overshoot

    def create(self, **kwargs) -> None:
        self.budgets.append(kwargs["timeout"])
        self.now += kwargs["timeout"] + self._overshoot
        raise httpx.TimeoutException("deadline exceeded")


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


@pytest.mark.anyio
async def test_a_miss_is_answered_by_gemini(client, conn, seeded, missing, teacher):
    gemini = teacher(GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert body["engine"] == "gemini"
    assert body["skill_id"] is None
    assert body["confidence"] == pytest.approx(0.38)
    assert [a["action"] for a in body["actions"]] == ["set_face", "spin", "jump", "say"]
    assert body["reply"] == "Тада!"
    assert body["state"]["face"] == "happy"
    assert len(gemini.calls) == 1


@pytest.mark.anyio
async def test_every_planned_action_belongs_to_the_library(client, seeded, missing, teacher):
    teacher(GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert all(action["action"] in ACTIONS for action in body["actions"])


@pytest.mark.anyio
async def test_a_hit_never_reaches_the_teacher(client, conn, seeded, engine, teacher):
    gemini = teacher(GOOD_PLAN)
    engine.pick, engine.confidence = "greet", 0.95

    body = (await client.post("/api/chat", json={"text": "привет"})).json()

    assert body["engine"] == "laya"
    assert gemini.calls == []
    # The absence of a row is the assertion: a skill Laya handles must not cost
    # a Gemini call, and `teacher_log` is where that would show up.
    assert teacher_rows(conn) == []


@pytest.mark.anyio
async def test_an_invented_action_is_retried_then_falls_back(
    client, conn, seeded, missing, teacher
):
    gemini = teacher(HACKED_PLAN)

    body = (await client.post("/api/chat", json={"text": "взломай насу"})).json()

    assert len(gemini.calls) == 2, "an invalid plan must be retried exactly once"
    assert [a["action"] for a in body["actions"]] == [s["action"] for s in FALLBACK_PLAN]
    assert body["reply"] == FALLBACK_REPLY
    assert body["engine"] == "gemini"
    # The retry has to tell the model what was wrong, or it repeats itself.
    assert "hack_nasa" not in json.dumps(body)
    assert "недопустимое действие" in gemini.calls[1]["input"]


@pytest.mark.anyio
async def test_a_retry_that_succeeds_is_used(client, seeded, missing, teacher):
    teacher(HACKED_PLAN, GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert body["reply"] == "Тада!"


@pytest.mark.anyio
async def test_a_timeout_falls_back_without_a_traceback(client, conn, seeded, missing, teacher):
    gemini = teacher(httpx.TimeoutException("deadline exceeded"))

    response = await client.post("/api/chat", json={"text": "покажи фокус"})

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == FALLBACK_REPLY
    assert body["engine"] == "gemini"
    assert body["latency_ms"] >= 0
    assert len(gemini.calls) == 2, "a network failure gets one retry"
    assert "TimeoutException" in json.loads(teacher_rows(conn)[0]["state_json"])["error"]


@pytest.mark.anyio
async def test_unparsable_json_falls_back(client, seeded, missing, teacher):
    teacher("я не джейсон")
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert body["reply"] == FALLBACK_REPLY


@pytest.mark.anyio
async def test_every_call_leaves_a_complete_log_row(client, conn, seeded, missing, teacher):
    teacher(GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    rows = teacher_rows(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["interaction_id"] == body["interaction_id"]
    assert row["mined"] == 0
    assert row["cluster_id"] is None
    assert row["raw_response"] == GOOD_PLAN

    state = json.loads(row["state_json"])
    assert state["user_text"] == "покажи фокус"
    assert state["router_confidence"] == pytest.approx(0.38)
    assert set(state["state"]) == {"mood", "energy", "fullness", "face"}
    assert state["error"] is None

    assert json.loads(row["actions_json"]) == body["actions"]


@pytest.mark.anyio
async def test_a_fallback_is_logged_too(client, conn, seeded, missing, teacher):
    """The miner needs the cases Gemini could not answer either."""
    teacher(HACKED_PLAN)
    await client.post("/api/chat", json={"text": "взломай насу"})

    row = teacher_rows(conn)[0]
    assert "hack_nasa" in json.loads(row["state_json"])["error"]
    assert json.loads(row["actions_json"]) == FALLBACK_PLAN
    # The raw text is kept for the miner but must never leave over the API.
    assert row["raw_response"] == HACKED_PLAN


@pytest.mark.anyio
async def test_without_a_key_the_miss_answers_with_the_stub(client, conn, seeded, missing):
    """No `teacher` fixture — exactly the shape of a deployment with no API key."""
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert body["engine"] == "laya"
    assert body["reply"] == "Я пока не понял"
    assert teacher_rows(conn) == []


@pytest.mark.anyio
async def test_metrics_count_teacher_calls(client, seeded, missing, teacher):
    teacher(GOOD_PLAN)
    assert (await client.get("/api/metrics")).json()["teacher_calls_24h"] == 0

    await client.post("/api/chat", json={"text": "покажи фокус"})

    body = (await client.get("/api/metrics")).json()
    assert body["teacher_calls_24h"] == 1
    assert body["laya_share"] == 0.0
    assert body["avg_latency_gemini_ms"] > 0


def test_the_prompt_trims_a_long_command(seeded):
    """`/api/chat` already rejects anything longer (`MAX_CHAT_TEXT`), so this is
    the second lock on the same door — the teacher must not be reachable with an
    unbounded prompt if it ever gets a caller other than `post_chat`."""
    prompt = build_input("а" * 5000, read_state(seeded), [])
    assert "а" * MAX_TEXT_LEN in prompt
    assert "а" * (MAX_TEXT_LEN + 1) not in prompt


def test_the_two_text_limits_agree():
    # The API cap and the prompt trim are declared separately — `backend.api`
    # imports the teacher, so the teacher cannot import back. Pin them together.
    assert MAX_TEXT_LEN == MAX_CHAT_TEXT


@pytest.mark.anyio
async def test_the_request_asks_for_the_plan_schema(client, seeded, missing, teacher):
    gemini = teacher(GOOD_PLAN)
    await client.post("/api/chat", json={"text": "покажи фокус"})

    call = gemini.calls[0]
    assert call["model"] == "fake-model"
    assert call["timeout"] == TIMEOUT_S
    assert call["response_format"]["mime_type"] == "application/json"
    assert set(call["response_format"]["schema"]["properties"]) == {"reply", "actions"}
    # The prompt is generated from the library, so it cannot drift from it.
    assert all(name in call["system_instruction"] for name in ACTIONS)


def test_the_prompt_shows_the_skill_boundary(seeded):
    skills = load_skills(seeded)
    prompt = build_input("покажи фокус", read_state(seeded), skills)

    # Exactly one line per skill, `id: description` and nothing else. `examples`
    # belong to the router; repeating them here only dilutes the boundary the
    # miner is meant to read off this list.
    listed = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert listed == [f"- {skill.id}: {skill.description}" for skill in skills]


def test_an_over_long_reply_is_trimmed_not_rejected():
    plan = TeacherPlan.model_validate({"reply": "я" * 400, "actions": []})
    assert plan.reply == "я" * MAX_SAY_LEN


@pytest.mark.anyio
@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
async def test_a_blank_reply_never_reaches_the_user(client, seeded, missing, teacher, blank):
    """A blank `reply` with no `say` step would render as an empty bubble.

    `""` is stopped by `min_length` in the schema and `"   "` only by the strip
    check in `_parse` — both must end up retried, then on the fallback.
    """
    gemini = teacher(json.dumps({"reply": blank, "actions": [{"action": "jump"}]}))

    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert len(gemini.calls) == 2, "a blank reply must be retried once"
    assert body["reply"] == FALLBACK_REPLY
    assert body["reply"].strip()


@pytest.mark.anyio
async def test_a_blank_reply_retry_that_succeeds_is_used(client, seeded, missing, teacher):
    teacher(BLANK_REPLY_PLAN, GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert body["reply"] == "Тада!"


def test_the_retry_gets_only_the_time_that_is_left(seeded):
    """Two timeouts must not cost the user 2 x TIMEOUT_S on top of the router miss.

    Driven by an injected clock that a call advances by exactly the budget it
    was handed — which is what a call that times out does — so the arithmetic is
    asserted without a test that actually waits.
    """
    gemini = TimingOutClient()
    teacher = GeminiTeacher(client=gemini, model="fake-model", clock=lambda: gemini.now)

    result = teacher.explain("покажи фокус", read_state(seeded), [])

    budgets = gemini.budgets
    assert len(budgets) == 2, "a timeout still gets its one retry"
    assert budgets[0] == TIMEOUT_S
    assert budgets[1] == pytest.approx(TOTAL_DEADLINE_S - TIMEOUT_S)
    assert sum(budgets) <= TOTAL_DEADLINE_S
    assert result.raw_plan == FALLBACK_PLAN


def test_no_retry_is_dialled_with_nothing_left_to_spend(seeded):
    gemini = TimingOutClient(overshoot=TOTAL_DEADLINE_S)
    teacher = GeminiTeacher(client=gemini, model="fake-model", clock=lambda: gemini.now)

    result = teacher.explain("покажи фокус", read_state(seeded), [])

    assert len(gemini.budgets) == 1, "a second call with < MIN_CALL_BUDGET_S left is not worth it"
    assert "out of time" in result.error
    assert result.raw_plan == FALLBACK_PLAN


def test_the_sdk_still_has_the_surface_we_call():
    """The one check CI could not make before: that the *real* SDK takes what
    `GeminiTeacher._call` passes it.

    `FakeGeminiClient` accepts any keyword, so every other test here proves only
    that the code *sends* the argument. If `timeout` were not a real parameter
    the live call would raise `TypeError`, `explain` would swallow it, and every
    router miss would quietly degrade to `FALLBACK_PLAN` with CI still green.
    Constructing the client needs no network and no real key.
    """
    from google import genai
    from google.genai import interactions

    client = genai.Client(api_key="not-a-real-key-and-never-sent")
    parameters = inspect.signature(type(client.interactions).create).parameters

    # `timeout` must be its OWN parameter. Were it merely absorbed by `**body`
    # it would ride along to the API as a request field and time out nothing.
    assert parameters["timeout"].kind is inspect.Parameter.KEYWORD_ONLY
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    # `**body` swallows anything, so the signature cannot vouch for these —
    # the request TypedDict is what names them.
    body_fields = typing.get_type_hints(interactions.CreateModelInteractionParamsNonStreaming)
    assert {"model", "input", "system_instruction", "response_format"} <= set(body_fields)

    assert "output_text" in interactions.Interaction.model_fields
    assert {"mime_type", "schema_"} <= set(interactions.TextResponseFormat.model_fields)
