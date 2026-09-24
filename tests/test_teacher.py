"""Stage 3: the teacher, driven end to end against a scripted Gemini client.

Nothing here touches the network. What is actually being tested is the promise
:mod:`backend.teacher.client` makes — that *whatever* comes back from the model,
the caller gets a plan made of library actions and the case lands in
``teacher_log``.
"""

import inspect
import json

import httpx
import pytest

from backend.actions import ACTIONS, MAX_SAY_LEN
from backend.api import MAX_CHAT_TEXT
from backend.brain.skill import load_skills
from backend.main import app
from backend.miner.case import load_pool
from backend.state import read_state
from backend.teacher import FALLBACK_PLAN, QUOTA_PLAN, QUOTA_REPLY, GeminiTeacher
from backend.teacher.client import (
    DEFAULT_MODEL,
    FALLBACK_REPLY,
    MIN_SERVER_DEADLINE_S,
    QUOTA_ERROR,
    TIMEOUT_S,
    TOTAL_DEADLINE_S,
)
from backend.teacher.prompt import MAX_TEXT_LEN, build_input
from backend.teacher.schema import TeacherPlan

GOOD_PLAN = json.dumps(
    {
        "reply": "Смотри, что я умею!",
        "handled": True,
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
    {
        "reply": "Сейчас!",
        "handled": True,
        "actions": [{"action": "hack_nasa"}, {"action": "jump"}],
    },
    ensure_ascii=False,
)

BLANK_REPLY_PLAN = json.dumps(
    {"reply": "   ", "handled": True, "actions": [{"action": "jump"}]}
)

#: The teacher answered, and declined: a real plan, a real reply, `handled` false.
DECLINED_PLAN = json.dumps(
    {
        "reply": "Я не умею заказывать еду, научи меня чему-нибудь другому",
        "handled": False,
        "actions": [
            {"action": "set_face", "face": "curious"},
            {"action": "say", "text": "Я не умею заказывать еду"},
        ],
    },
    ensure_ascii=False,
)


class TimingOutClient:
    """A client whose every call burns its whole budget and then times out.

    Drives `GeminiTeacher`'s injected clock, so the deadline arithmetic is
    exercised at full speed instead of in real seconds. `budgets` is recorded in
    SECONDS — the call carries milliseconds, which is exactly the conversion
    worth pinning here.
    """

    def __init__(self, overshoot: float = 0.0) -> None:
        self.now = 0.0
        self.budgets: list[float] = []
        self.deadlines: list[int] = []
        self.models = self
        self._overshoot = overshoot

    def generate_content(self, **kwargs) -> None:
        http_options = kwargs["config"]["http_options"]
        self.deadlines.append(int(http_options["headers"]["X-Server-Timeout"]))
        budget = http_options["timeout"] / 1000
        self.budgets.append(budget)
        self.now += budget + self._overshoot
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
    assert "недопустимое действие" in gemini.calls[1]["contents"]


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


#: The `retryDelay` Google actually sent on the stand — nearly three times
#: `TOTAL_DEADLINE_S`, which is what makes the retry unwinnable by construction.
QUOTA_RETRY_DELAY_S = 32

#: Verbatim from the stand on 2026-09-24 (JEB-1600, container `7d2e195585cf`),
#: trimmed to the fields the classifier can see.
QUOTA_RESPONSE = {
    "error": {
        "code": 429,
        "message": (
            "You exceeded your current quota, please check your plan and billing details. "
            "* Quota exceeded for metric: "
            "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            "limit: 20, model: gemini-2.5-flash-lite\n"
            f"Please retry in {QUOTA_RETRY_DELAY_S}.769315624s."
        ),
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaValue": "20",
                    }
                ],
            },
            {
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": f"{QUOTA_RETRY_DELAY_S}s",
            },
        ],
    }
}


def quota_error():
    """The real `google.genai` exception, not a stand-in for it.

    The classifier reads `code` and the rendered message, and both are the SDK's
    to shape — a hand-rolled double could keep passing through a rename that
    breaks the live path. The SDK is a hard dependency of the image, so importing
    it here costs a test run nothing.
    """
    from google.genai import errors

    return errors.ClientError(429, QUOTA_RESPONSE)


@pytest.mark.anyio
async def test_a_429_is_not_retried_and_says_so(client, conn, seeded, missing, teacher):
    """Branch 1 of JEB-1600: quota, and the one failure a retry cannot help with.

    The API names a `retryDelay` of 32 s inside the very response that failed,
    against a 12 s `TOTAL_DEADLINE_S` — so the second call could not land even
    with a perfect backoff, and the old `continue` fired it 247 ms later. It
    always failed, and it always cost a second unit of a 20-per-day quota, which
    is what halved the number of commands the stand could serve.
    """
    error = quota_error()
    assert QUOTA_RETRY_DELAY_S > TOTAL_DEADLINE_S, "otherwise waiting it out would be an option"
    gemini = teacher(error)

    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert len(gemini.calls) == 1, "a 429 must never spend a second unit of quota"
    assert body["reply"] == QUOTA_REPLY
    assert [a["action"] for a in body["actions"]] == [s["action"] for s in QUOTA_PLAN]

    state = json.loads(teacher_rows(conn)[0]["state_json"])
    assert state["error"].startswith(QUOTA_ERROR)
    assert "RESOURCE_EXHAUSTED" in state["error"], "the operator gets the cause, not a paraphrase"
    assert state["handled"] is False, "an outage is not something the miner may learn"


@pytest.mark.anyio
async def test_quota_and_not_understood_are_different_answers(
    client, conn, seeded, missing, teacher
):
    """Same fallback for both is how a billing problem read as Pixel being dumb.

    `FALLBACK_REPLY` invites the user to teach the robot another way — advice
    that cannot work while the teacher is unreachable, and that costs another
    unit of the quota to discover.
    """
    teacher(quota_error())
    quota = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert quota["reply"] != FALLBACK_REPLY
    assert quota["actions"] != FALLBACK_PLAN
    assert quota["engine"] == "gemini", "it is still the teacher's turn that failed"


@pytest.mark.anyio
async def test_the_older_rate_limit_shape_is_recognised_too(client, seeded, missing, teacher):
    """`teacher_log` on the stand holds 8 rows from the openai-compatible client.

    That client is gone from the tree, its rows are not: whichever SDK a future
    deployment runs, a 429 is a 429, and the classifier matches on the message
    rather than on an importable exception type.
    """

    class RateLimitError(Exception):
        pass

    gemini = teacher(
        RateLimitError(
            "Error code: 429 - {'error': {'message': 'Rate limit exceeded for model "
            "gemini-2.5-flash-lite (limit: 20 requests per day on Free Tier). "
            "Please retry in 29s'}}"
        )
    )

    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert len(gemini.calls) == 1
    assert body["reply"] == QUOTA_REPLY


@pytest.mark.anyio
async def test_a_non_quota_failure_keeps_its_retry(client, seeded, missing, teacher):
    """The counterweight: only 429 loses the retry.

    A 503 on this model is routine on bursts and a second call genuinely fixes
    it, so classifying too broadly would trade one defect for another.
    """
    gemini = teacher(httpx.ConnectError("connection reset"), GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert len(gemini.calls) == 2
    assert body["reply"] == "Тада!"
    assert body["teacher_unavailable"] is False, "a retried failure is not an outage"


@pytest.mark.anyio
async def test_a_quota_refusal_is_flagged_for_the_client(client, seeded, missing, teacher):
    """The reply alone is not a contract — the UI needs a field it can branch on.

    `engine` cannot carry it: it is the frozen stage-1 shape and a new value
    there breaks every client that reads it. `teacher_log.error` cannot either —
    it never leaves the server. Hence the added flag, default false, which an
    old frontend simply ignores (JEB-1603).
    """
    teacher(quota_error())
    quota = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert quota["teacher_unavailable"] is True
    assert quota["engine"] == "gemini", "the frozen field keeps its frozen meaning"


@pytest.mark.anyio
async def test_an_ordinary_answer_is_not_flagged_unavailable(client, seeded, missing, teacher):
    teacher(GOOD_PLAN)
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()
    assert body["teacher_unavailable"] is False


@pytest.mark.anyio
async def test_a_quota_refusal_keeps_its_label_after_a_reload(client, seeded, missing, teacher):
    """The flag goes through the database, or the reload relabels an outage.

    `GET /api/history` is what redraws the chat after F5, and it reads rows, not
    the `TeacherResult` that wrote them.
    """
    teacher(quota_error())
    await client.post("/api/chat", json={"text": "покажи фокус"})
    teacher(GOOD_PLAN)
    await client.post("/api/chat", json={"text": "покажи другой фокус"})

    history = (await client.get("/api/history")).json()

    assert [item["teacher_unavailable"] for item in history] == [True, False]
    assert history[0]["reply"] == QUOTA_REPLY


@pytest.mark.anyio
async def test_a_quota_refusal_never_reaches_the_mining_pool(
    client, conn, seeded, missing, teacher
):
    """A billing outage is not an example of anything — it must not be learned."""
    teacher(quota_error())
    await client.post("/api/chat", json={"text": "покажи фокус"})

    state = json.loads(teacher_rows(conn)[0]["state_json"])
    assert state["error"], "the error field is what keeps the row out of the pool"
    assert load_pool(conn) == []


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
    assert state["handled"] is True

    assert json.loads(row["actions_json"]) == body["actions"]


@pytest.mark.anyio
async def test_a_refusal_with_nothing_to_do_still_speaks(client, seeded, missing, teacher):
    """The live model declines with `actions: []` — there is nothing in the
    library to do, only something to say. Retrying that cost two round trips and
    replaced a good refusal with FALLBACK_REPLY (JEB-1547)."""
    gemini = teacher(
        json.dumps(
            {"reply": "Я не умею предсказывать погоду", "handled": False, "actions": []},
            ensure_ascii=False,
        )
    )
    body = (await client.post("/api/chat", json={"text": "какая погода"})).json()

    assert len(gemini.calls) == 1, "a refusal is an answer, not something to retry"
    assert body["reply"] == "Я не умею предсказывать погоду"
    assert body["actions"] == [
        {"action": "say", "args": {"text": "Я не умею предсказывать погоду"}}
    ]


@pytest.mark.anyio
async def test_an_empty_plan_that_claims_to_have_acted_is_retried(
    client, seeded, missing, teacher
):
    """`handled` true and nothing done is a contradiction, not a refusal."""
    gemini = teacher(
        json.dumps({"reply": "Смотри!", "handled": True, "actions": []}), GOOD_PLAN
    )
    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert len(gemini.calls) == 2
    assert body["reply"] == "Тада!"


@pytest.mark.anyio
async def test_a_declined_answer_reaches_the_user_and_is_marked_for_the_miner(
    client, conn, seeded, missing, teacher
):
    """The teacher answered, and said it cannot do this.

    Nothing changes for the user — a plan, a reply, the `gemini` badge. What
    changes is the row: `handled` false is what keeps the miner from turning a
    refusal into a permanent Laya skill (JEB-1547, `backend/miner/case._parse`).
    """
    teacher(DECLINED_PLAN)
    body = (await client.post("/api/chat", json={"text": "закажи пиццу"})).json()

    assert body["engine"] == "gemini"
    assert "не умею заказывать" in body["reply"]

    row = teacher_rows(conn)[0]
    state = json.loads(row["state_json"])
    assert state["error"] is None, "the call succeeded — this is not a failure"
    assert state["handled"] is False
    # Still written, still kept: "the user keeps asking for food delivery" is
    # worth reading even though no skill can come of it.
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
    config = call["config"]
    assert call["model"] == "fake-model"
    # Milliseconds on this path, so the first call asks for TIMEOUT_S x 1000.
    assert config["http_options"]["timeout"] == int(TIMEOUT_S * 1000)
    # ...and the server deadline is sent separately, because the API floors it
    # at 10 s and would 400 our 8 s outright.
    assert config["http_options"]["headers"]["X-Server-Timeout"] == "10"
    assert config["response_mime_type"] == "application/json"
    assert set(config["response_schema"]["properties"]) == {"reply", "handled", "actions"}
    # The prompt is generated from the library, so it cannot drift from it.
    assert all(name in config["system_instruction"] for name in ACTIONS)


def test_the_prompt_shows_the_skill_boundary(seeded):
    skills = load_skills(seeded)
    prompt = build_input("покажи фокус", read_state(seeded), skills)

    # Exactly one line per skill, `id: description` and nothing else. `examples`
    # belong to the router; repeating them here only dilutes the boundary the
    # miner is meant to read off this list.
    listed = [line for line in prompt.splitlines() if line.startswith("- ")]
    assert listed == [f"- {skill.id}: {skill.description}" for skill in skills]


def test_an_over_long_reply_is_trimmed_not_rejected():
    plan = TeacherPlan.model_validate({"reply": "я" * 400, "handled": True, "actions": []})
    assert plan.reply == "я" * MAX_SAY_LEN


@pytest.mark.anyio
@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
async def test_a_blank_reply_never_reaches_the_user(client, seeded, missing, teacher, blank):
    """A blank `reply` with no `say` step would render as an empty bubble.

    `""` is stopped by `min_length` in the schema and `"   "` only by the strip
    check in `_parse` — both must end up retried, then on the fallback.
    """
    gemini = teacher(
        json.dumps({"reply": blank, "handled": True, "actions": [{"action": "jump"}]})
    )

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


def test_the_shortened_retry_still_sends_a_deadline_the_api_accepts(seeded):
    """The retry's budget is 4 s, and the API rejects any deadline under 10 s.

    Tying the two together would turn every retry into a `400 INVALID_ARGUMENT`
    — a fallback for a reason that has nothing to do with the user's command.
    The client-side budget shrinks; the deadline we announce does not.
    """
    gemini = TimingOutClient()
    teacher = GeminiTeacher(client=gemini, model="fake-model", clock=lambda: gemini.now)

    teacher.explain("покажи фокус", read_state(seeded), [])

    assert gemini.budgets[1] < MIN_SERVER_DEADLINE_S, "the retry is the case that matters here"
    assert gemini.deadlines == [MIN_SERVER_DEADLINE_S, MIN_SERVER_DEADLINE_S]


def test_a_budget_above_the_floor_is_announced_to_the_server():
    """`MIN_SERVER_DEADLINE_S` is a floor, not the deadline we always announce.

    Today's `TIMEOUT_S` keeps every budget under the floor, so this case is
    unreachable through `explain` — until someone raises the ceiling. Announcing
    a flat 10 s then has the server cut the call at 10 s while httpx waits out
    the full budget: the extra seconds buy nothing and the user sees a bare
    timeout. `_call` takes the budget as a parameter, so it is asked directly.
    """
    gemini = TimingOutClient()
    teacher = GeminiTeacher(client=gemini, model="fake-model", clock=lambda: gemini.now)

    budget = MIN_SERVER_DEADLINE_S + 5.5
    with pytest.raises(httpx.TimeoutException):
        teacher._call("покажи фокус", budget)

    # Rounded UP, the way the SDK's own `populate_server_timeout_header` does it:
    # a deadline announced shorter than the budget is the whole defect.
    assert gemini.deadlines == [16]
    assert gemini.deadlines[0] >= budget


def test_no_retry_is_dialled_with_nothing_left_to_spend(seeded):
    gemini = TimingOutClient(overshoot=TOTAL_DEADLINE_S)
    teacher = GeminiTeacher(client=gemini, model="fake-model", clock=lambda: gemini.now)

    result = teacher.explain("покажи фокус", read_state(seeded), [])

    assert len(gemini.budgets) == 1, "a second call with < MIN_CALL_BUDGET_S left is not worth it"
    assert "out of time" in result.error
    assert result.raw_plan == FALLBACK_PLAN


def test_the_default_model_is_the_one_the_owner_picked(monkeypatch):
    # The hot path runs on the cheap 2.5 model split the bill uses, written with
    # the `models/` prefix like the miner. A silent drift here is billed per miss.
    monkeypatch.delenv("GEMINI_TEACHER_MODEL", raising=False)
    assert DEFAULT_MODEL == "models/gemini-2.5-flash-lite"
    assert GeminiTeacher()._model == DEFAULT_MODEL


def test_the_env_still_overrides_the_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_TEACHER_MODEL", "models/some-other-model")
    assert GeminiTeacher()._model == "models/some-other-model"
    # An explicit argument wins over both — that is what every test above relies on.
    assert GeminiTeacher(model="fake-model")._model == "fake-model"


def test_the_sdk_still_has_the_surface_we_call():
    """The one check CI could not make before: that the *real* SDK takes what
    `GeminiTeacher._call` passes it.

    `FakeGeminiClient` accepts any keyword, so every other test here proves only
    that the code *sends* the argument. `config` is a plain dict, so a renamed
    field would not even raise — it would ride along to the API, the answer
    would come back unschema'd, and every router miss would quietly degrade to
    `FALLBACK_PLAN` with CI still green. That is precisely the defect JEB-1513
    fixed. Constructing the client needs no network and no real key.
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
    # The timeout lives here on this path, in milliseconds, and `headers` is how
    # the server deadline is kept off it (`MIN_SERVER_DEADLINE_S`).
    assert {"timeout", "headers"} <= set(types.HttpOptions.model_fields)
    assert hasattr(types.GenerateContentResponse, "text")

    # The whole `MIN_SERVER_DEADLINE_S` trick rests on the SDK filling
    # `X-Server-Timeout` from the timeout only when we have not set it ourselves.
    # If that ever flips, our 8 s budget goes back out as the server deadline and
    # every teacher call 400s.
    from google.genai._api_client import populate_server_timeout_header

    headers = {"X-Server-Timeout": "10"}
    populate_server_timeout_header(headers, 8.0)
    assert headers["X-Server-Timeout"] == "10"


@pytest.mark.anyio
async def test_a_fenced_response_is_not_parsed(client, conn, seeded, missing, teacher):
    """The fences stay a failure, deliberately.

    ` ```json ` around an otherwise valid plan is what `interactions.create`
    returned on `models/gemini-2.5-flash-lite`, and stripping it in `_parse`
    would have hidden the wrong call shape rather than fixed it. So a fenced
    answer must still be retried and still end on the fallback — that is the
    signal the call path is wrong.
    """
    gemini = teacher(f"```json\n{GOOD_PLAN}\n```")

    body = (await client.post("/api/chat", json={"text": "покажи фокус"})).json()

    assert len(gemini.calls) == 2, "a fenced answer is retried like any other bad response"
    assert body["reply"] == FALLBACK_REPLY
    assert [a["action"] for a in body["actions"]] == [s["action"] for s in FALLBACK_PLAN]
    assert "did not match the schema" in json.loads(teacher_rows(conn)[0]["state_json"])["error"]
