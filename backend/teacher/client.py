"""Calling Gemini, and surviving every way that can go wrong.

The contract this module keeps with the rest of the app: :meth:`Teacher.explain`
*always* returns a usable plan built from the action library. A timeout, a 500,
a hallucinated primitive, unparsable JSON — none of them reach the caller as an
exception, and none of them show the user a traceback. The worst case is
:data:`FALLBACK_PLAN` plus a reason recorded in ``teacher_log``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ..actions import MAX_PLAN_LEN, InvalidPlan, validate_plan
from ..brain.skill import Skill
from ..state import RobotState
from .prompt import SYSTEM_PROMPT, build_input, retry_hint
from .schema import TeacherPlan

log = logging.getLogger(__name__)

#: The online path wants the cheapest, fastest tier — the teacher writes an
#: 8-primitive plan, not a chain of reasoning; stage 4 mines offline and can
#: afford a bigger model.
#:
#: The owner settled this on 2026-09-23: the same cheap 2.5 model split the bill
#: already runs (`src/services/image_parser.py`), $0.30 / 1M in, $2.50 / 1M out.
#: The miner uses it too (`backend/miner/generate.py`), so both halves of the
#: project name the model the same way, `models/` prefix included.
#:
#: Careful if a thinking budget is ever added here: `models/gemini-2.5-flash-lite`
#: rejects `thinking_budget=1` with `400 INVALID_ARGUMENT` and wants >= 512
#: (measured in split the bill, `src/services/gemini_thinking_budget.py`). The
#: teacher passes no budget today — don't add one without that floor.
#:
#: TODO: once `GEMINI_API_KEY` lands (JEB-1508), confirm against
#: `client.models.list()` that the model is visible and that this is the id form
#: the live call accepts. Overridable via `GEMINI_TEACHER_MODEL` either way.
DEFAULT_MODEL = "models/gemini-2.5-flash-lite"

#: Per-call ceiling, as JEB-1500 specifies.
TIMEOUT_S = 8.0

#: Ceiling across *all* attempts. Without it a retried timeout costs the user
#: 2 x TIMEOUT_S on top of the router miss they already waited out; the retry is
#: worth having, 16 s of dead air is not. Each call gets whatever is left.
TOTAL_DEADLINE_S = 12.0

#: One first try plus one retry — of either kind (network or bad plan).
MAX_ATTEMPTS = 2

#: Below this there is no point dialling the API at all.
MIN_CALL_BUDGET_S = 1.0

#: The shortest deadline the API will *accept*, and it is longer than ours.
#:
#: ``http_options.timeout`` does two jobs in google-genai 2.25.0: it is the httpx
#: client-side timeout, and ``ceil()``-ed to seconds it is also sent as the
#: ``X-Server-Timeout`` header. Measured live on 2026-09-23 against
#: ``models/gemini-2.5-flash-lite``: anything that rounds below 10 s is rejected
#: outright — ``400 INVALID_ARGUMENT: Manually set deadline 8s is too short.
#: Minimum allowed deadline is 10s`` — so our 8 s budget cannot be the header.
#: 9400 ms passes, 9000 ms does not.
#:
#: The SDK only fills the header in when it is absent, so :meth:`_call` sets it
#: explicitly: the server gets its legal minimum, httpx still aborts at *our*
#: budget, and JEB-1500's per-call ceiling survives the move off
#: ``interactions.create`` (which took plain seconds and never saw this floor).
MIN_SERVER_DEADLINE_S = 10

FALLBACK_REPLY = "Я не понял, научи меня по-другому"

FALLBACK_PLAN: list[dict[str, Any]] = [
    {"action": "set_face", "args": {"face": "curious"}},
    {"action": "say", "args": {"text": FALLBACK_REPLY}},
]


@dataclass(frozen=True)
class TeacherResult:
    """What the teacher hands back — never an exception.

    ``raw_plan`` has already been through :func:`validate_plan`, so it only ever
    names library actions. ``raw_response`` is the model's untouched text and
    goes straight into ``teacher_log`` for the miner; it is never returned over
    the API. ``error`` is set exactly when the plan is the fallback.
    """

    raw_plan: list[dict[str, Any]]
    reply: str
    raw_response: str
    error: str | None = None


def _fallback(reason: str, raw_response: str = "") -> TeacherResult:
    return TeacherResult(
        raw_plan=list(FALLBACK_PLAN),
        reply=FALLBACK_REPLY,
        raw_response=raw_response,
        error=reason,
    )


class Teacher(Protocol):
    def explain(self, text: str, state: RobotState, skills: list[Skill]) -> TeacherResult: ...


def _with_reply(raw_plan: list[dict[str, Any]], reply: str) -> list[dict[str, Any]]:
    """Make sure the robot actually says its reply.

    The model usually puts the line in a ``say`` step itself. When it does not,
    the reply would otherwise never reach the user, since ``execute_plan``
    derives the visible text from the plan. ``reply`` is non-blank by the time
    this runs — :func:`_parse` rejects a blank one rather than passing it on.
    """
    if any(step["action"] == "say" for step in raw_plan):
        return raw_plan
    return [*raw_plan[: MAX_PLAN_LEN - 1], {"action": "say", "args": {"text": reply}}]


class GeminiTeacher:
    """The real teacher. One client, reused; no state between calls.

    ``client`` is injectable so the tests drive the retry and fallback paths
    without a network — and so ``google.genai`` stays off the import path of a
    test run, exactly like ``laya`` in :mod:`backend.brain.engine`.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client
        self._model = model or os.environ.get("GEMINI_TEACHER_MODEL", DEFAULT_MODEL)
        # Injectable so the deadline arithmetic is testable without sleeping.
        self._clock = clock

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Imported lazily: an app started without GEMINI_API_KEY must not
            # need the SDK on the import path at all.
            from google import genai

            self._client = genai.Client()
        return self._client

    def _call(self, prompt: str, timeout: float) -> str:
        """One round trip, over ``models.generate_content``.

        Not ``interactions.create``, which is what this used to call. Measured
        against the live API on 2026-09-23 with google-genai 2.25.0:
        ``interactions.create`` honours ``response_format`` on
        ``gemini-3.8-flash`` but *not* on ``models/gemini-2.5-flash-lite`` —
        that model answers inside a ```` ```json ```` fence, which :func:`_parse`
        rejects on both attempts, so every router miss degraded to
        :data:`FALLBACK_PLAN` behind a single ``log.warning`` while
        ``teacher_log`` filled with fallbacks and starved the miner. The same
        model on ``models.generate_content`` with ``response_schema`` returns
        bare JSON. The miner made the same move for the same reason
        (``backend/miner/generate.py``).

        The fences are a symptom of the wrong call shape, not a response format
        to support: ``_parse`` stays strict — it is what made this visible.

        ``config`` is a plain dict rather than ``types.GenerateContentConfig``
        so that ``google.genai`` stays off the import path of a key-less app,
        exactly like the lazy client construction above.
        ``tests/test_teacher.py::test_the_sdk_still_has_the_surface_we_call``
        asserts the field names against the installed package instead — without
        it a rename would ride along silently and 400 at the API.
        """
        response = self._ensure_client().models.generate_content(
            model=self._model,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": TeacherPlan.model_json_schema(),
                "http_options": {
                    # On this path the timeout lives in `http_options`, and there
                    # it is in MILLISECONDS — `TIMEOUT_S` seconds x 1000.
                    "timeout": int(timeout * 1000),
                    # ...but the same value also becomes the server deadline,
                    # which has a 10 s floor. See `MIN_SERVER_DEADLINE_S`: this
                    # header keeps the request legal without lengthening the
                    # client-side wait above.
                    "headers": {"X-Server-Timeout": str(MIN_SERVER_DEADLINE_S)},
                },
            },
        )
        return response.text or ""

    def explain(self, text: str, state: RobotState, skills: list[Skill]) -> TeacherResult:
        prompt = build_input(text, state, skills)
        deadline = self._clock() + TOTAL_DEADLINE_S
        last_raw = ""
        last_reason = "teacher did not produce a plan"

        for attempt in range(MAX_ATTEMPTS):
            budget = min(TIMEOUT_S, deadline - self._clock())
            if budget < MIN_CALL_BUDGET_S:
                last_reason = f"out of time after {attempt} attempt(s): {last_reason}"
                break

            hint = "" if attempt == 0 else retry_hint(last_reason)
            try:
                last_raw = self._call(prompt + hint, budget)
            except Exception as exc:  # noqa: BLE001 — a 500 must not reach the user
                last_reason = f"{type(exc).__name__}: {exc}"
                log.warning("teacher call failed (attempt %d): %s", attempt + 1, last_reason)
                continue

            result, last_reason = _parse(last_raw)
            if result is not None:
                return result
            log.warning("teacher plan rejected (attempt %d): %s", attempt + 1, last_reason)

        return _fallback(last_reason, last_raw)


def _parse(raw: str) -> tuple[TeacherResult | None, str]:
    """Both validation layers. A ``None`` result carries the reason to retry on."""
    try:
        plan = TeacherPlan.model_validate_json(raw)
    except ValueError as exc:
        return None, f"response did not match the schema: {exc}"

    # `max_length` bounds the reply but nothing stops the model returning "" or
    # whitespace, and an empty `reply` with no `say` step reaches the user as an
    # empty chat bubble. Retry it; a second blank one gets FALLBACK_REPLY.
    if not plan.reply.strip():
        return None, "plan came back with an empty reply"

    proposed = plan.to_raw_plan()
    try:
        actions = validate_plan(proposed)
    except InvalidPlan as exc:
        return None, str(exc)

    # `validate_plan` drops what it does not recognise rather than raising, so a
    # shorter result is how an invented primitive announces itself.
    if len(actions) != len(proposed):
        named = [step["action"] for step in proposed]
        return None, f"plan named an action outside the library: {named}"
    if not actions:
        return None, "plan was empty"

    raw_plan = _with_reply([a.to_dict() for a in actions], plan.reply)
    return TeacherResult(raw_plan=raw_plan, reply=plan.reply, raw_response=raw), ""


def build_teacher() -> Teacher | None:
    """The teacher, or ``None`` when there is no API key.

    A missing key is a supported configuration, not a startup failure: Pixel
    runs on Laya alone and ``/api/metrics`` simply shows a Gemini share of zero.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    return GeminiTeacher()


_teacher: Teacher | None = None


def set_teacher(teacher: Teacher | None) -> None:
    global _teacher
    _teacher = teacher


def get_teacher() -> Teacher | None:
    """``None`` means the teacher is off — the caller answers with the stub."""
    return _teacher
