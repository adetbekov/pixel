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
from dataclasses import dataclass
from typing import Any, Protocol

from ..actions import MAX_PLAN_LEN, InvalidPlan, validate_plan
from ..brain.skill import Skill
from ..state import RobotState
from .prompt import SYSTEM_PROMPT, build_input, retry_hint
from .schema import TeacherPlan

log = logging.getLogger(__name__)

#: `gemini-3.5-flash-lite` (named in JEB-1500) is not a model id google-genai
#: 2.25.0 knows; `gemini-3.1-flash-lite` is the current flash-lite tier. The
#: online path wants the cheapest, fastest tier — the teacher writes an
#: 8-primitive plan, not a chain of reasoning. Stage 4 mines offline and can
#: afford a bigger model.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

#: The user is already waiting out a 217-240 ms router miss before this starts.
TIMEOUT_S = 8.0

#: One first try plus one retry — of either kind (network or bad plan).
MAX_ATTEMPTS = 2

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
    derives the visible text from the plan.
    """
    if not reply.strip() or any(step["action"] == "say" for step in raw_plan):
        return raw_plan
    return [*raw_plan[: MAX_PLAN_LEN - 1], {"action": "say", "args": {"text": reply}}]


class GeminiTeacher:
    """The real teacher. One client, reused; no state between calls.

    ``client`` is injectable so the tests drive the retry and fallback paths
    without a network — and so ``google.genai`` stays off the import path of a
    test run, exactly like ``laya`` in :mod:`backend.brain.engine`.
    """

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self._client = client
        self._model = model or os.environ.get("GEMINI_TEACHER_MODEL", DEFAULT_MODEL)

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Imported lazily: an app started without GEMINI_API_KEY must not
            # need the SDK on the import path at all.
            from google import genai

            self._client = genai.Client()
        return self._client

    def _call(self, prompt: str) -> str:
        """One round trip. Verified against google-genai 2.25.0, where
        ``client.interactions.create`` / ``interaction.output_text`` is the
        canonical path (``client.models.generate_content`` also still exists)."""
        interaction = self._ensure_client().interactions.create(
            model=self._model,
            input=prompt,
            system_instruction=SYSTEM_PROMPT,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": TeacherPlan.model_json_schema(),
            },
            timeout=TIMEOUT_S,
        )
        return interaction.output_text or ""

    def explain(self, text: str, state: RobotState, skills: list[Skill]) -> TeacherResult:
        prompt = build_input(text, state, skills)
        last_raw = ""
        last_reason = "teacher did not produce a plan"

        for attempt in range(MAX_ATTEMPTS):
            hint = "" if attempt == 0 else retry_hint(last_reason)
            try:
                last_raw = self._call(prompt + hint)
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
