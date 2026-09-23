"""Does the live model still answer in the shape our parsers demand?

Two call shapes reach Gemini from this repo — `GeminiTeacher._call` (online,
one reply) and `GeminiSkillGenerator._call` (offline, one skill draft) — and
both hand the raw text straight to a pydantic model with no cleaning in
between. That "no cleaning" is the contract: `TeacherPlan.model_validate_json`
and `SkillDraft.model_validate_json` reject a ```` ```json ```` fence, and a
model that starts fencing takes the whole feature down while every test stays
green.

Which is exactly what happened twice (JEB-1513, JEB-1546):
`interactions.create` + `response_format` does not hold structured output on
`models/gemini-2.5-flash-lite` — the answer comes back fenced — while
`models.generate_content` + `response_schema` returns bare JSON. Both times CI
saw nothing, because `tests/fakes.py::FakeGeminiClient` returns bare JSON
whatever it is asked and `test_the_sdk_still_has_the_surface_we_call` checks
that the *arguments* exist, not that the answer obeys them.

So this script asserts the one thing only a live call can: the bytes coming
back parse. It calls each `_call` for real, once, with the shortest useful
prompt, and parses the result with the same schema production uses, unmodified.
It never touches the app's logic — a failure here means the model's behaviour
moved, not that a plan was bad.

Run by `.github/workflows/live-gemini-contract.yml` on a schedule, never on a
PR: it needs `GEMINI_API_KEY`, which a fork PR cannot have.

Exit codes, which the workflow's alert step branches on:

  0  both shapes answered in a parsable form
  1  a real finding — at least one shape's answer did not parse
  2  the probe could not be run at all (no key, SDK missing, import error);
     this is *not* evidence that the contract broke
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from backend.miner.case import Case
from backend.miner.generate import SYSTEM_PROMPT as MINER_SYSTEM_PROMPT
from backend.miner.generate import GeminiSkillGenerator
from backend.miner.generate import build_input as build_miner_input
from backend.miner.schema import SkillDraft
from backend.state import RobotState
from backend.teacher.client import GeminiTeacher
from backend.teacher.prompt import build_input as build_teacher_input
from backend.teacher.schema import TeacherPlan

#: One first try plus one retry, mirroring `MAX_ATTEMPTS` on both real paths: a
#: single malformed answer is a retry in production too, so alerting on one
#: would report a state the app itself survives. Two in a row is the contract.
ATTEMPTS = 2

#: How much of the answer goes into the failure message. Enough to see a fence,
#: a prose preamble or an error envelope; short enough to read in a log line.
PREVIEW_CHARS = 160

#: The SDK entry points this repo has ever called, so the failure message can
#: name the shape that produced the answer instead of just the method it lives
#: in. The pairing with the config key is the part that matters — it is the
#: `interactions.create` + `response_format` combination that fences.
SDK_ENTRY_POINTS = (
    ("interactions.create", "response_format"),
    ("models.generate_content", "response_schema"),
)


def _code_of(func: Callable) -> str:
    """`func`'s body with its docstring and comments removed.

    Both `_call` docstrings *narrate* the shape they moved away from — naming
    `interactions.create` in a paragraph about why it is no longer called. Read
    naively, the teacher therefore reports both shapes. `ast.unparse` drops
    comments outright and the docstring is dropped explicitly, so what is left
    is only what actually executes.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    definition = tree.body[0]
    body = getattr(definition, "body", [])
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        definition.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def call_shape(func: Callable) -> str:
    """Which SDK call `func` makes, read off its own source.

    The failure message has to say *which shape* was probed, and hard-coding
    that would go stale in the same commit that changes the call — the very
    commit this script exists to judge.
    """
    try:
        source = _code_of(func)
    except (OSError, TypeError, SyntaxError):  # pragma: no cover - source is always readable here
        return "an SDK entry point this script could not read"
    named = [
        f"{method} + {config_key}"
        for method, config_key in SDK_ENTRY_POINTS
        if method in source and config_key in source
    ]
    return " and ".join(named) if named else "an SDK entry point this script does not recognise"


def teacher_probe() -> str:
    """One real teacher round trip. Shortest prompt that is still a real one."""
    state = RobotState(mood=60.0, energy=60.0, fullness=60.0, face="curious")
    prompt = build_teacher_input("покрутись", state, [])
    teacher = GeminiTeacher()
    return teacher._call(prompt, 8.0)


def miner_probe() -> str:
    """One real generator round trip, on a two-command cluster."""
    state = RobotState(mood=60.0, energy=60.0, fullness=60.0, face="curious")
    actions = [
        {"action": "spin", "args": {}},
        {"action": "say", "args": {"text": "Тада!"}},
    ]
    cases = [
        Case(id=1, user_text="покажи фокус", state=state, actions=actions),
        Case(id=2, user_text="сделай фокус", state=state, actions=actions),
    ]
    prompt = build_miner_input(cases, [])
    generator = GeminiSkillGenerator()
    return generator._call(MINER_SYSTEM_PROMPT, prompt, SkillDraft.model_json_schema())


@dataclass(frozen=True)
class Probe:
    """One call shape, its live invocation and the schema its answer must fit."""

    name: str
    call: Callable[[], str]
    #: The bound `_call` whose source names the SDK shape, for the message.
    shape_of: Callable
    parse: Callable[[str], object]


PROBES = (
    Probe(
        name="GeminiTeacher._call",
        call=teacher_probe,
        shape_of=GeminiTeacher._call,
        parse=TeacherPlan.model_validate_json,
    ),
    Probe(
        name="GeminiSkillGenerator._call",
        call=miner_probe,
        shape_of=GeminiSkillGenerator._call,
        parse=SkillDraft.model_validate_json,
    ),
)


def preview(raw: str) -> str:
    """The head of the answer, quoted — escapes included, so a fence is visible."""
    head = raw[:PREVIEW_CHARS]
    suffix = "..." if len(raw) > PREVIEW_CHARS else ""
    return f"{head!r}{suffix}"


def diagnose(raw: str) -> str:
    """The extra sentence that turns "did not parse" into "do this"."""
    stripped = raw.lstrip()
    if stripped.startswith("```"):
        return (
            "The answer came back inside a Markdown fence. This is the JEB-1546 / "
            "JEB-1513 failure: the call shape is not holding structured output on "
            "this model. Move the call to `models.generate_content` + "
            "`response_schema` — do NOT add fence-stripping to the parser, which is "
            "what made this visible."
        )
    if not stripped:
        return "The answer was empty — no text came back at all."
    if not stripped.startswith(("{", "[")):
        return (
            "The answer does not even start as JSON. The model is prefacing it with "
            "prose, so structured output is not in force on this call."
        )
    return (
        "The answer is JSON but does not fit the schema this repo parses it with. "
        "Compare the schema the call sends against the model's reply in the log above."
    )


def run(probe: Probe) -> str | None:
    """`None` when the shape is healthy, otherwise the finding to print."""
    failures = []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            raw = probe.call()
        except Exception as exc:  # noqa: BLE001 — every failure is reportable here
            failures.append(f"attempt {attempt}: the call itself raised {type(exc).__name__}: {exc}")
            continue

        try:
            probe.parse(raw)
        except ValueError as exc:
            failures.append(
                f"attempt {attempt}: answer did not parse — {exc}\n"
                f"    answer starts: {preview(raw)}\n"
                f"    {diagnose(raw)}"
            )
            continue

        print(f"ok: {probe.name} ({call_shape(probe.shape_of)}) answered in a parsable form")
        return None

    joined = "\n  ".join(failures)
    return (
        f"{probe.name} — called as {call_shape(probe.shape_of)} — did not return an "
        f"answer this repo can parse, in {ATTEMPTS} attempt(s):\n  {joined}"
    )


def main() -> int:
    if not os.environ.get("GEMINI_API_KEY"):
        print(
            "GEMINI_API_KEY is not set, so the live contract was not checked. "
            "This is not a pass and not a finding.",
            file=sys.stderr,
        )
        return 2

    try:
        import google.genai  # noqa: F401
    except ImportError:
        print(
            "google-genai is not installed, so the live contract was not checked. "
            "This is not a pass and not a finding.",
            file=sys.stderr,
        )
        return 2

    findings = []
    for probe in PROBES:
        # Both shapes are always probed: a teacher finding says nothing about the
        # miner, and half an answer reads like a whole one in the alert issue.
        try:
            finding = run(probe)
        except Exception:  # noqa: BLE001 — a crash in one probe must not hide the other
            traceback.print_exc()
            findings.append(f"{probe.name}: the probe itself crashed, see the traceback above")
            continue
        if finding:
            findings.append(finding)

    if not findings:
        print("Both call shapes hold their contract with the live model.")
        return 0

    print("\nFINDINGS:", file=sys.stderr)
    for finding in findings:
        print(f"- {finding}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
