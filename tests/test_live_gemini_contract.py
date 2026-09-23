"""The live-contract probe's own judgement, checked without a network.

`.github/scripts/check_live_gemini_contract.py` only ever runs on a schedule with
a real key, so the thing that can rot unnoticed is the verdict itself — a gate
that cannot fail looks exactly like a gate that passes. These tests feed it the
two answers that matter, the fenced one that took the feature down twice
(JEB-1513, JEB-1546) and the bare-JSON one, and assert it separates them.

The live calls are never made here: each probe is replaced with a function that
returns a canned string, so no key and no SDK round trip is involved.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "check_live_gemini_contract.py"

spec = importlib.util.spec_from_file_location("check_live_gemini_contract", SCRIPT)
contract = importlib.util.module_from_spec(spec)
# Registered before it executes: `@dataclass` resolves annotations through
# `sys.modules[cls.__module__]`, and a module loaded by path is not there yet.
sys.modules[spec.name] = contract
spec.loader.exec_module(contract)

GOOD_PLAN = json.dumps(
    {"reply": "Кручусь!", "actions": [{"action": "spin"}, {"action": "say", "text": "Кручусь!"}]},
    ensure_ascii=False,
)

FENCED_PLAN = f"```json\n{GOOD_PLAN}\n```"


def probe(answer: str) -> contract.Probe:
    """A teacher probe whose round trip is a constant."""
    return contract.Probe(
        name="GeminiTeacher._call",
        call=lambda: answer,
        shape_of=contract.GeminiTeacher._call,
        parse=contract.TeacherPlan.model_validate_json,
    )


def test_a_bare_json_answer_passes():
    assert contract.run(probe(GOOD_PLAN)) is None


def test_a_fenced_answer_is_a_finding():
    """The exact break this gate exists for. The parser stays strict, so the
    fence has to fail here — if it ever passes, someone taught the parser to
    clean the response and the gate is blind again."""
    finding = contract.run(probe(FENCED_PLAN))

    assert finding is not None
    # The message has to be actionable from the log alone: which shape, what came
    # back, and what the fence means.
    assert "GeminiTeacher._call" in finding
    assert "```json" in finding
    assert "models.generate_content" in finding


def test_prose_and_emptiness_are_distinguished():
    assert "does not even start as JSON" in contract.diagnose("Конечно! Вот план: {}")
    assert "was empty" in contract.diagnose("   ")
    assert "does not fit the schema" in contract.diagnose('{"reply": 1}')


def test_a_retry_covers_one_bad_answer():
    """Both real paths retry once, so one malformed answer is a state the app
    survives — alerting on it would report a break that is not one."""
    answers = iter([FENCED_PLAN, GOOD_PLAN])
    flaky = contract.Probe(
        name="GeminiTeacher._call",
        call=lambda: next(answers),
        shape_of=contract.GeminiTeacher._call,
        parse=contract.TeacherPlan.model_validate_json,
    )

    assert contract.run(flaky) is None


def test_a_raising_call_is_a_finding_naming_the_exception():
    def boom():
        raise TimeoutError("deadline exceeded")

    finding = contract.run(
        contract.Probe(
            name="GeminiSkillGenerator._call",
            call=boom,
            shape_of=contract.GeminiSkillGenerator._call,
            parse=contract.SkillDraft.model_validate_json,
        )
    )

    assert finding is not None
    assert "TimeoutError: deadline exceeded" in finding


def quota_error():
    """The exact refusal the free tier answers with once the day is spent."""
    errors = pytest.importorskip("google.genai.errors")
    return errors.ClientError(
        429,
        {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "message": (
                    "Quota exceeded for metric: generativelanguage.googleapis.com/"
                    "generate_content_free_tier_requests, limit: 20"
                ),
            }
        },
    )


def raising_probe(exc: BaseException, calls: list[int] | None = None) -> contract.Probe:
    def boom():
        if calls is not None:
            calls.append(1)
        raise exc

    return contract.Probe(
        name="GeminiTeacher._call",
        call=boom,
        shape_of=contract.GeminiTeacher._call,
        parse=contract.TeacherPlan.model_validate_json,
    )


def test_an_exhausted_quota_is_not_a_contract_finding():
    """The free-tier day is 20 calls and the live stand spends from the same
    twenty (JEB-1553), so a refused night is routine. Reported as a finding it
    opens an issue saying the answer no longer parses — when no answer came."""
    with pytest.raises(contract.ProbeUnavailable) as raised:
        contract.run(raising_probe(quota_error()))

    assert "429" in str(raised.value)
    assert "RESOURCE_EXHAUSTED" in str(raised.value)
    assert "GeminiTeacher._call" in str(raised.value)


def test_a_quota_refusal_is_not_retried():
    """The window is a day, not a minute — measured. A second call buys nothing
    and the first one already proved the day is spent."""
    calls: list[int] = []
    with pytest.raises(contract.ProbeUnavailable):
        contract.run(raising_probe(quota_error(), calls))

    assert len(calls) == 1


def test_an_api_error_that_is_not_quota_stays_a_finding():
    """Only the quota refusal is exempt: a 500 or a 400 on the shape we call is
    still the call failing, and the gate must keep saying so."""
    errors = pytest.importorskip("google.genai.errors")
    server_error = errors.ServerError(500, {"error": {"code": 500, "status": "INTERNAL"}})

    finding = contract.run(raising_probe(server_error))

    assert finding is not None
    assert "ServerError" in finding


def test_an_unreachable_model_exits_two_and_a_finding_still_wins(monkeypatch, capsys):
    """Two shapes, two verdicts. Nothing checked is exit 2; one shape proving the
    contract broke is worth alerting on even when the other never ran."""
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-and-never-sent")
    pytest.importorskip("google.genai")

    monkeypatch.setattr(contract, "PROBES", (raising_probe(quota_error()),))
    assert contract.main() == 2
    err = capsys.readouterr().err
    assert "not a pass and not a finding" in err
    assert "RESOURCE_EXHAUSTED" in err

    monkeypatch.setattr(contract, "PROBES", (raising_probe(quota_error()), probe(FENCED_PLAN)))
    assert contract.main() == 1


def test_the_exit_two_alert_names_the_quota_as_a_cause():
    """The probe now exits 2 on a refused day, so the issue that exit opens has to
    say so. Without it the reader is sent to debug a missing key or a broken
    install for the one cause that is neither — and is the expected one."""
    workflow = (ROOT / ".github" / "workflows" / "live-gemini-contract.yml").read_text()
    # The `2)` arm of the `case` that builds the alert, up to the next arm.
    branch = workflow.split("\n            2)\n", 1)[1].split("\n            *)\n", 1)[0]

    assert "429" in branch
    assert "RESOURCE_EXHAUSTED" in branch
    assert "free tier" in branch


def test_the_reported_shape_is_read_from_the_source_not_hardcoded():
    """The failure message names the call shape so a reader knows what was
    probed. Hard-coding it would go stale in the very commit that changes the
    call — the commit this gate judges."""

    def fenced_shape(self):
        self.interactions.create(response_format={})

    def schema_shape(self):
        self.models.generate_content(config={"response_schema": {}})

    assert contract.call_shape(fenced_shape) == "interactions.create + response_format"
    assert contract.call_shape(schema_shape) == "models.generate_content + response_schema"


def test_both_production_call_shapes_are_recognised():
    """Neither `_call` may drift onto an SDK entry point this script cannot name:
    an unrecognised shape still fails on a bad answer, but the alert issue would
    no longer say what was called."""
    for func in (contract.GeminiTeacher._call, contract.GeminiSkillGenerator._call):
        assert "does not recognise" not in contract.call_shape(func)


def test_a_docstring_naming_the_old_shape_is_not_mistaken_for_the_call():
    """`GeminiTeacher._call`'s docstring explains at length why it no longer
    calls `interactions.create`. Read naively that makes the teacher report both
    shapes at once, which is worse than useless in an alert issue."""
    assert contract.call_shape(contract.GeminiTeacher._call) == (
        "models.generate_content + response_schema"
    )


def test_no_key_is_not_reported_as_a_pass_or_a_finding(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert contract.main() == 2
    assert "not a pass and not a finding" in capsys.readouterr().err


def test_a_finding_exits_one_and_a_healthy_pair_exits_zero(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-and-never-sent")
    # `google-genai` is a runtime dependency of this repo, so the import guard in
    # `main` is satisfied here without a network or a real key.
    pytest.importorskip("google.genai")

    monkeypatch.setattr(contract, "PROBES", (probe(GOOD_PLAN),))
    assert contract.main() == 0

    monkeypatch.setattr(contract, "PROBES", (probe(GOOD_PLAN), probe(FENCED_PLAN)))
    assert contract.main() == 1
