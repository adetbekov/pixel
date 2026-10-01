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
import subprocess
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
    {
        "reply": "Кручусь!",
        "handled": True,
        "actions": [{"action": "spin"}, {"action": "say", "text": "Кручусь!"}],
    },
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
    """A call that blows up on something other than "the model never answered"
    is still the call failing on the shape we send it."""

    def boom():
        raise ValueError("the SDK rejected the config this repo sends")

    finding = contract.run(
        contract.Probe(
            name="GeminiSkillGenerator._call",
            call=boom,
            shape_of=contract.GeminiSkillGenerator._call,
            parse=contract.SkillDraft.model_validate_json,
        )
    )

    assert finding is not None
    assert "ValueError: the SDK rejected the config this repo sends" in finding


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


# --- JEB-1650: exit 1 means a parse finding and nothing else -----------------
#
# Two runs on `main` reported "contract broken" having judged no answer at all:
# 36115530738 died on `import numpy` (a traceback is exit 1 in Python, the code
# reserved for a finding) and 36801077007 reported a `ReadTimeout` plus a
# `503 UNAVAILABLE` as "did not return an answer this repo can parse" — which is
# true only in the sense that nothing was returned. Both now classify as
# "unchecked", and the alert step can tell the three states apart.


def read_timeout():
    """The transport failure seen on run 36801077007, verbatim message."""
    httpx = pytest.importorskip("httpx")
    return httpx.ReadTimeout("The read operation timed out")


def overloaded_error():
    """The other half of that run: the model is up, but has no capacity now."""
    errors = pytest.importorskip("google.genai.errors")
    return errors.ServerError(
        503,
        {
            "error": {
                "code": 503,
                "status": "UNAVAILABLE",
                "message": "This model is currently experiencing high demand...",
            }
        },
    )


def test_a_transport_timeout_is_not_a_contract_finding():
    """Nothing came back, so the shape of the answer was never on the table. The
    old message said the answer "did not parse" about an answer that never was."""
    with pytest.raises(contract.ProbeUnavailable) as raised:
        contract.run(raising_probe(read_timeout()))

    reason = str(raised.value)
    assert "ReadTimeout" in reason
    assert "unchecked" in reason
    assert "GeminiTeacher._call" in reason


def test_an_overloaded_model_is_not_a_contract_finding():
    """`503 UNAVAILABLE` is the same class as 429: the call was never served. A
    500 `INTERNAL` is not — a model took that one and broke on it."""
    with pytest.raises(contract.ProbeUnavailable) as raised:
        contract.run(raising_probe(overloaded_error()))

    reason = str(raised.value)
    assert "503" in reason
    assert "UNAVAILABLE" in reason
    assert "no capacity" in reason


def test_a_bad_request_stays_a_finding():
    """The call shape itself being rejected is exactly what this gate watches
    for, so a 400 must not be swept into "not checked" with the refusals."""
    errors = pytest.importorskip("google.genai.errors")
    bad_request = errors.ClientError(
        400,
        {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "response_schema"}},
    )

    finding = contract.run(raising_probe(bad_request))

    assert finding is not None
    assert "ClientError" in finding


def test_a_timeout_is_retried_but_a_quota_refusal_is_not():
    """Opposite windows: a quota day cannot clear between two calls, a busy
    minute can. So one is given the second attempt and the other is not."""
    timeouts: list[int] = []
    with pytest.raises(contract.ProbeUnavailable):
        contract.run(raising_probe(read_timeout(), timeouts))
    assert len(timeouts) == contract.ATTEMPTS

    quota: list[int] = []
    with pytest.raises(contract.ProbeUnavailable):
        contract.run(raising_probe(quota_error(), quota))
    assert len(quota) == 1


def test_an_answer_after_a_timeout_is_a_pass():
    """The retry exists to be used: a timed-out first call followed by a parsable
    answer is the contract holding, not a degraded result."""
    outcomes = [read_timeout(), GOOD_PLAN]

    def flaky():
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    probe_with_a_hiccup = contract.Probe(
        name="GeminiTeacher._call",
        call=flaky,
        shape_of=contract.GeminiTeacher._call,
        parse=contract.TeacherPlan.model_validate_json,
    )

    assert contract.run(probe_with_a_hiccup) is None


def test_one_bad_answer_beside_an_unanswered_call_is_unchecked():
    """Production retries, so a single fenced answer is a state the app survives
    — the finding needs both attempts. With the other attempt never answered,
    there is no second data point, so the verdict is "unchecked", with both
    attempts still spelled out for the log."""
    outcomes = [read_timeout(), FENCED_PLAN]

    def flaky():
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    with pytest.raises(contract.ProbeUnavailable) as raised:
        contract.run(
            contract.Probe(
                name="GeminiTeacher._call",
                call=flaky,
                shape_of=contract.GeminiTeacher._call,
                parse=contract.TeacherPlan.model_validate_json,
            )
        )

    reason = str(raised.value)
    assert "ReadTimeout" in reason
    assert "did not parse" in reason


def test_a_timed_out_night_exits_two(monkeypatch, capsys):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-and-never-sent")
    pytest.importorskip("google.genai")

    monkeypatch.setattr(
        contract, "PROBES", (raising_probe(read_timeout()), raising_probe(overloaded_error()))
    )

    assert contract.main() == contract.EXIT_UNCHECKED
    assert "not a pass and not a finding" in capsys.readouterr().err


def test_a_crash_in_the_harness_exits_three_not_one(monkeypatch, capsys):
    """`run` classifies everything the call and the parser can raise, so anything
    escaping it is this script misbehaving — which says nothing about the model's
    answers and must not be reported as a finding."""
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-and-never-sent")
    pytest.importorskip("google.genai")

    def not_a_parser(_raw):
        raise TypeError("the harness called the parser wrong")

    broken = contract.Probe(
        name="GeminiTeacher._call",
        call=lambda: GOOD_PLAN,
        shape_of=contract.GeminiTeacher._call,
        parse=not_a_parser,
    )

    monkeypatch.setattr(contract, "PROBES", (broken, probe(FENCED_PLAN)))

    assert contract.main() == contract.EXIT_CRASHED
    err = capsys.readouterr().err
    assert "THE PROBE ITSELF FAILED" in err
    assert "TypeError" in err


def test_cli_never_lets_a_traceback_leave_as_a_finding(monkeypatch, capsys):
    """Python's exit code for an unhandled exception is 1 — the code this gate
    reserves for "a shape's answer no longer parses". `cli` is the remap."""

    def boom(_argv=None):
        raise RuntimeError("something nobody anticipated")

    monkeypatch.setattr(contract, "main", boom)

    assert contract.cli() == contract.EXIT_CRASHED
    assert "RuntimeError" in capsys.readouterr().err


def test_cli_passes_a_normal_verdict_through(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key-and-never-sent")
    pytest.importorskip("google.genai")

    monkeypatch.setattr(contract, "PROBES", (probe(GOOD_PLAN),))
    assert contract.cli() == contract.EXIT_OK

    monkeypatch.setattr(contract, "PROBES", (probe(FENCED_PLAN),))
    assert contract.cli() == contract.EXIT_FINDING


def test_an_unresolved_import_exits_three_not_one():
    """Case 1, reproduced: run 36115530738 died on `import numpy` four modules
    below `backend.miner.case` and exited 1, so the alert step opened "a call
    shape's answer no longer parses" without a call having been placed. The
    import is fixed (JEB-1601); this asserts the classification, by taking numpy
    away again in a subprocess."""
    blocker = f"""
import runpy, sys
from importlib.abc import MetaPathFinder

class Blocked(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        return None

sys.meta_path.insert(0, Blocked())
sys.argv = [{str(SCRIPT)!r}]
runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3, result.stderr
    assert "No module named 'numpy'" in result.stderr
    assert "not a pass and not a finding" in result.stderr


def alert_step() -> str:
    """The body of the workflow step that turns an exit code into an issue."""
    workflow = (ROOT / ".github" / "workflows" / "live-gemini-contract.yml").read_text()
    return workflow.split("Open, update or close the alert issue", 1)[1]


def test_the_alert_step_routes_exit_three_to_failed_before_it_could_probe():
    """The script's own "I fell over" code must not share an arm with a finding.
    It shares one with an empty PROBE_EXIT, which is the same state a step
    earlier."""
    step = alert_step()
    finding_arm = step.split("\n            1)\n", 1)[1].split("\n            2)\n", 1)[0]
    crash_arm = step.split("\n            3|*)\n", 1)[1].split("\n          esac", 1)[0]

    assert "no longer parses" in finding_arm
    assert "failed before it could probe" in crash_arm


def test_the_exit_two_alert_names_the_timeout_and_the_overload():
    """A reader of the exit-2 issue needs to know a 503 or a timeout lands here
    too, or they go looking for a missing key that is not missing."""
    step = alert_step()
    branch = step.split("\n            2)\n", 1)[1].split("\n            3|*)\n", 1)[0]

    assert "503" in branch
    assert "UNAVAILABLE" in branch
    assert "ReadTimeout" in branch


def test_the_alert_issue_is_only_reused_for_the_same_exit_code():
    """Case 3: the old step reused *any* open issue with the label, so a quota
    night commented "Still failing" on an issue about a parse finding, and a real
    finding could only ever arrive as a comment on a false one."""
    step = alert_step()

    assert 'MARKER="<!-- live-gemini-contract exit=${PROBE_EXIT:-none} -->"' in step
    # The lookup filters on that marker rather than taking the first open issue.
    assert "contains(env.MARKER)" in step
    assert "--json number,body" in step


def test_a_green_run_closes_the_open_alert():
    """An alert that outlives its cause is the failure mode: #71 stayed open
    through three green nights, blocking any new issue the whole time."""
    workflow = (ROOT / ".github" / "workflows" / "live-gemini-contract.yml").read_text()
    step = alert_step()

    # It cannot close anything if it only runs on failure.
    assert "if: failure() && github.event_name == 'schedule'" not in workflow
    assert "!cancelled() && github.event_name == 'schedule'" in workflow
    assert 'if [ "${PROBE_EXIT:-none}" = "0" ]; then' in step
    assert "gh issue close" in step


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
    branch = workflow.split("\n            2)\n", 1)[1].split("\n            3|*)\n", 1)[0]

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


# --- JEB-1601: the gate's own environment -----------------------------------
#
# The probe's verdict was tested above; what was not, and what took the nightly
# down on its first execution ever, is whether the job can reach that verdict at
# all. `schedule:` and `workflow_dispatch` resolve only from the default branch,
# so the file first executed on `main` — and died on `import numpy`, four modules
# below `backend.miner.case`, which the install step's hand-typed list never
# mentioned because the script never names it.

REQUIREMENTS_SCRIPT = ROOT / ".github" / "scripts" / "live_contract_requirements.py"

_req_spec = importlib.util.spec_from_file_location(
    "live_contract_requirements", REQUIREMENTS_SCRIPT
)
requirements = importlib.util.module_from_spec(_req_spec)
sys.modules[_req_spec.name] = requirements
_req_spec.loader.exec_module(requirements)


def project_dependencies() -> list[str]:
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return pyproject["project"]["dependencies"]


def test_the_nightly_installs_every_project_dependency_but_the_excluded_ones():
    """The list is pyproject's, not a remembered one. numpy is the regression:
    it is a project dependency, it is imported transitively by the probe, and it
    was the module the gate died on."""
    nightly = requirements.requirements(ROOT / "pyproject.toml")
    installed = {requirements.requirement_name(spec) for spec in nightly}
    declared = {requirements.requirement_name(spec) for spec in project_dependencies()}

    assert installed == declared - requirements.EXCLUDED
    assert "numpy" in installed
    assert "laya" not in installed


def test_the_exclusion_list_stays_justified():
    """Only laya is dropped, and only because torch is a gigabyte. Anything added
    to EXCLUDED has to be argued from the import graph — this test is the place
    that stops it from being argued from convenience."""
    assert requirements.EXCLUDED == frozenset({"laya"})


def test_requirement_specifiers_are_passed_through_verbatim():
    """A pin dropped on the way into the nightly would let it run against an SDK
    line the app is not on — the drift the old step's comment already worried
    about. So the strings are copied, not rebuilt from names."""
    installed = requirements.requirements(ROOT / "pyproject.toml")

    assert [spec for spec in project_dependencies() if not spec.startswith("laya")] == installed
    assert any(spec.startswith("google-genai>=") for spec in installed)


def test_check_imports_resolves_without_a_key(monkeypatch, capsys):
    """The PR-side mode. It must return before the key is read, or it cannot run
    on a fork PR — which is the only place it can catch the defect early."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert contract.main(["--check-imports"]) == 0
    assert "imports resolve" in capsys.readouterr().out


def test_nothing_the_probe_imports_needs_laya():
    """Why the nightly may skip laya (and its ~1 GB of torch): no module on the
    probe's import path says `import laya` at import time. `backend/brain/engine.py`
    keeps those imports inside the two methods that load a model, and this asserts
    that invariant from the outside — in a subprocess, because the dev environment
    has laya installed and would hide a regression here."""
    blocker = f"""
import runpy, sys
from importlib.abc import MetaPathFinder

class Blocked(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "laya" or name.startswith("laya.") or name == "torch":
            raise ImportError("the nightly gate does not install " + name)
        return None

sys.meta_path.insert(0, Blocked())
sys.argv = [{str(SCRIPT)!r}, "--check-imports"]
runpy.run_path({str(SCRIPT)!r}, run_name="__main__")
"""
    result = subprocess.run(
        [sys.executable, "-c", blocker],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "imports resolve" in result.stdout


def test_both_workflows_install_from_the_same_script():
    """The PR check is only evidence about the nightly while both install the
    same way. Two copies of an install step drift, and the drift would be
    invisible until the nightly ran — which is the failure mode itself."""
    nightly = (ROOT / ".github" / "workflows" / "live-gemini-contract.yml").read_text()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    reference = ".github/scripts/live_contract_requirements.py"

    assert reference in nightly
    assert reference in ci
    assert "--check-imports" in ci


def test_the_nightly_does_not_name_packages_by_hand():
    """The defect in one line: the install step listed what the author remembered
    the script imports. If a package name reappears in a `pip install` there, the
    single source is gone again."""
    nightly = (ROOT / ".github" / "workflows" / "live-gemini-contract.yml").read_text()
    install_commands = [
        line
        for line in nightly.splitlines()
        if "pip install" in line and not line.lstrip().startswith("#")
    ]

    assert install_commands
    for line in install_commands:
        assert "-r live-contract-requirements.txt" in line, line
