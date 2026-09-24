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
