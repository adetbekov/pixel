"""Tests for scripts/assert_required_checks.py (JEB-1523).

The script's whole value is that it fails on drift, so the tests that matter are
the ones that make it fail: a context dropped from protection, a renamed job, a
context required on GitHub that the map does not know about, an unreachable
trigger. A test suite that only proved the happy path would pass just as well
against a script that always printed OK.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "assert_required_checks.py"
_WORKFLOWS = _ROOT / ".github" / "workflows"
_AUDIT = _WORKFLOWS / "required-checks-audit.yml"


def _load():
    """Import the script by path — `scripts/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("assert_required_checks", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


arc = _load()


CI_ON_BOTH = """
name: CI
on:
  pull_request:
    branches: [dev, main]
jobs:
  test:
    name: lint + tests
    runs-on: ubuntu-latest
  frontend:
    name: frontend lint
    runs-on: ubuntu-latest
"""

MAP = {"lint + tests": "ci.yml", "frontend lint": "ci.yml"}


@pytest.fixture
def workflows(tmp_path, monkeypatch):
    """A workflows dir whose ci.yml declares both audited contexts."""
    directory = tmp_path / "workflows"
    directory.mkdir()
    (directory / "ci.yml").write_text(CI_ON_BOTH)
    monkeypatch.setattr(arc, "REQUIRED_CONTEXTS", {"dev": dict(MAP), "main": dict(MAP)})
    # A `pull_request` checkout stands for the copy GitHub reads, which is what
    # the reachability assertion needs the local tree to mean.
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    return directory


def _check(workflows, contexts, *, protected=True, base=CI_ON_BOTH, branch="dev", **kwargs):
    return arc.check(
        contexts,
        protected,
        workflows,
        base_workflows={"ci.yml": base},
        branch=branch,
        **kwargs,
    )


# --- the map is about this repository, not a copy of another one -------------


def test_map_covers_dev_and_main():
    assert sorted(arc.REQUIRED_CONTEXTS) == ["dev", "main"]


@pytest.mark.parametrize("branch", ["dev", "main"])
def test_every_asserted_context_is_a_real_job_name_in_the_file_it_names(branch):
    """The rename this script exists to catch must not already have happened.

    Each entry is checked against the workflow *it* names, not against one
    hardcoded file: `main` is served by two of them already, and pinning the
    filename would make adding a third gate fail here instead of where it
    belongs.
    """
    for context, workflow_file in arc.REQUIRED_CONTEXTS[branch].items():
        path = _WORKFLOWS / workflow_file
        assert path.exists(), f"{context!r} names a workflow that is not in the repo"
        names, _unresolved = arc.job_display_names(arc.parse_workflow(path.read_text()))
        assert context in names


@pytest.mark.parametrize("branch", ["dev", "main"])
def test_every_asserted_context_is_reachable_for_the_branch_it_is_asserted_on(branch):
    """A context no PR into `branch` can produce would wedge `branch`.

    This is what keeps `dev`'s map a strict subset of `main`'s rather than a
    copy of it: `main PRs must come from dev` is declared
    `on: pull_request: branches: [main]`, so requiring it on `dev` would leave
    every PR into `dev` blocked on a context nothing reports.
    """
    for context, workflow_file in arc.REQUIRED_CONTEXTS[branch].items():
        doc = arc.parse_workflow((_WORKFLOWS / workflow_file).read_text())
        assert arc.reachability_findings(context, workflow_file, doc, doc, branch) == []


@pytest.mark.parametrize("branch", ["dev", "main"])
def test_the_audit_never_asserts_itself(branch):
    """The audit reads protection, so it must not be able to block it.

    Listing its own job name in the map would be the first step toward making it
    a required check, which is the one thing this workflow must never become.
    """
    audit_name = yaml.safe_load(_AUDIT.read_text())["jobs"]["audit"]["name"]
    assert audit_name not in arc.REQUIRED_CONTEXTS[branch]
    assert "required-checks-audit.yml" not in arc.REQUIRED_CONTEXTS[branch].values()


def test_audit_workflow_triggers_on_schedule_and_workflow_paths():
    doc = arc.parse_workflow(_AUDIT.read_text())
    on = arc.workflow_on_block(doc)
    assert "cron" in on["schedule"][0]
    assert ".github/workflows/**" in on["pull_request"]["paths"]


# --- assertion 1: membership -------------------------------------------------


def test_green_when_every_context_is_required(workflows):
    assert _check(workflows, ["lint + tests", "frontend lint"]) == []


def test_missing_context_is_a_finding(workflows):
    findings = _check(workflows, ["lint + tests"])
    assert len(findings) == 1
    assert "'frontend lint'" in findings[0]
    assert "missing from dev's protection" in findings[0]


def test_unprotected_branch_is_a_finding(workflows):
    findings = _check(workflows, [], protected=False)
    assert "not branch-protected at all" in findings[0]
    # ...and every mapped context is still reported as unenforced.
    assert len(findings) == 3


# --- assertion 2: context <-> job name --------------------------------------


def test_renamed_job_orphans_its_context(workflows):
    """The defect the whole file exists for: the context is matched by name."""
    (workflows / "ci.yml").write_text(CI_ON_BOTH.replace("frontend lint", "frontend checks"))
    findings = _check(workflows, ["lint + tests", "frontend lint"], base=None)
    assert len(findings) == 1
    assert "matches no job name" in findings[0]
    assert "'frontend lint'" in findings[0]


def test_deleted_workflow_is_a_finding(workflows):
    (workflows / "ci.yml").unlink()
    findings = _check(workflows, ["lint + tests", "frontend lint"])
    assert all("is gone" in finding for finding in findings)


def test_unexpandable_job_name_is_a_note_not_a_finding(workflows):
    (workflows / "ci.yml").write_text(
        CI_ON_BOTH.replace("name: frontend lint", "name: frontend ${{ matrix.tool }}")
    )
    notes: list[str] = []
    findings = _check(workflows, ["lint + tests", "frontend lint"], base=None, notes=notes)
    assert findings == []
    assert "could not be verified" in notes[0]


# --- assertion 3: the reverse direction (JEB-1227) ---------------------------


def test_context_required_on_github_but_absent_from_the_map_is_a_finding(workflows):
    findings = _check(workflows, ["lint + tests", "frontend lint", "image build"])
    assert len(findings) == 1
    assert "'image build'" in findings[0]
    assert "is not asserted here" in findings[0]


# --- assertion 4: trigger reachability ---------------------------------------


def test_pull_request_target_on_head_while_base_still_has_pull_request(workflows):
    """Neither copy is wrong alone; the pair produces nothing (JEB-1393)."""
    head = CI_ON_BOTH.replace("pull_request:", "pull_request_target:")
    (workflows / "ci.yml").write_text(head)
    findings = _check(workflows, ["lint + tests", "frontend lint"], base=CI_ON_BOTH)
    assert len(findings) == 2
    assert all("is unreachable" in finding for finding in findings)


def test_branch_filter_that_excludes_the_audited_branch_is_a_finding(workflows):
    narrowed = CI_ON_BOTH.replace("branches: [dev, main]", "branches: [main]")
    (workflows / "ci.yml").write_text(narrowed)
    findings = _check(workflows, ["lint + tests", "frontend lint"], base=narrowed, branch="dev")
    assert all("is unreachable" in finding for finding in findings)
    # ...and the same pair is reachable for the branch the filter does name.
    assert _check(workflows, ["lint + tests", "frontend lint"], base=narrowed, branch="main") == []


def test_types_that_cannot_fire_while_the_pr_is_open_is_a_finding(workflows):
    closed = CI_ON_BOTH.replace(
        "    branches: [dev, main]", "    branches: [dev, main]\n    types: [closed]"
    )
    (workflows / "ci.yml").write_text(closed)
    findings = _check(workflows, ["lint + tests", "frontend lint"], base=closed)
    assert all("is unreachable" in finding for finding in findings)


def test_path_filter_is_a_note_not_a_finding(workflows):
    filtered = CI_ON_BOTH.replace(
        "    branches: [dev, main]", "    branches: [dev, main]\n    paths: ['backend/**']"
    )
    (workflows / "ci.yml").write_text(filtered)
    notes: list[str] = []
    findings = _check(workflows, ["lint + tests", "frontend lint"], base=filtered, notes=notes)
    assert findings == []
    assert all("path filter" in note for note in notes)


def test_reachability_is_skipped_when_the_base_copy_was_not_fetched(workflows):
    """No offline substitute for the base copy — and no guess at one."""
    findings = arc.check(
        ["lint + tests", "frontend lint"], True, workflows, base_workflows=None, branch="dev"
    )
    assert findings == []


# --- cannot-determine is never a pass and never a finding --------------------


def test_unknown_branch_raises_rather_than_asserting_an_empty_map():
    with pytest.raises(arc.CannotDetermine):
        arc.contexts_for("release/1.0")


def test_unparseable_workflow_is_undetermined_not_a_finding(workflows):
    (workflows / "ci.yml").write_text("name: CI\njobs: [\n")
    undetermined: list[str] = []
    findings = _check(
        workflows, ["lint + tests", "frontend lint"], base=None, undetermined=undetermined
    )
    assert findings == []
    assert undetermined and "not valid YAML" in undetermined[0]


def test_protected_true_without_a_protection_block_is_undetermined():
    """"The token cannot see protection" is not "the branch is unprotected"."""
    with pytest.raises(arc.CannotDetermine):
        arc.required_contexts({"protected": True})


def test_protected_false_reads_as_an_empty_live_list():
    assert arc.required_contexts({"protected": False}) == []


def test_missing_protected_key_is_undetermined():
    with pytest.raises(arc.CannotDetermine):
        arc.required_contexts({"name": "dev"})


def test_contexts_come_from_the_branch_object_not_the_admin_only_endpoint():
    payload = {
        "protected": True,
        "protection": {"required_status_checks": {"contexts": ["lint + tests"]}},
    }
    assert arc.required_contexts(payload) == ["lint + tests"]


# --- YAML's `on:` key resolves to the boolean True ---------------------------


def test_on_block_is_read_through_yaml_true():
    doc = arc.parse_workflow("on:\n  pull_request:\n    branches: [dev]\n")
    assert arc.workflow_on_block(doc) == {"pull_request": {"branches": ["dev"]}}


def test_quoted_on_key_is_read_too():
    doc = arc.parse_workflow('"on":\n  pull_request:\n    branches: [dev]\n')
    assert arc.workflow_on_block(doc) == {"pull_request": {"branches": ["dev"]}}


@pytest.mark.parametrize(
    ("patterns", "ref", "expected"),
    [
        (["dev", "main"], "dev", True),
        (["main"], "dev", False),
        (["releases/**"], "releases/1/2", True),
        (["releases/*"], "releases/1/2", False),
        (["**", "!dev"], "dev", False),
    ],
)
def test_branch_filter_patterns(patterns, ref, expected):
    assert arc._filter_matches(patterns, ref) is expected
