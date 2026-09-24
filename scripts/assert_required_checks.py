#!/usr/bin/env python3
"""
assert_required_checks.py — assert that a protected branch's protection still
*requires* every check the path through it depends on, and that those required
context strings still match the job names that actually produce them.

Audits `main` and `dev`; `--branch` picks which, and REQUIRED_CONTEXTS is keyed
by the same name. Both branches matter here for different reasons: every feature
PR lands on `dev`, and `main` is deploy-on-merge.

Why this exists (JEB-1523). The required-check list lives on GitHub, not in the
diff: editing it leaves no commit, no PR and no red check. So renaming a job in
`ci.yml` silently orphans the required context that was matched to it **by
name** — every later PR sits forever on "Expected — waiting for status to be
reported" — and dropping a context from protection shows up nowhere at all.
Nothing in the repository notices either move. The four sibling studio repos
close this with this script plus `.github/workflows/required-checks-audit.yml`;
this is pixel's copy.

Four independent assertions, because they fail independently:

  1. **Required-list membership.** Each context in the audited branch's map is
     in that branch's required status checks. Someone dropping one — by hand in
     the UI, or by a full `PUT .../protection` that rebuilds the object and
     forgets a context — is caught within a day.
  2. **Context <-> job-name agreement.** Each context string is the `name:` of a
     job in the workflow file that is supposed to report it. A required context
     is matched to a check run *by name*: rename the job and the context never
     reports, which leaves every PR into that branch permanently blocked.

     A required context need not come from a workflow job at all: `gates
     recorded` on `dev` is a **commit status** posted over
     `POST /repos/.../statuses/<sha>` by the PR auto-merge bot (JEB-1571), and
     the workflow variant of that publisher (#50) was closed unmerged. Such a
     context is mapped to the `EXTERNAL_STATUS` sentinel instead of a filename,
     and assertions 2 and 4 — both of which ask questions about a *workflow
     file* — are skipped for it. Assertions 1 and 3 are not: membership in the
     live required list is exactly what must keep being asserted, and that is
     the whole reason the entry exists. See the comment on REQUIRED_CONTEXTS for
     the rule that keeps the sentinel from spreading to workflow-backed
     contexts.
  3. **The reverse direction (JEB-1227).** Every context in the branch's live
     required list is in that branch's map. Assertions 1 and 2 only walk the
     map, so a context required on GitHub but absent from the dict is invisible:
     the script prints `OK` while a real gate goes unmonitored, and can later be
     removed silently — the exact drift this file exists to catch.

     Assertion 3 is a **finding (exit 1)**, not a `::warning::` note, even
     though an unknown-but-required context does not break a gate the way an
     orphaned context does. The audit's job is to fail loudly on drift, a
     warning in a scheduled run is read by nobody, and the remedy is a one-line
     dict edit in the PR that changed the protection. It reports on the *live*
     list only, so it cannot fire on the exit-2 path: a read that failed raises
     CannotDetermine before `check()` is reached, and an unreadable list is
     never treated as an empty one.
  4. **Trigger reachability (JEB-1393).** The workflow that declares a required
     context must have a trigger that can actually fire for a PR targeting the
     audited branch. Assertions 1-3 all stop at *declared*: the context is
     required, and some job somewhere is named after it. None of them asks
     whether any event can ever start that job.

     Which copy of the file GitHub reads differs per event:

       * `pull_request` is read from the PR's **merge ref** — the head's copy.
       * `pull_request_target` is read from the **base branch's** copy.

     So the set of events that can produce a context on a PR into branch B is
     `{pull_request if the head copy declares it reachable-for-B}` u
     `{pull_request_target if B's copy declares it reachable-for-B}`. Carry
     `pull_request_target` on the head while B still carries `pull_request` and
     that set is empty: neither event has a definition that declares the job, so
     it never runs, and the aggregate check status still reads green. Neither
     copy is wrong on its own — the defect lives only in the relation between
     them, which is why a reviewer reading one diff cannot see it.

     This script evaluates exactly that set. The **head copy** is the local
     checkout (on a `pull_request` run `actions/checkout` gives the merge ref,
     which is the copy GitHub itself would use; on the scheduled run from the
     default branch it is simply that branch's copy). The **base copy** is
     fetched over the API at `?ref=<branch>`. `branches` / `branches-ignore`
     filter patterns are evaluated against the audited branch, and a `types:`
     list that cannot fire while a PR is open (`[closed]`) counts as unreachable
     — a check that only runs after the merge can never be satisfied before it.

     `paths` / `paths-ignore` on a reachable trigger are reported as a
     `::warning::` note, not a finding: a path filter narrows *which* PRs
     produce the context rather than making it unproducible, so it is a real
     hazard for a required check but not the defect this assertion is about. A
     trigger whose shape cannot be evaluated at all — a `${{ }}` expression
     where a branch filter belongs — is a note too. A workflow file that does
     not *parse* is neither: it is exit 2, because a note there would leave the
     gate unevaluated while the audit reported green, which is this issue's own
     failure shape.

A job `name:` carrying an unexpandable `${{ }}` template is reported as a
`::warning::` note ("cannot verify"), never a finding: membership stays
asserted, and an answer that could not be read is never reported as a failed
answer. pixel has no matrix job producing a required context, so no expander is
carried here — if one is ever added, port `_matrix_combinations()` from
`adetbekov/split-the-bill` rather than turning the note into a finding.

Transport is one read-only GET against `GET /repos/{repo}/branches/{branch}`,
**not** `GET /repos/{repo}/branches/{branch}/protection`. That is deliberate:
the `/protection` endpoint requires **admin** on the repo, which the Actions
`GITHUB_TOKEN` does not have and cannot be granted (there is no
`administration:` key in a workflow's `permissions:` block). The branch object
carries the same `required_status_checks.contexts` list under its `protection`
key and needs only read access, so this runs on the built-in token with no PAT
secret to provision, rotate, or leak.

Exit codes, kept distinct on purpose — an error is NOT "unprotected":

  0  every required context is present, matches its job name, and is reachable
     by a trigger that fires for PRs into the audited branch
  1  a real finding: a context is missing from the required list, a context no
     longer matches any job name, a context is declared by a workflow no event
     can start for a PR into that branch, or the branch is not protected
  2  cannot determine: the read failed, returned no protection block, returned a
     workflow file that does not parse, or `--branch` names a branch this script
     has no context map for. This is NOT "unprotected" and must never be
     reported as one — it means the answer could not be read.

Usage:
    python3 scripts/assert_required_checks.py [--repo owner/name] [--branch main|dev]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / ".github" / "workflows"


class _ExternalStatus:
    """Producer sentinel: required, and deliberately not a workflow job.

    Its own type rather than a magic string, so it can never be mistaken for a
    filename: `workflows_dir / EXTERNAL_STATUS` raises instead of quietly looking
    for `.github/workflows/EXTERNAL_STATUS` and reporting the context as orphaned
    because its imaginary file is absent.
    """

    def __repr__(self) -> str:
        return "EXTERNAL_STATUS"


EXTERNAL_STATUS = _ExternalStatus()

Producer = str | _ExternalStatus

# Every context required on a protected branch, mapped to the workflow file whose
# job `name:` must equal it. Adding a required context on GitHub without adding
# it here leaves the new gate unmonitored (assertion 3 fails on exactly that);
# removing one here silently retires the assertion, which is the drift this
# script exists to catch — so **both edits belong in the same PR as the
# protection change**.
#
# A value of EXTERNAL_STATUS instead of a filename means "required, and produced
# by something other than a workflow job". It is an escape hatch from assertions
# 2 and 4 only, and it must stay narrow: a context that IS a workflow job name
# marked EXTERNAL_STATUS would silently stop being checked for renames, which is
# assertion 2's whole job. `test_external_status_is_never_used_for_a_workflow_job`
# fails on exactly that — it rejects any EXTERNAL_STATUS context that matches a
# job `name:` anywhere in .github/workflows/. Every such entry must name its
# publisher in a comment, so a reader can find the thing that posts it.
#
# Keyed by branch, because `main` and `dev` need not gate the same set. The
# audited branch selects the map; `--branch` names it.
#
# All three of pixel's CI jobs are declared `on: pull_request: branches: [dev,
# main]`, so each reports for a PR into either branch and both lists carry them.
# `main` is that trio plus the release-only gate, which `dev` must NOT require:
#
#   * `main PRs must come from dev` — guard-main-head.yml is
#     `on: pull_request: branches: [main]` (JEB-1522), so no PR into `dev` can
#     start it. Requiring it on `dev` would leave every such PR blocked on a
#     context nothing reports.
#
# `image build` (JEB-1518) was deliberately absent from both lists until the job
# was on `dev` — asserting a context that is not yet required is a finding, and
# requiring one no PR into that branch can produce would wedge the branch. It
# landed with #17, reported on live PRs into both branches, and JEB-1524 made it
# required on `dev` and `main`; this dict edit is that change's other half.
REQUIRED_CONTEXTS = {
    "main": {
        "lint + tests": "ci.yml",
        "frontend lint": "ci.yml",
        "image build": "ci.yml",
        # Produced by the `live-contract-imports` job in ci.yml (JEB-1601, #63);
        # required on both branches by JEB-1605. It belongs in the required list
        # for the same reason the three above do and `Live Gemini Contract` and
        # `Required Checks Audit` do not: those two read the outside world — the
        # live model API and GitHub's own protection — and can go red for a
        # reason that is not in the diff, so gating a merge on them would hand
        # an outage a veto. This job is hermetic: it installs the extras the
        # live-contract probe imports and resolves the import graph, with no key
        # and no call to a model, so red means the diff broke an import.
        "live gemini contract imports": "ci.yml",
        "main PRs must come from dev": "guard-main-head.yml",
    },
    "dev": {
        "lint + tests": "ci.yml",
        "frontend lint": "ci.yml",
        "image build": "ci.yml",
        # Same job, same rationale as the `main` entry above (JEB-1605): the
        # `live-contract-imports` job in ci.yml, hermetic, so a red one is a
        # real defect in the diff and worth a server-side gate. Every feature PR
        # lands here, so this is the copy that does the day-to-day blocking.
        "live gemini contract imports": "ci.yml",
        # Not a workflow job (JEB-1596): the PR auto-merge bot posts this as a
        # commit status via POST /repos/adetbekov/pixel/statuses/<sha> once
        # TechLead APPROVED and the qa_gate comment both match the current head
        # (JEB-1571). The workflow variant of that publisher, #50, was closed
        # unmerged, so no .github/workflows/ file declares it and none should be
        # invented to satisfy this map. Asserted here because PATCH of
        # .../branches/dev/protection/required_status_checks succeeds on the
        # studio PAT and leaves no trace on any PR — this audit is the only thing
        # that would notice the lock being un-armed.
        "gates recorded": EXTERNAL_STATUS,
    },
}

# Any `${{ ... }}` expression. This script does not expand them — see the
# module docstring on why an unexpandable name is a note, not a finding.
_ANY_EXPR = re.compile(r"\$\{\{.*?\}\}")

EXIT_OK = 0
EXIT_FINDING = 1
EXIT_CANNOT_DETERMINE = 2


class CannotDetermine(Exception):
    """The answer could not be read. Never a synonym for 'not protected'."""


def contexts_for(branch: str) -> dict[str, Producer]:
    """The `context -> producer` map this script asserts for `branch`.

    A producer is the workflow filename whose job `name:` must equal the context,
    or EXTERNAL_STATUS for a context posted by something that is not a workflow
    job.

    An unaudited branch is CannotDetermine, not an empty map: an empty map would
    make every assertion vacuously true and the script would print `OK` for a
    branch it knows nothing about — the fail-open shape this file exists to
    prevent.
    """
    try:
        return REQUIRED_CONTEXTS[branch]
    except KeyError:
        raise CannotDetermine(
            f"no asserted context map for branch {branch!r} — "
            f"scripts/assert_required_checks.py audits {sorted(REQUIRED_CONTEXTS)}. "
            f"Add a {branch!r} entry to REQUIRED_CONTEXTS in the same PR as the "
            f"protection change, or audit one of those branches."
        ) from None


def workflow_producers(asserted: dict[str, Producer]) -> set[str]:
    """The workflow filenames in a context map, without the EXTERNAL_STATUS ones."""
    return {producer for producer in asserted.values() if isinstance(producer, str)}


def describe_producer(producer: Producer) -> str:
    """How a finding should refer to whatever is supposed to report a context."""
    if isinstance(producer, str):
        return f"produced by .github/workflows/{producer}"
    return (
        "posted as a commit status by something other than a workflow job — see the "
        "comment on its REQUIRED_CONTEXTS entry for the publisher"
    )


def _get(url: str, accept: str, token: str | None) -> str:
    req = urllib.request.Request(url)
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode()


def fetch_branch(repo: str, branch: str, token: str | None) -> dict:
    url = f"https://api.github.com/repos/{repo}/branches/{branch}"
    try:
        return json.loads(_get(url, "application/vnd.github+json", token))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        raise CannotDetermine(f"GET {url} -> HTTP {exc.code}: {exc.reason}") from exc
    except OSError as exc:  # pragma: no cover - network path
        raise CannotDetermine(f"GET {url} failed: {exc}") from exc


def fetch_workflow_source(repo: str, ref: str, filename: str, token: str | None) -> str | None:
    """The raw text of `.github/workflows/<filename>` on `ref`, or None if absent.

    A 404 is a real answer — the file is not on that ref — and assertion 4
    handles it as such. Any other transport failure is CannotDetermine: the audit
    must never read "I could not fetch the base copy" as "the base copy has no
    trigger".
    """
    url = f"https://api.github.com/repos/{repo}/contents/.github/workflows/{filename}?ref={ref}"
    try:
        return _get(url, "application/vnd.github.raw+json", token)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        if exc.code == 404:
            return None
        raise CannotDetermine(f"GET {url} -> HTTP {exc.code}: {exc.reason}") from exc
    except OSError as exc:  # pragma: no cover - network path
        raise CannotDetermine(f"GET {url} failed: {exc}") from exc


def fetch_base_workflows(repo: str, ref: str, token: str | None) -> dict[str, str | None]:
    """`filename -> source-on-ref` for every workflow `ref`'s context map names.

    Only the files that declare a required context are fetched — the same set
    assertions 1 and 2 already walk. EXTERNAL_STATUS producers name no file, so
    they contribute nothing to fetch.
    """
    return {
        filename: fetch_workflow_source(repo, ref, filename, token)
        for filename in sorted(workflow_producers(contexts_for(ref)))
    }


def required_contexts(branch_payload: dict) -> list[str]:
    """Pull the required-status-check contexts out of a branch object.

    Raises CannotDetermine when the payload carries no protection block at all —
    that shape means "this token cannot see protection", which is a different
    answer from "protected: false" (a real, reportable finding handled by the
    caller).
    """
    if "protected" not in branch_payload:
        raise CannotDetermine("branch payload has no 'protected' key — unexpected response shape")
    if not branch_payload["protected"]:
        return []
    protection = branch_payload.get("protection")
    if not isinstance(protection, dict):
        raise CannotDetermine(
            "branch reports protected: true but carries no 'protection' block — "
            "the token cannot see it; this is NOT evidence the branch is unprotected"
        )
    checks = protection.get("required_status_checks") or {}
    contexts = checks.get("contexts")
    if contexts is None:
        raise CannotDetermine("'protection' block carries no required_status_checks.contexts")
    return list(contexts)


def _display_names(doc: dict) -> Iterator[tuple[str, str, bool]]:
    """Yield `(job_id, display_name, resolved)` for every job in `doc`.

    A check run is named after the job's `name:`, falling back to the job id when
    `name:` is absent — the same rule GitHub applies when matching a required
    context to a check run. `resolved` is False for a name carrying a `${{ }}`
    expression, which this script does not expand.
    """
    for job_id, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        raw = job.get("name") or job_id
        if not isinstance(raw, str):
            continue
        yield job_id, raw, not _ANY_EXPR.search(raw)


def job_display_names(doc: dict) -> tuple[set[str], set[str]]:
    """(resolved check-run names, unresolvable name templates) for a workflow."""
    resolved: set[str] = set()
    unresolved: set[str] = set()
    for _job_id, name, is_resolved in _display_names(doc):
        (resolved if is_resolved else unresolved).add(name)
    return resolved, unresolved


def _could_match_template(template: str, context: str) -> bool:
    """Would `context` be a plausible expansion of an unresolvable `name:`?"""
    pattern = "".join(
        ".+" if part.startswith("${{") else re.escape(part)
        for part in re.split(r"(\$\{\{.*?\}\})", template)
    )
    return re.fullmatch(pattern, context) is not None


# --- assertion 4: can any event actually produce this context? --------------

# Every documented `pull_request` activity type except `closed`. A trigger whose
# `types:` names none of these cannot fire while the PR is open, so the context
# it declares can never be satisfied before the merge — which is the only time a
# required status check is read.
_OPEN_PR_TYPES = frozenset(
    {
        "assigned",
        "unassigned",
        "labeled",
        "unlabeled",
        "opened",
        "edited",
        "reopened",
        "synchronize",
        "converted_to_draft",
        "ready_for_review",
        "locked",
        "unlocked",
        "milestoned",
        "demilestoned",
        "review_requested",
        "review_request_removed",
        "auto_merge_enabled",
        "auto_merge_disabled",
        "enqueued",
        "dequeued",
    }
)


def parse_workflow(source: str) -> dict:
    """A workflow's YAML as a mapping.

    Raises CannotDetermine when it does not parse — "this file is unreadable" is
    the same answer as an HTTP error on the read, so it is exit 2, never a
    finding and never a pass. A note would be the fail-open reading: a copy goes
    unparseable, reachability is never evaluated, and the scheduled audit reports
    green.
    """
    try:
        doc = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        # A YAML error is multi-line; `::error::` only renders its first line, so
        # flatten it or the annotation loses the reason.
        raise CannotDetermine(f"not valid YAML: {' '.join(str(exc).split())}") from exc
    if not isinstance(doc, dict):
        raise CannotDetermine(
            f"parses as {type(doc).__name__}, not a workflow mapping — triggers cannot be read"
        )
    return doc


def checkout_stands_for(branch: str) -> bool:
    """Does the working tree stand in for the copy a PR into `branch` carries?

    On a `pull_request` run actions/checkout leaves the *merge* ref checked out —
    head merged into base — and for the PR's **own** base that is exactly the
    copy GitHub reads for `pull_request`. For the *other* audited branch it is
    not that copy but a stand-in for it: this PR's changes are what a later
    `dev -> main` release PR would carry, so evaluating them against `main` now
    fails on the `dev` PR rather than on the release. That errs toward earlier
    detection deliberately — it is cheaper to fail here than to wedge the
    release path — which is why the check returns True for every audited branch
    on a `pull_request` event.

    On the scheduled run the tree is the default branch and stands for it. A
    `workflow_dispatch` from some other branch stands for neither: using that
    tree as the audited branch's proposed copy would report ordinary branch
    divergence as a defect, so the audited branch's own committed copy is used
    instead.
    """
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        return True
    return os.environ.get("GITHUB_REF_NAME") == branch


def workflow_on_block(doc: dict) -> object:
    """The `on:` mapping of a parsed workflow.

    PyYAML resolves the bare key `on` to the boolean `True` (YAML 1.1 treats
    on/off/yes/no as booleans), so a workflow's triggers land under `True`, not
    `"on"`, unless the author quoted the key. Both spellings are read — getting
    this wrong would silently report every workflow as having no triggers at all.
    """
    if "on" in doc:
        return doc["on"]
    if True in doc:
        return doc[True]
    return None


def _event_config(doc: dict, event: str) -> tuple[bool, dict]:
    """(is the event declared, its config mapping) for one workflow document.

    `on: pull_request` and `on: [push, pull_request]` declare the event with no
    filters; so does a `pull_request:` key with an empty value.
    """
    on_block = workflow_on_block(doc)
    if isinstance(on_block, str):
        return on_block == event, {}
    if isinstance(on_block, list):
        return event in on_block, {}
    if isinstance(on_block, dict):
        if event not in on_block:
            return False, {}
        config = on_block[event]
        return True, config if isinstance(config, dict) else {}
    return False, {}


def _filter_regex(pattern: str) -> re.Pattern[str]:
    """A GitHub filter pattern compiled to an anchored regex.

    Follows the documented cheat sheet: `**` matches any character including `/`,
    `*` matches any character except `/`, `?` and `+` quantify the preceding
    atom, `[]` is a character class and `\\` escapes. Each atom is emitted as its
    own group so a quantifier binds to the whole atom rather than turning a
    preceding `*` into a lazy match.
    """
    atoms: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == "\\" and i + 1 < len(pattern):
            atoms.append(re.escape(pattern[i + 1]))
            i += 2
        elif char == "*":
            double = pattern[i : i + 2] == "**"
            atoms.append(".*" if double else "[^/]*")
            i += 2 if double else 1
        elif char == "[" and "]" in pattern[i + 1 :]:
            close = pattern.index("]", i + 1)
            atoms.append(pattern[i : close + 1])
            i = close + 1
        elif char in "?+" and atoms:
            atoms[-1] = f"(?:{atoms[-1]}){char}"
            i += 1
        else:
            atoms.append(re.escape(char))
            i += 1
    return re.compile("".join(atoms) + r"\Z")


def _filter_matches(patterns: list[str], ref: str) -> bool:
    """Does `ref` survive an ordered list of filter patterns?

    Patterns are evaluated in order and a leading `!` negates the positives
    before it, which is GitHub's own rule. Starting from "no match" means a list
    of negations alone never matches — GitHub requires at least one positive
    pattern in that case anyway.
    """
    matched = False
    for raw in patterns:
        negated = raw.startswith("!")
        if _filter_regex(raw[1:] if negated else raw).match(ref):
            matched = not negated
    return matched


def _string_list(value: object) -> list[str] | None:
    """A filter list as plain strings, or None when it cannot be read as one.

    An entry carrying a `${{ }}` expression is unreadable rather than a literal
    pattern: GitHub does not evaluate expressions in `on:` at all, but comparing
    the raw text against a branch name would silently report the trigger as
    filtered out.
    """
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        return None
    if any(_ANY_EXPR.search(item) for item in items):
        return None
    return list(items)


def trigger_verdict(doc: dict, event: str, branch: str) -> tuple[bool | None, str]:
    """Can `event` in this workflow fire for a PR whose base is `branch`?

    Returns `(True, reason)` when it can, `(False, reason)` when it provably
    cannot, and `(None, reason)` when the trigger is declared in a shape this
    script cannot evaluate. `None` is never a finding — an answer that could not
    be read is not a failed answer.
    """
    declared, config = _event_config(doc, event)
    if not declared:
        return False, f"`on.{event}` is not declared"

    branches = config.get("branches")
    ignore = config.get("branches-ignore")
    if branches is not None and ignore is not None:
        return (
            False,
            (
                f"`on.{event}` sets both `branches` and `branches-ignore`, which GitHub "
                f"rejects — the workflow does not run at all"
            ),
        )
    if branches is not None:
        patterns = _string_list(branches)
        if patterns is None:
            return None, f"`on.{event}.branches` is not a list of patterns"
        if not _filter_matches(patterns, branch):
            return False, f"`on.{event}.branches` {patterns} does not match {branch!r}"
    elif ignore is not None:
        patterns = _string_list(ignore)
        if patterns is None:
            return None, f"`on.{event}.branches-ignore` is not a list of patterns"
        if _filter_matches(patterns, branch):
            return False, f"`on.{event}.branches-ignore` {patterns} excludes {branch!r}"

    if "types" in config:
        types = _string_list(config["types"])
        if types is None:
            return None, f"`on.{event}.types` is not a list of activity types"
        if not _OPEN_PR_TYPES.intersection(types):
            return (
                False,
                (
                    f"`on.{event}.types` {types} contains no activity type that fires while "
                    f"the PR is open, so the check can never report before the merge"
                ),
            )

    reason = f"`on.{event}` fires for PRs into {branch!r}"
    if "paths" in config or "paths-ignore" in config:
        reason += " (narrowed by a path filter)"
    return True, reason


def _has_path_filter(doc: dict, event: str) -> bool:
    declared, config = _event_config(doc, event)
    return declared and ("paths" in config or "paths-ignore" in config)


def reachability_findings(
    context: str,
    workflow_file: str,
    head_doc: dict | None,
    base_doc: dict | None,
    branch: str,
    notes: list[str] | None = None,
) -> list[str]:
    """Assertion 4 for one required context. See the module docstring.

    `head_doc` is the local checkout's copy of the workflow — the merge ref on a
    PR run, which is the copy GitHub reads for `pull_request`. `base_doc` is the
    copy on `branch`, which is the copy GitHub reads for `pull_request_target`.
    Either may be None when the file is absent there.
    """
    head_verdict, head_reason = (
        trigger_verdict(head_doc, "pull_request", branch)
        if head_doc is not None
        else (False, "the workflow file is absent from the PR's copy of the repo")
    )
    base_verdict, base_reason = (
        trigger_verdict(base_doc, "pull_request_target", branch)
        if base_doc is not None
        else (False, f"the workflow file is absent from {branch}")
    )

    if head_verdict or base_verdict:
        if notes is not None:
            for doc, event, where in (
                (head_doc, "pull_request", "the PR's copy"),
                (base_doc, "pull_request_target", branch),
            ):
                if doc is not None and _has_path_filter(doc, event):
                    notes.append(
                        f"required context {context!r} is produced by a trigger with a path "
                        f"filter: `on.{event}` in {where}'s .github/workflows/{workflow_file}. "
                        f"A PR into {branch} that touches none of those paths never produces "
                        f"the context, and branch protection waits for it. Reachability is "
                        f"still satisfied, so this is a note, not a finding."
                    )
        return []

    if head_verdict is None or base_verdict is None:
        if notes is not None:
            notes.append(
                f"trigger reachability for {context!r} could not be verified: "
                f".github/workflows/{workflow_file} declares a trigger in a shape this "
                f"script cannot evaluate ({head_reason}; {base_reason}). Membership and "
                f"name-match are still asserted."
            )
        return []

    return [
        (
            f"required context {context!r} is unreachable: no event can start the job that "
            f"produces it for a PR into {branch}. GitHub reads `pull_request` from the PR's "
            f"merge ref and `pull_request_target` from {branch}, and neither copy of "
            f".github/workflows/{workflow_file} declares one that fires — {head_reason}; "
            f"{base_reason}. The context will never report, so every PR into {branch} stays "
            f"blocked on it. Keep `on: pull_request` on both copies until {branch} carries "
            f"the replacement trigger, then switch (JEB-1393)."
        )
    ]


def check(
    contexts: list[str],
    protected: bool,
    workflows_dir: Path,
    notes: list[str] | None = None,
    base_workflows: dict[str, str | None] | None = None,
    branch: str = "main",
    undetermined: list[str] | None = None,
) -> list[str]:
    """Return a list of human-readable findings; empty means everything holds.

    `notes`, when passed, collects non-failing observations — the "name-match
    could not be verified" case from an unexpandable template, and assertion 4's
    unevaluable-trigger and path-filter notes. They are printed as warnings and
    never change the exit code.

    `undetermined`, when passed, collects answers that could not be read at all —
    today only a workflow file that does not parse. Those are exit 2, never a
    finding and never a pass.

    `base_workflows` maps a workflow filename to its source on `branch` (None
    when the file is absent there), as returned by `fetch_base_workflows`.
    Assertion 4 needs that copy because `pull_request_target` is read from the
    base branch, not from the PR. Passing None skips assertion 4 entirely — there
    is no offline substitute for the base copy, and guessing at it would turn a
    read the script could not perform into a finding.
    """
    findings: list[str] = []
    # Both contexts come from ci.yml; parse each distinct source once per call,
    # keyed by the text itself so the two copies of an unchanged file share one
    # entry — and one unparseable file is one exit-2 answer.
    docs: dict[str, tuple[dict | None, CannotDetermine | None]] = {}
    names_by_source: dict[str, tuple[set[str], set[str]]] = {}

    def parse_once(source: str | None) -> tuple[dict | None, CannotDetermine | None]:
        """(doc, error) for one workflow source, parsed at most once per call."""
        if source is None:
            return None, None
        if source not in docs:
            try:
                docs[source] = (parse_workflow(source), None)
            except CannotDetermine as exc:
                docs[source] = (None, exc)
        return docs[source]

    def cannot_read(workflow_file: str, error: CannotDetermine) -> None:
        if undetermined is not None:
            undetermined.append(
                f"cannot read .github/workflows/{workflow_file} as it applies to {branch} — {error}"
            )

    asserted = contexts_for(branch)

    if not protected:
        findings.append(
            f"{branch} is not branch-protected at all — every required check below is unenforced"
        )

    present = set(contexts)
    for context, producer in asserted.items():
        if context not in present:
            findings.append(
                f"required status check missing from {branch}'s protection: {context!r} "
                f"({describe_producer(producer)}). Without it GitHub "
                f"allows the merge even when the check is red. Restore it with "
                f"PATCH /repos/<repo>/branches/{branch}/protection/required_status_checks."
            )

        # Assertions 2 and 4 both ask questions about a workflow file. An
        # EXTERNAL_STATUS context has none, so they are skipped rather than
        # answered against an invented filename — membership above is the whole
        # assertion for these, and assertion 3 below still covers the reverse
        # direction. Skipping here is what keeps the escape hatch from weakening
        # anything for the contexts that really are workflow jobs.
        if not isinstance(producer, str):
            continue
        workflow_file = producer

        path = workflows_dir / workflow_file
        if not path.exists():
            findings.append(
                f"workflow .github/workflows/{workflow_file} is gone, but {context!r} is "
                f"still expected to report — that context can never go green again"
            )
            continue

        # A file that does not parse is not a file with no jobs and no triggers —
        # exit 2, the same answer as a read that failed, never a note. A note
        # would be the fail-open reading: the copy goes unparseable, the
        # assertions below never run, and the audit reports green.
        local_source = path.read_text()
        local_doc, local_error = parse_once(local_source)
        if local_error is not None:
            cannot_read(workflow_file, local_error)
            continue

        # Assertion 4 (JEB-1393): declared is not the same as reachable.
        if base_workflows is not None:
            on_branch = base_workflows.get(workflow_file)
            # The checkout is the copy a PR into `branch` would carry only on the
            # triggers that stand for it; otherwise read the branch's own.
            if checkout_stands_for(branch):
                head_doc, head_error = local_doc, None
            else:
                head_doc, head_error = parse_once(on_branch)
            base_doc, base_error = parse_once(on_branch)
            error = head_error or base_error
            if error is not None:
                cannot_read(workflow_file, error)
            else:
                findings.extend(
                    reachability_findings(context, workflow_file, head_doc, base_doc, branch, notes)
                )

        if local_source not in names_by_source:
            names_by_source[local_source] = job_display_names(local_doc)
        names, unresolvable = names_by_source[local_source]
        if context not in names:
            plausible = sorted(t for t in unresolvable if _could_match_template(t, context))
            if plausible:
                # A template this script does not expand could still be producing
                # the context. Not evidence of a rename — say so and move on.
                if notes is not None:
                    notes.append(
                        f"name-match for {context!r} could not be verified: "
                        f".github/workflows/{workflow_file} names a job with an unresolvable "
                        f"template ({plausible[0]!r}), so the context may or may not be one of "
                        f"its expansions. Membership in {branch}'s required list is still asserted."
                    )
                continue
            findings.append(
                f"required context {context!r} matches no job name in "
                f".github/workflows/{workflow_file} (jobs there are named: "
                f"{sorted(n for n in names if n)}). A required context is matched to a "
                f"check run by name, so this context will never report and every PR into "
                f"{branch} stays blocked on it."
            )

    # Assertion 3 (JEB-1227): the reverse direction. `contexts` is the live list,
    # so this only ever runs on a read that succeeded — the exit-2 path never
    # reaches here, and an unreadable list is not an empty one.
    for context in sorted(set(contexts) - set(asserted)):
        findings.append(
            f"required status check on {branch} is not asserted here: {context!r} is in "
            f"{branch}'s required list but missing from REQUIRED_CONTEXTS[{branch!r}] in "
            f"scripts/assert_required_checks.py, so this audit does not monitor it and would "
            f"not notice it being dropped. Add {context!r} to REQUIRED_CONTEXTS[{branch!r}] — "
            f"mapped to the .github/workflows/ file whose job name produces it, or to "
            f"EXTERNAL_STATUS with a comment naming the publisher if nothing in "
            f".github/workflows/ produces it — in the same PR as the protection change."
        )

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assert a branch's required status checks.")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "adetbekov/pixel"),
        help="owner/name (default: $GITHUB_REPOSITORY)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help=f"protected branch to audit (audited: {', '.join(sorted(REQUIRED_CONTEXTS))})",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")

    try:
        asserted = contexts_for(args.branch)
        payload = fetch_branch(args.repo, args.branch, token)
        contexts = required_contexts(payload)
        # Assertion 4 needs the branch's own copy of each declaring workflow —
        # that is the copy GitHub reads for `pull_request_target`, and on a PR run
        # it is emphatically not the checkout. Same exit-2 rule as above: a fetch
        # that fails is "cannot determine", not "no trigger".
        base_workflows = fetch_base_workflows(args.repo, args.branch, token)
    except CannotDetermine as exc:
        print(f"::error::CANNOT DETERMINE — {exc}")
        print(
            "This is not a pass and not a failure of the gate itself: the audit could not "
            f"evaluate {args.branch}'s protection. If the read is what failed, re-run with a "
            f"token that can read repos/{args.repo}/branches/{args.branch}."
        )
        return EXIT_CANNOT_DETERMINE

    notes: list[str] = []
    undetermined: list[str] = []
    findings = check(
        contexts,
        bool(payload.get("protected")),
        _WORKFLOWS,
        notes,
        base_workflows=base_workflows,
        branch=args.branch,
        undetermined=undetermined,
    )

    print(f"{args.repo}@{args.branch} required status checks ({len(contexts)}):")
    for context in contexts:
        print(f"  - {context}")

    for note in notes:
        print(f"::warning::{note}")

    if undetermined:
        # One unreadable workflow file is one answer that could not be read, not
        # one per required context that happens to point at it. Reported before
        # findings and returned ahead of them: a run that could not evaluate part
        # of the gate has not passed it, whatever else it saw.
        for message in dict.fromkeys(undetermined):
            print(f"::error::CANNOT DETERMINE — {message}")
        for finding in findings:
            print(f"::error::{finding}")
        return EXIT_CANNOT_DETERMINE

    if findings:
        for finding in findings:
            print(f"::error::{finding}")
        return EXIT_FINDING

    print(
        f"OK: all {len(asserted)} asserted contexts are required on {args.branch}, their "
        f"context strings match their job names, every one of them is reachable by a "
        f"trigger that fires for PRs into {args.branch}, and {args.branch} requires "
        f"nothing this script does not assert."
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
