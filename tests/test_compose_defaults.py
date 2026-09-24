"""No document may hold a second copy of a code default that can drift silently.

JEB-1602: the stack ships `${ROUTER_THRESHOLD:-0.6}` while `backend/brain/router.py`
was calibrated to 0.66. `ROUTER_THRESHOLD` is not set in the stack's Env, so the
compose copy wins and production keeps running the pre-calibration number — a
released commit that never reaches the container. Same for `MINER_SIM` (0.75 vs
0.88). The interpolation form `${VAR:-default}` stays: it is how the stack Env can
still override. What must not drift is the default inside it.

JEB-1609: gating `docker-compose.yml` alone left three more copies of every default
running on author discipline — `.env.example`, the README env table, and the
prose. JEB-1606 moved `GEMINI_MINER_MODEL` and had to touch all four by hand; the
next such change has no reason to get it right. So `.env.example` and the README
table are now read and compared the same way, against the same registry.

**What is gated and what is not.** Three documents are parsed:

* `docker-compose.yml` — every `${VAR:-default}` in the `pixel` service.
* `.env.example` — every `KEY=value`. No interpolation form appears there, so the
  comparison is direct.
* `README.md` — the table under `## Environment` only, by variable name. Its rows
  are `| \\`KEY\\` | \\`value\\` | prose |`, which parses without a markdown library.

README **prose** is deliberately *not* gated. A model name in running text is not
always a claim about the current default: `README.md:393` names
`models/gemini-2.5-flash-lite` while describing a defect that happened on that
model, and it must keep naming it after the miner moves elsewhere. A gate cannot
tell that apart from `README.md:189`, which *is* a default claim, without
understanding the sentence. Rather than encode a brittle guess, the rule is: put a
default in the table, where it is gated, and treat any prose copy as an annotation
a reviewer checks. Same for the docstrings in `scripts/`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from backend import db, feedback
from backend.brain import engine, router
from backend.miner import attempts, backtest, case, cluster, generate, run
from backend.teacher import client as teacher_client

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"
README = ROOT / "README.md"
SERVICE = "pixel"

#: `${NAME:-default}` — the only interpolation form used in the file, and the only
#: one that can carry a default at all (`${NAME}` and `${NAME:?err}` cannot).
INTERPOLATION = re.compile(r"^\$\{(?P<name>[A-Z0-9_]+):-(?P<default>.*)\}$")

#: `KEY=value` in `.env.example`; comments and blank lines are skipped.
ENV_LINE = re.compile(r"^(?P<name>[A-Z0-9_]+)=(?P<value>.*)$")

#: One row of the README env table: `| `KEY` | `value` | prose |`.
README_ROW = re.compile(r"^\|\s*`(?P<name>[A-Z0-9_]+)`\s*\|\s*(?P<value>[^|]*?)\s*\|")

#: Env var -> the constant the code uses when the var is unset. One entry per
#: variable that has a code-side default; every document must read the same value.
CODE_DEFAULTS = {
    "ROUTER_THRESHOLD": router.DEFAULT_THRESHOLD,
    "MINER_SIM": cluster.DEFAULT_SIM,
    "MINER_MIN_CLUSTER": cluster.DEFAULT_MIN_CLUSTER,
    "MINER_BATCH": run.DEFAULT_BATCH,
    "MINER_POOL_WINDOW": case.DEFAULT_POOL_WINDOW,
    "MINER_MAX_ATTEMPTS": attempts.DEFAULT_MAX_ATTEMPTS,
    "MINER_MIN_MATCH": backtest.DEFAULT_MIN_MATCH,
    "SKILL_DISLIKE_LIMIT": feedback.DEFAULT_DISLIKE_LIMIT,
    "SKILL_MIN_RATED": feedback.DEFAULT_MIN_RATED,
    "LAYA_MODEL": engine.DEFAULT_MODEL,
    "LAYA_DEVICE": engine.DEFAULT_DEVICE,
    "GEMINI_TEACHER_MODEL": teacher_client.DEFAULT_MODEL,
    "GEMINI_MINER_MODEL": generate.DEFAULT_MODEL,
    "PIXEL_DB_PATH": db.DEFAULT_DB_PATH,
}

#: Variables with no code default to compare against. Registering one is a
#: decision, not a fallthrough: an unknown variable fails the tests below, so a new
#: knob lands either in CODE_DEFAULTS or here, with the reason.
NO_CODE_DEFAULT = {
    # Secret. `backend/miner/generate.py` and `backend/teacher/client.py` only ask
    # whether it is set; there is nothing to default to.
    "GEMINI_API_KEY": "secret, no default value exists",
    # Read by huggingface_hub, not by our code — the checkpoint cache on the volume.
    "HF_HOME": "consumed by huggingface_hub, no constant in this repo",
    # Read by the base image's libc/tzdata, not by our code.
    "TZ": "container timezone, no constant in this repo",
    # Diagnostic switch read inline in `backend/main.py` against the literal "1";
    # there is no constant and off is the absence of the flag.
    "PIXEL_SKIP_MODEL": 'diagnostic switch, compared inline against "1"',
}

#: (document, variable) -> why that document is *supposed* to say something else.
#: An exemption is per document, not global: `PIXEL_DB_PATH` has a real code
#: default and `.env.example` and the README are held to it; only the container
#: overrides it, because that is where the volume is mounted.
DELIBERATE_OVERRIDES = {
    ("docker-compose.yml", "PIXEL_DB_PATH"): (
        f"container volume path; code default {db.DEFAULT_DB_PATH!r} is for local runs"
    ),
}


def compose_environment() -> dict[str, str]:
    service = yaml.safe_load(COMPOSE.read_text())["services"][SERVICE]
    return {str(k): str(v) for k, v in service["environment"].items()}


def env_example_environment() -> dict[str, str]:
    declared = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        match = ENV_LINE.match(line.strip())
        if match is not None:
            declared[match.group("name")] = match.group("value").strip()
    return declared


def readme_environment() -> dict[str, str]:
    """The `## Environment` table, by variable name.

    Bounded to that one section: other tables in the README have a backticked
    first cell too, and a repo-wide row scan would pick them up.
    """
    lines = README.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "## Environment")
    end = next(
        (i for i, line in enumerate(lines[start + 1 :], start + 1) if line.startswith("## ")),
        len(lines),
    )
    declared = {}
    for line in lines[start:end]:
        match = README_ROW.match(line)
        if match is not None:
            declared[match.group("name")] = match.group("value").strip().strip("`")
    return declared


def mismatches(environment: dict[str, str], document: str = "docker-compose.yml") -> list[str]:
    """Every default in `document` that disagrees with its code default, named.

    `docker-compose.yml` writes `${VAR:-default}` and the default is read out of the
    interpolation — dropping it would take the stack Env's override path away, so
    that is a failure of its own. `.env.example` and the README write the value
    directly.
    """
    interpolated = document == "docker-compose.yml"
    failures = []
    for name, value in environment.items():
        if name not in CODE_DEFAULTS or (document, name) in DELIBERATE_OVERRIDES:
            continue
        expected = CODE_DEFAULTS[name]
        written = value
        if interpolated:
            match = INTERPOLATION.match(value)
            if match is None:
                failures.append(
                    f"{name}: compose pins {value!r} with no `${{{name}:-...}}` override path; "
                    f"code default is {expected!r}"
                )
                continue
            written = match.group("default")
        try:
            same = type(expected)(written) == expected
        except ValueError:
            same = False
        if not same:
            failures.append(
                f"{name}: {document} default {written!r} != code default {expected!r}"
            )
    return failures


def unaccounted(environment: dict[str, str]) -> list[str]:
    return [
        name
        for name in environment
        if name not in CODE_DEFAULTS and name not in NO_CODE_DEFAULT
    ]


def test_the_committed_compose_file_matches_the_code():
    assert mismatches(compose_environment()) == []


def test_env_example_matches_the_code():
    # JEB-1609. `.env.example` is what a developer copies to `.env` and what every
    # README reader treats as the list of defaults; a stale line here is the same
    # defect as a stale compose line, one layer further from production.
    assert mismatches(env_example_environment(), ".env.example") == []


def test_the_readme_env_table_matches_the_code():
    assert mismatches(readme_environment(), "README.md") == []


def test_the_miner_and_the_teacher_default_to_different_models():
    # JEB-1606. The free-tier quota bucket is counted per (project, model), and this
    # project's key is shared with warmplace. Collapse the two defaults onto one
    # model and both halves spend one 20-requests-a-day bucket: the miner starves
    # silently and the teacher's misses come back as an opaque FALLBACK (JEB-1600).
    # A stack redeploy without the Env override must not be able to cause that.
    assert generate.DEFAULT_MODEL != teacher_client.DEFAULT_MODEL


def test_every_compose_variable_is_accounted_for():
    # A knob added to compose and to nothing else is exactly how 0.6 survived: the
    # comparison has to notice the variable before it can compare it.
    unknown = unaccounted(compose_environment())
    assert not unknown, (
        f"compose sets {unknown} — add each to CODE_DEFAULTS, or to NO_CODE_DEFAULT with the reason"
    )


def test_every_documented_variable_is_accounted_for():
    unknown = unaccounted(env_example_environment() | readme_environment())
    assert not unknown, (
        f".env.example / README document {unknown} — add each to CODE_DEFAULTS, "
        f"or to NO_CODE_DEFAULT with the reason"
    )


def test_env_example_and_the_readme_table_list_the_same_variables():
    # The two are read as one list by anyone setting the stack up. A knob added to
    # one and not the other is undocumented in practice — and invisible to the
    # comparison above, which only sees what a document declares.
    in_env = set(env_example_environment())
    in_readme = set(readme_environment())
    assert in_env == in_readme, (
        f"only in .env.example: {sorted(in_env - in_readme)}; "
        f"only in the README table: {sorted(in_readme - in_env)}"
    )


def test_every_registered_variable_is_actually_documented_somewhere():
    # The other direction: a variable dropped from every document must drop its
    # entry too, otherwise the registry grows rows that assert nothing. Not every
    # knob is in compose — `MINER_POOL_WINDOW` and the `SKILL_*` pair are code
    # defaults the stack never sets — so the check is against the union.
    documented = (
        set(compose_environment()) | set(env_example_environment()) | set(readme_environment())
    )
    stale = [name for name in {**CODE_DEFAULTS, **NO_CODE_DEFAULT} if name not in documented]
    assert not stale, f"registered but in no document: {stale}"


def test_every_deliberate_override_still_has_something_to_override():
    # An exemption outlives the line it excuses if nothing checks. Both halves must
    # still exist: the variable in the registry, and the document declaring it.
    documents = {
        "docker-compose.yml": compose_environment(),
        ".env.example": env_example_environment(),
        "README.md": readme_environment(),
    }
    stale = [
        (document, name)
        for document, name in DELIBERATE_OVERRIDES
        if name not in CODE_DEFAULTS or name not in documents[document]
    ]
    assert not stale, f"exemptions that no longer apply: {stale}"


@pytest.mark.parametrize(
    ("name", "drifted"),
    [
        ("ROUTER_THRESHOLD", "0.6"),  # the JEB-1602 value, verbatim
        ("MINER_SIM", "0.75"),
        ("MINER_BATCH", "9"),
        ("LAYA_DEVICE", "cuda"),
    ],
)
def test_a_drifted_compose_default_fails(name, drifted):
    environment = compose_environment() | {name: f"${{{name}:-{drifted}}}"}
    failures = mismatches(environment)
    assert any(name in failure for failure in failures)


@pytest.mark.parametrize(
    ("name", "drifted"),
    [
        # The JEB-1606 case verbatim: the code moved, `.env.example` did not.
        ("GEMINI_MINER_MODEL", "models/gemini-2.5-flash-lite"),
        ("ROUTER_THRESHOLD", "0.6"),
        ("PIXEL_DB_PATH", "/data/pixel.db"),
    ],
)
def test_a_drifted_env_example_default_fails(name, drifted):
    environment = env_example_environment() | {name: drifted}
    failures = mismatches(environment, ".env.example")
    assert any(name in failure for failure in failures)


@pytest.mark.parametrize(
    ("name", "drifted"),
    [
        ("GEMINI_MINER_MODEL", "models/gemini-2.5-flash-lite"),
        ("MINER_MAX_ATTEMPTS", "5"),
    ],
)
def test_a_drifted_readme_default_fails(name, drifted):
    environment = readme_environment() | {name: drifted}
    failures = mismatches(environment, "README.md")
    assert any(name in failure for failure in failures)


def test_a_hardcoded_compose_value_fails():
    # Dropping the interpolation would take the stack Env's override path away, so
    # it fails even when the number itself happens to be right.
    environment = compose_environment() | {"ROUTER_THRESHOLD": str(router.DEFAULT_THRESHOLD)}
    assert any("override path" in failure for failure in mismatches(environment))


def test_the_parsers_see_the_real_documents():
    # Every comparison above is vacuously green on an empty parse. Anchor all three
    # on a variable that is in each by construction.
    assert "ROUTER_THRESHOLD" in compose_environment()
    assert "ROUTER_THRESHOLD" in env_example_environment()
    assert "ROUTER_THRESHOLD" in readme_environment()
