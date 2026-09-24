"""`docker-compose.yml` must not hold a second copy of a code default.

JEB-1602: the stack ships `${ROUTER_THRESHOLD:-0.6}` while `backend/brain/router.py`
was calibrated to 0.66. `ROUTER_THRESHOLD` is not set in the stack's Env, so the
compose copy wins and production keeps running the pre-calibration number — a
released commit that never reaches the container. Same for `MINER_SIM` (0.75 vs
0.88). The interpolation form `${VAR:-default}` stays: it is how the stack Env can
still override. What must not drift is the default inside it.

These tests read the committed compose file and compare every `${VAR:-default}`
against the `DEFAULT_*` constant the code falls back to when `VAR` is unset.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from backend import db
from backend.brain import engine, router
from backend.miner import backtest, cluster, generate, run
from backend.teacher import client as teacher_client

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.yml"
SERVICE = "pixel"

#: `${NAME:-default}` — the only interpolation form used in the file, and the only
#: one that can carry a default at all (`${NAME}` and `${NAME:?err}` cannot).
INTERPOLATION = re.compile(r"^\$\{(?P<name>[A-Z0-9_]+):-(?P<default>.*)\}$")

#: Env var -> the constant the code uses when the var is unset. One entry per
#: variable that has a code-side default; both sides must read the same number.
CODE_DEFAULTS = {
    "ROUTER_THRESHOLD": router.DEFAULT_THRESHOLD,
    "MINER_SIM": cluster.DEFAULT_SIM,
    "MINER_MIN_CLUSTER": cluster.DEFAULT_MIN_CLUSTER,
    "MINER_BATCH": run.DEFAULT_BATCH,
    "MINER_MIN_MATCH": backtest.DEFAULT_MIN_MATCH,
    "LAYA_MODEL": engine.DEFAULT_MODEL,
    "LAYA_DEVICE": engine.DEFAULT_DEVICE,
    "GEMINI_TEACHER_MODEL": teacher_client.DEFAULT_MODEL,
    "GEMINI_MINER_MODEL": generate.DEFAULT_MODEL,
}

#: Variables with no code default to compare against. Registering one is a
#: decision, not a fallthrough: an unknown variable fails the test below, so a new
#: knob lands either in CODE_DEFAULTS or here, with the reason.
NO_CODE_DEFAULT = {
    # Secret. `backend/miner/generate.py` and `backend/teacher/client.py` only ask
    # whether it is set; there is nothing to default to.
    "GEMINI_API_KEY": "secret, no default value exists",
    # Deployment path, not a tuning knob: `db.DEFAULT_DB_PATH` is `./pixel.db` for a
    # developer run, `/data/pixel.db` is where the container's volume is mounted.
    "PIXEL_DB_PATH": f"container volume path; code default {db.DEFAULT_DB_PATH!r} is for local runs",
    # Read by huggingface_hub, not by our code — the checkpoint cache on the volume.
    "HF_HOME": "consumed by huggingface_hub, no constant in this repo",
    # Read by the base image's libc/tzdata, not by our code.
    "TZ": "container timezone, no constant in this repo",
    # Diagnostic switch read inline in `backend/main.py` against the literal "1";
    # there is no constant and off is the absence of the flag.
    "PIXEL_SKIP_MODEL": 'diagnostic switch, compared inline against "1"',
}


def compose_environment() -> dict[str, str]:
    service = yaml.safe_load(COMPOSE.read_text())["services"][SERVICE]
    return {str(k): str(v) for k, v in service["environment"].items()}


def mismatches(environment: dict[str, str]) -> list[str]:
    """Every compose default that disagrees with its code default, named."""
    failures = []
    for name, value in environment.items():
        if name not in CODE_DEFAULTS:
            continue
        expected = CODE_DEFAULTS[name]
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
            failures.append(f"{name}: compose default {written!r} != code default {expected!r}")
    return failures


def test_the_committed_compose_file_matches_the_code():
    assert mismatches(compose_environment()) == []


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
    unknown = [
        name
        for name in compose_environment()
        if name not in CODE_DEFAULTS and name not in NO_CODE_DEFAULT
    ]
    assert not unknown, (
        f"compose sets {unknown} — add each to CODE_DEFAULTS, or to NO_CODE_DEFAULT with the reason"
    )


def test_every_registered_variable_is_actually_in_compose():
    # The other direction: a variable dropped from compose must drop its entry too,
    # otherwise the registry grows rows that assert nothing.
    environment = compose_environment()
    stale = [name for name in {**CODE_DEFAULTS, **NO_CODE_DEFAULT} if name not in environment]
    assert not stale, f"registered but absent from docker-compose.yml: {stale}"


@pytest.mark.parametrize(
    ("name", "drifted"),
    [
        ("ROUTER_THRESHOLD", "0.6"),  # the JEB-1602 value, verbatim
        ("MINER_SIM", "0.75"),
        ("MINER_BATCH", "9"),
        ("LAYA_DEVICE", "cuda"),
    ],
)
def test_a_drifted_default_fails(name, drifted):
    environment = compose_environment() | {name: f"${{{name}:-{drifted}}}"}
    failures = mismatches(environment)
    assert any(name in failure for failure in failures)


def test_a_hardcoded_value_fails():
    # Dropping the interpolation would take the stack Env's override path away, so
    # it fails even when the number itself happens to be right.
    environment = compose_environment() | {"ROUTER_THRESHOLD": str(router.DEFAULT_THRESHOLD)}
    assert any("override path" in failure for failure in mismatches(environment))
