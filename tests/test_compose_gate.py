"""The compose gate's rules, checked without a Docker daemon.

`.github/scripts/verify_compose.py` only runs inside the `image build` job, so the
one thing that can rot unnoticed is the comparison itself — a gate that never fails
looks exactly like a gate that passes. These tests feed it parsed `docker compose
config` output directly and assert both directions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "verify_compose.py"

spec = importlib.util.spec_from_file_location("verify_compose", SCRIPT)
verify_compose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_compose)

REF = "pixel:latest"
BUILT = "pixel:ci"
CONTEXT = str(ROOT)


def config(**service):
    return {"services": {"pixel": service}}


def test_readme_documents_the_redeploy_build():
    ref, context = verify_compose.documented_build((ROOT / "README.md").read_text())
    assert verify_compose.repository(ref) == "pixel"
    assert (ROOT / context).resolve() == ROOT


def test_the_real_compose_file_passes():
    # `docker compose config` resolves `image:` verbatim, so the committed file is
    # its own fixture here — this is the assertion the CI step makes for real.
    ref, context = verify_compose.documented_build((ROOT / "README.md").read_text())
    assert verify_compose.check(config(image=ref), BUILT, ref, context) == []


def test_an_image_nobody_builds_fails():
    failures = verify_compose.check(
        config(image="ghcr.io/adetbekov/pixel:latest"), BUILT, REF, CONTEXT
    )
    assert any("ghcr.io/adetbekov/pixel:latest" in f for f in failures)


def test_a_drifted_tag_fails():
    failures = verify_compose.check(config(image="pixel:prod"), BUILT, REF, CONTEXT)
    assert any("pixel:prod" in f for f in failures)


def test_a_build_context_outside_the_repo_root_fails():
    failures = verify_compose.check(
        config(build={"context": str(ROOT / "backend")}), BUILT, REF, CONTEXT
    )
    assert any("builds from" in f for f in failures)


def test_a_second_dockerfile_fails():
    failures = verify_compose.check(
        config(build={"context": CONTEXT, "dockerfile": "Dockerfile.prod"}), BUILT, REF, CONTEXT
    )
    assert any("Dockerfile.prod" in f for f in failures)


def test_a_build_section_on_the_repo_root_passes():
    # Building in compose is a legitimate alternative to `image:` + `pull_policy: never`;
    # the gate is about *which sources*, not about which of the two styles is used.
    assert verify_compose.check(config(build={"context": CONTEXT}), BUILT, REF, CONTEXT) == []


def test_a_job_tag_for_another_image_fails():
    failures = verify_compose.check(config(image=REF), "other:ci", REF, CONTEXT)
    assert any("different images" in f for f in failures)


def test_no_services_fails():
    assert verify_compose.check({"services": {}}, BUILT, REF, CONTEXT)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("pixel:latest", "pixel"),
        ("pixel", "pixel"),
        ("ghcr.io/adetbekov/pixel:1.2", "ghcr.io/adetbekov/pixel"),
        ("registry:5000/pixel", "registry:5000/pixel"),
    ],
)
def test_repository_strips_only_the_tag(ref, expected):
    assert verify_compose.repository(ref) == expected
