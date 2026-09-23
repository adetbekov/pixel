"""Assert docker-compose.yml starts the artifact the `image build` job just built.

    python .github/scripts/verify_compose.py pixel:ci

`docker compose config` alone only proves the file parses. It happily validates a
stack that pulls `ghcr.io/somebody/pixel:latest`, or builds from `./backend` — and
with `pull_policy: never` on the NAS that is a redeploy that either fails outright or
silently keeps running the old container, after a fully green PR.

So the reference is compared against the repo, never against a literal retyped here:
the documented redeploy command in `README.md` is the one place the image tag and the
build context are written down, and the NAS runs exactly those two lines.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ROOT / "docker-compose.yml"

# The `docker build -t <ref> <context>` line in README.md's Deploy block — what the
# NAS actually runs, and therefore the only image a `pull_policy: never` stack can find.
BUILD_RE = re.compile(r"^docker build -t (?P<ref>\S+) (?P<context>\S+)(?:\s+#.*)?$", re.MULTILINE)


def repository(ref: str) -> str:
    """`pixel:ci` -> `pixel`. A `:` inside the last path segment is a tag, not a port."""
    head, sep, tail = ref.rpartition(":")
    return head if sep and "/" not in tail else ref


def documented_build(readme: str) -> tuple[str, str]:
    match = BUILD_RE.search(readme)
    if match is None:
        sys.exit("README.md documents no `docker build -t <ref> <context>` redeploy command")
    return match.group("ref"), match.group("context")


def compose_config() -> dict:
    proc = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config", "--format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"`docker compose config` rejected {COMPOSE.name}:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def check(config: dict, built: str, ref: str, context: str) -> list[str]:
    """Every service must resolve to the image the documented build produces."""
    failures = []

    # The job's own tag differs (`:ci`, never pushed), but it has to be the same image.
    if repository(built) != repository(ref):
        failures.append(
            f"the job builds {built} while README.md documents {ref} — "
            "the gate and the NAS are building different images"
        )
    if (ROOT / context).resolve() != ROOT:
        failures.append(
            f"README.md builds from {context}, the job builds from the repo root"
        )

    services = config.get("services") or {}
    if not services:
        failures.append(f"{COMPOSE.name} declares no services")

    for name, service in services.items():
        build = service.get("build")
        if build is not None:
            # A build section makes the service self-sufficient — compose builds the
            # image itself — so the tag is free, but the sources must still be ours.
            if (ROOT / build.get("context", "")).resolve() != ROOT:
                failures.append(
                    f"service `{name}` builds from {build.get('context')!r}, "
                    "not the repo root the job builds"
                )
            dockerfile = build.get("dockerfile")
            if dockerfile is not None and Path(dockerfile).name != "Dockerfile":
                failures.append(
                    f"service `{name}` builds {dockerfile}, not the Dockerfile the job builds"
                )
            continue

        image = service.get("image")
        if image != ref:
            failures.append(
                f"service `{name}` runs image {image!r} and builds nothing, but the "
                f"documented redeploy builds {ref!r} — nothing creates that image on the NAS"
            )

    return failures


def main() -> int:
    built = sys.argv[1] if len(sys.argv) > 1 else "pixel:ci"
    ref, context = documented_build((ROOT / "README.md").read_text())
    failures = check(compose_config(), built, ref, context)

    for line in failures:
        print(f"FAIL: {line}")
    if failures:
        return 1

    print(f"ok: {COMPOSE.name} runs {ref}, the image this job builds as {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
