"""Assert a built Pixel image carries the dependencies the repo says it should.

    python .github/scripts/verify_image_deps.py pixel:ci

Every expectation is read out of `pyproject.toml`'s `engine` extra — the one place
the torch pin and the laya range live, and the declaration the Dockerfile installs
with `uv sync --extra engine`. A literal in this file would only be the next copy to
drift — which is the failure this gate exists to catch (JEB-1516).

The image is also the only environment in CI that has laya installed, so the
`embed_fn_from_agent` contract `LayaEngine.embed` relies on is probed here too
(`tests/test_cluster.py` skips it wherever laya is absent).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]

# Runs inside the image. `torch.version.cuda` is None only on a CPU wheel — the
# CUDA build is ~1 GB and would quietly bloat every NAS pull of the image.
#
# `embed_fn_from_agent`: one positional `agent`, the rest defaulted — `LayaEngine.embed`
# passes nothing else, so a newly required argument has to fail here, not live.
PROBE = """
import importlib.metadata as md, inspect, json, laya, torch
helper = getattr(laya, "embed_fn_from_agent", None)
params = list(inspect.signature(helper).parameters.values()) if helper else []
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "laya": md.version("laya"),
    "embed_fn": helper is not None
        and bool(params) and params[0].name == "agent"
        and all(p.default is not inspect.Parameter.empty for p in params[1:]),
}))
"""


def engine_requirements() -> dict[str, Requirement]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extra = data["project"]["optional-dependencies"].get("engine", [])
    found = {req.name: req for req in map(Requirement, extra)}
    for name in ("torch", "laya"):
        if name not in found:
            sys.exit(f"pyproject.toml's `engine` extra lists no `{name}` dependency")
    if not any(spec.operator == "==" for spec in found["torch"].specifier):
        sys.exit(f"torch is not pinned (`==`) in pyproject.toml's `engine` extra: {found['torch']}")
    return found


def probe(image: str) -> dict:
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python", image, "-c", PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"probing {image} failed:\n{proc.stderr.strip()}")
    return json.loads(proc.stdout)


def main() -> int:
    image = sys.argv[1] if len(sys.argv) > 1 else "pixel:ci"
    wanted = engine_requirements()
    got = probe(image)

    failures = []
    # `==2.14.0` is satisfied by the CPU index's `2.14.0+cpu` — a local version label
    # never fails a pin, which is why the CUDA check below is separate.
    if not wanted["torch"].specifier.contains(got["torch"], prereleases=True):
        failures.append(f"torch is {got['torch']}, pyproject.toml pins {wanted['torch'].specifier}")
    if got["cuda"] is not None:
        failures.append(f"CUDA build leaked into the image: torch.version.cuda = {got['cuda']}")
    if Version(got["laya"]) not in wanted["laya"].specifier:
        failures.append(
            f"laya is {got['laya']}, pyproject.toml asks for {wanted['laya'].specifier}"
        )
    if not got["embed_fn"]:
        failures.append(
            "laya.embed_fn_from_agent is gone or grew a required argument — see LayaEngine.embed"
        )

    for line in failures:
        print(f"FAIL: {line}")
    if failures:
        return 1

    print(f"ok: {image} has torch {got['torch']} (cpu), laya {got['laya']}, embed_fn_from_agent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
