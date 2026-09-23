"""Assert a built Pixel image carries the dependencies the repo says it should.

    python .github/scripts/verify_image_deps.py pixel:ci

Both expectations are read out of the repo, never retyped here: the torch pin from
`requirements-torch.txt` (the file the image installs from) and the laya range from
`pyproject.toml`. A literal in this file would only be the next copy to drift — which
is the failure this gate exists to catch.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[2]

# Runs inside the image. `torch.version.cuda` is None only on a CPU wheel — the
# CUDA build is ~1 GB and would quietly bloat every NAS pull of the image.
PROBE = """
import importlib.metadata as md, json, torch
print(json.dumps({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "laya": md.version("laya"),
}))
"""


def torch_pin() -> str:
    for raw in (ROOT / "requirements-torch.txt").read_text().splitlines():
        line = raw.strip()
        if line.startswith("torch=="):
            return line.split("==", 1)[1]
    sys.exit("requirements-torch.txt carries no `torch==` pin")


def laya_specifier() -> SpecifierSet:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for dep in data["project"]["dependencies"]:
        req = Requirement(dep)
        if req.name == "laya":
            return req.specifier
    sys.exit("pyproject.toml lists no `laya` dependency")


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
    pin = torch_pin()
    laya_range = laya_specifier()
    got = probe(image)

    failures = []
    if got["torch"] != pin:
        failures.append(f"torch is {got['torch']}, requirements-torch.txt pins {pin}")
    if got["cuda"] is not None:
        failures.append(f"CUDA build leaked into the image: torch.version.cuda = {got['cuda']}")
    if Version(got["laya"]) not in laya_range:
        failures.append(f"laya is {got['laya']}, pyproject.toml asks for {laya_range}")

    for line in failures:
        print(f"FAIL: {line}")
    if failures:
        return 1

    print(f"ok: {image} has torch {got['torch']} (cpu), laya {got['laya']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
