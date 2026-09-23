"""The torch pin lives in exactly one file.

JEB-1516: the Dockerfile and ci.yml each carried their own `torch==` line, they
drifted, and CI stayed green because nothing compared them. The image gate in
`.github/workflows/ci.yml` reads the pin from `requirements-torch.txt`; these tests
keep a second copy from reappearing somewhere it would drift again.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = ROOT / "requirements-torch.txt"
PIN_RE = re.compile(r"^torch==(?P<version>\S+)$", re.MULTILINE)


def test_the_pin_file_carries_a_cpu_pin():
    match = PIN_RE.search(PIN_FILE.read_text())
    assert match is not None, "requirements-torch.txt must pin `torch==<version>`"
    # `+cpu` is part of the pin so a CUDA wheel can never satisfy it silently.
    assert match.group("version").endswith("+cpu")
    assert "--index-url https://download.pytorch.org/whl/cpu" in PIN_FILE.read_text()


def test_nothing_else_pins_torch():
    # Everything that installs or documents torch. `verify_image_deps.py` is not in
    # the list on purpose: it matches on the `torch==` prefix to *read* the pin.
    others = [
        ROOT / "Dockerfile",
        ROOT / ".github" / "workflows" / "ci.yml",
        ROOT / "pyproject.toml",
        ROOT / "README.md",
    ]
    duplicates = [p.name for p in others if p.exists() and "torch==" in p.read_text()]
    assert not duplicates, f"torch pinned outside requirements-torch.txt: {duplicates}"
