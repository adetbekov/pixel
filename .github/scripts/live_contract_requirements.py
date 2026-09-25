#!/usr/bin/env python3
"""The dependency list `check_live_gemini_contract.py` needs, read off pyproject.

The nightly gate used to install a list typed from memory — the Gemini SDK and
pydantic — while the probe's import graph reached four modules deeper:
``backend.miner.case`` executes ``backend/miner/__init__.py`` -> ``run`` ->
``cluster``, and ``cluster`` imports numpy. Nothing in the script *names* numpy,
so no amount of reading it revealed the gap; the gate died on import on its
first execution ever, which was on the default branch, at night (JEB-1601).

So the list is no longer typed anywhere. It is `[project].dependencies` out of
pyproject.toml — the same declaration `pip install -e ".[dev]"` uses in
`ci.yml`'s `lint + tests` job — minus the packages named in :data:`EXCLUDED`.
A dependency added to the project is installed by the next nightly without
anyone remembering this file exists.

Only stdlib is used (``tomllib``, ``re``): this runs *before* pip has installed
anything, so it cannot import `packaging` to parse the requirement strings.

Usage:

    python .github/scripts/live_contract_requirements.py > requirements.txt
    pip install -r requirements.txt
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

#: Left out of the nightly install, with the reason each one is safe to omit.
#:
#: ``laya`` depends on torch — ~1 GB of CUDA wheel for a job that makes two API
#: calls. It is safe to drop because nothing on the probe's path imports it:
#: ``backend/brain/engine.py`` says ``import laya`` only *inside* the two
#: methods that load a model, which is a deliberate invariant of that module
#: ("``import laya`` appearing anywhere else is a bug"), asserted by
#: `tests/test_live_gemini_contract.py::test_nothing_the_probe_imports_needs_laya`.
#:
#: Anything added here must be justified the same way — by an import graph, not
#: by "the probe probably does not need it". Everything else is installed.
EXCLUDED = frozenset({"laya"})

#: Requirement strings look like ``name[extra]>=1,<2 ; marker``. The name is
#: whatever precedes the first extra/specifier/marker character.
_NAME = re.compile(r"[\s<>=!~\[;(]")


def requirement_name(spec: str) -> str:
    """The distribution name in a PEP 508 requirement string, normalised."""
    return _NAME.split(spec.strip(), 1)[0].strip().lower().replace("_", "-")


def requirements(pyproject: pathlib.Path) -> list[str]:
    """Every project dependency except :data:`EXCLUDED`, verbatim (pins included)."""
    declared = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    return [spec for spec in declared if requirement_name(spec) not in EXCLUDED]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = pathlib.Path(args[0]) if args else pathlib.Path("pyproject.toml")
    for spec in requirements(root):
        print(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
