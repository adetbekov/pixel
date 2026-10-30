#!/usr/bin/env python3
"""
check_router_foreign_commands.py — a command the library does not have must
reach the teacher, often enough for the miner to ever see it (JEB-1561).

Why this exists. The learning loop starts at a router *miss*: an unfamiliar
command goes to Gemini, lands in `teacher_log`, and every `MINER_BATCH`-th
mineable case asks for a mining run. A starter skill that takes such a command
above its threshold breaks that at the first step — the robot does the wrong
thing, nothing is logged, and the cluster never reaches the batch. QA on
JEB-1547 (`8dccba3`, `ROUTER_THRESHOLD=0.6`) found exactly that on a clean
database:

    сделай сальто        -> teacher  0.236
    покажи сальто        -> play     0.605   absorbed
    сальто умеешь?       -> teacher  0.137
    крутани сальто       -> play     0.631   absorbed
    хочу увидеть сальто  -> teacher  0.515

Three of five reached the teacher, the batch is five, the miner never ran.

Why `lint + tests` could not see it: every test there routes through
`tests/fakes.py::FakeEngine`, which answers whatever it is scripted to. Real
confidences against the real library are checked nowhere else in the
repository, so a recalibration (JEB-1548) or a new seed skill can quietly
reopen the hole. This runs the real `LayaEngine` and needs no `GEMINI_API_KEY`:
the router is local and the library is fixed, so the verdict is deterministic.

What it checks, per topic in `FOREIGN_TOPICS`, through `route()` — the same
function `/api/chat` calls, with its `examples` lookup and every skill's own
`threshold` override:

  1. no phrase is taken by an active starter skill;
  2. at least `MINER_BATCH` phrases fall through to the teacher.

The thresholds are read from the code (`default_threshold()`, `batch_size()`),
never restated here: the gate checks the numbers the product runs with.

    uv sync --frozen --no-dev --extra engine
    PYTHONPATH=. uv run --frozen --no-dev --extra engine \
        python .github/scripts/check_router_foreign_commands.py

Exit codes, same convention as `check_live_gemini_contract.py`:
  0  every topic reached the teacher in full
  1  a finding: a phrase was absorbed, or a topic falls short of the batch
  3  crashed before judging — import, weights download, a harness bug
"""

from __future__ import annotations

import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

EXIT_OK = 0
EXIT_FINDING = 1
EXIT_CRASHED = 3

# Python puts the script's own directory on sys.path, not the repository root;
# the workflow sets PYTHONPATH=. as well, this covers a bare local run.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from backend.brain.engine import DecisionEngine
    from backend.brain.router import RouterHit, default_threshold, route
    from backend.brain.skill import Skill, load_seed_skills
    from backend.miner.run import batch_size
    from backend.state import START_ENERGY, START_FACE, START_FULLNESS, START_MOOD, RobotState
except Exception:  # noqa: BLE001 - a broken import graph is exit 3, not a finding
    traceback.print_exc()
    sys.exit(EXIT_CRASHED)

#: Commands for intents no starter skill has, five ways each — one `MINER_BATCH`
#: worth, so a single absorbed phrase is already a cluster the miner never sees.
#: `сальто` is the QA reproduction verbatim; `фокус` is the stage-4 worked
#: example (`tests/trick_cluster.py`); the other two are requests the action
#: library can only answer with `say`, so no starter skill has a claim on them.
#:
#: Deliberately absent: songs. Measured on the starter library, `спой песенку`
#: (0.770), `давай споём` (0.685) and `исполни песню` (0.997) go to `play`, whose
#: own examples include `развлеки меня` and `потанцуем`. Whether "sing" is a kind
#: of "play" is a product call, not a routing defect, and a gate that is red on
#: an open question gets ignored.
FOREIGN_TOPICS: dict[str, list[str]] = {
    "сальто": [
        "сделай сальто",
        "покажи сальто",
        "сальто умеешь?",
        "крутани сальто",
        "хочу увидеть сальто",
    ],
    "фокус": [
        "покажи фокус",
        "сделай фокус",
        "фокус покажи",
        "удиви фокусом",
        "а фокус умеешь?",
    ],
    "анекдот": [
        "расскажи анекдот",
        "пошути",
        "расскажи шутку",
        "рассмеши меня",
        "знаешь анекдоты?",
    ],
    "стих": [
        "прочитай стих",
        "расскажи стихотворение",
        "почитай стихи",
        "знаешь стишок?",
        "продекламируй стих",
    ],
}

#: A robot on a clean database — where QA measured. Pass 1, the one that decides
#: hit or miss, never sees the state (`backend/brain/verbalize.py`); only the
#: plan a hit builds does.
FRESH_ROBOT = RobotState(
    mood=START_MOOD, energy=START_ENERGY, fullness=START_FULLNESS, face=START_FACE
)


@dataclass(frozen=True)
class Routed:
    topic: str
    text: str
    skill_id: str | None
    confidence: float


def route_topics(
    engine: DecisionEngine,
    skills: list[Skill],
    topics: dict[str, list[str]] = FOREIGN_TOPICS,
) -> list[Routed]:
    rows = []
    for topic, phrases in topics.items():
        for text in phrases:
            outcome = route(engine, skills, text, FRESH_ROBOT)
            skill_id = outcome.skill_id if isinstance(outcome, RouterHit) else None
            rows.append(Routed(topic, text, skill_id, outcome.confidence))
    return rows


def judge(rows: list[Routed]) -> list[str]:
    """Every reason the gate is red, in a form a PR reader can act on."""
    findings = []
    batch = batch_size()
    for topic in dict.fromkeys(row.topic for row in rows):
        own = [row for row in rows if row.topic == topic]
        for row in own:
            if row.skill_id is not None:
                findings.append(
                    f"{topic}: '{row.text}' absorbed by '{row.skill_id}' @{row.confidence:.3f}"
                    " — a command the library does not have never reaches the teacher"
                )
        reached = sum(1 for row in own if row.skill_id is None)
        if reached < batch:
            findings.append(
                f"{topic}: {reached}/{len(own)} reached the teacher, MINER_BATCH={batch}"
                " — the miner never runs on this cluster"
            )
    return findings


def report(rows: list[Routed]) -> None:
    print(f"ROUTER_THRESHOLD={default_threshold():.2f} MINER_BATCH={batch_size()}")
    print(f"  {'topic':<9} {'phrase':<26} {'routed to':<10} conf")
    for row in rows:
        target = row.skill_id or "teacher"
        mark = "   <- absorbed" if row.skill_id else ""
        print(f"  {row.topic:<9} {row.text:<26} {target:<10} {row.confidence:.3f}{mark}")


def main() -> int:
    try:
        from backend.brain.engine import LayaEngine

        started = time.perf_counter()
        engine = LayaEngine()
        print(f"checkpoint loaded in {time.perf_counter() - started:.1f}s")
        skills = [skill for skill in load_seed_skills() if skill.status == "active"]
        print(f"starter library: {', '.join(skill.id for skill in skills)}")
        rows = route_topics(engine, skills)
    except Exception:  # noqa: BLE001 - nothing was judged, so this must not read as a finding
        traceback.print_exc()
        return EXIT_CRASHED

    report(rows)
    findings = judge(rows)
    if not findings:
        print("OK: every unfamiliar command reached the teacher, one full batch per topic")
        return EXIT_OK
    for finding in findings:
        print(f"::error::{finding}")
    return EXIT_FINDING


if __name__ == "__main__":
    sys.exit(main())
