#!/usr/bin/env python3
"""
calibrate_router.py — measure ROUTER_THRESHOLD against the real Laya checkpoint,
over both the library Pixel ships with and the library the miner produces.

Why this exists (JEB-1548). `ROUTER_THRESHOLD=0.6` was picked before anyone had
run the `choice` head, and it is wrong in both directions at once: phrases a
skill lists in its own `examples` came back under it (`сделай фокус` -> 0.55,
`а фокус умеешь?` -> 0.52, `хай` -> 0.22), while commands in nobody's library
came back over it (`покажи сальто` -> `show_trick` @0.78). The two ranges
overlap end to end, so *no* single number separates them — which is the finding,
not a threshold to be tuned harder. What this script does is price the trade-off
so the number is chosen instead of guessed.

Read it with `backend/brain/router.py`: step 0 there answers the first half of
the problem for free (a phrase a skill lists routes to that skill without a
forward pass), and this script's sweep answers the second.

Needs the weights and therefore a network on first run — it is a tool, not a
test. CI covers the router itself with `FakeEngine` (`tests/fakes.py`).

    python scripts/calibrate_router.py

Three sections:

  1. **Confidences.** Every phrase in the probe, routed through pass 1 alone,
     against each registry. Grouped into what the library claims (`examples`),
     paraphrases of the same intent that nobody wrote down, and commands that
     belong to no skill at all.
  2. **Sweep.** Per threshold: hits, misses (which cost a Gemini call and are
     recoverable) and wrong fires (which make the robot do the wrong thing AND
     never reach `teacher_log`, so nothing ever learns from them).
  3. **Per skill.** The same, split by skill — because `Skill.threshold` exists.
     The numbers move when the registry does, which is why they are printed
     rather than written into the seed files.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.brain.router import (
    ROUTER_INSTRUCTIONS,
    ROUTER_QUESTION,
    UNKNOWN,
    build_criteria,
)
from backend.brain.skill import Skill, load_seed_skills
from backend.brain.verbalize import command_only

THRESHOLDS = [round(0.50 + 0.02 * i, 2) for i in range(24)]

#: What the miner produces on the worked example. The starter four are the
#: library as shipped; this is the library one accepted proposal later, and the
#: option set is what the `choice` head is actually calibrated against.
MINED_TRICK = {
    "id": "show_trick",
    "name": "Фокус",
    "description": "показать фокус",
    "examples": ["покажи фокус", "сделай фокус", "удиви фокусом"],
    "rules": [
        {
            "when": {},
            "actions": [{"action": "spin"}, {"action": "say", "args": {"text": "Тада!"}}],
        }
    ],
    "origin": "mined",
}

#: Phrasings of an intent the library *has* that nobody wrote into `examples`.
#: This is most real traffic, and the only group the head has to earn.
PARAPHRASES: list[tuple[str, str]] = [
    ("приветствую", "greet"),
    ("здарова", "greet"),
    ("дай кушать", "feed"),
    ("есть хочешь?", "feed"),
    ("давай поиграем", "play"),
    ("хочу играть", "play"),
    ("спать пора", "sleep"),
    ("засыпай", "sleep"),
    ("а фокус покажешь", "show_trick"),
    ("фокус давай", "show_trick"),
]

#: Commands no skill covers. Every one of these must reach the teacher: that is
#: both the correct answer and the only way the miner ever sees them.
FOREIGN: list[str] = [
    "покажи сальто",
    "спой мне песенку",
    "расскажи про квантовую физику",
    "какая погода на улице",
    "сколько будет два плюс два",
    "кто твой создатель",
    "расскажи анекдот",
    "прыгни",
    "покрутись",
    "подмигни",
    "который час",
    "как тебя зовут",
    "переведи слово кошка",
    "что такое чёрная дыра",
    "напиши стихотворение",
    "открой окно",
    "мне грустно",
    "посчитай до десяти",
    "включи музыку",
    "почему небо голубое",
]


def registries() -> dict[str, list[Skill]]:
    seed = load_seed_skills()
    return {"starter (4 skills)": seed, "after one proposal (5)": [*seed, Skill(**MINED_TRICK)]}


def probe(skills: list[Skill]) -> list[tuple[str, str, str]]:
    """``(text, expected skill id or "" for foreign, group)`` for this registry."""
    known = {skill.id for skill in skills}
    rows = [(text, skill.id, "example") for skill in skills for text in skill.examples]
    rows += [(text, want, "paraphrase") for text, want in PARAPHRASES if want in known]
    rows += [(text, "", "foreign") for text in FOREIGN]
    return rows


def ask(engine, skills: list[Skill], text: str) -> tuple[str, float]:
    # Pass 1 only: which skill, and how sure. `route()` would answer `examples`
    # from its lookup, and the lookup is not what is being calibrated here.
    picked = engine.choice(
        command_only(text), ROUTER_QUESTION, ROUTER_INSTRUCTIONS, build_criteria(skills)
    )
    return picked.label, float(picked.confidence)


def report_confidences(measured: list[tuple[str, str, str, str, float]]) -> None:
    print("\n=== 1. pass-1 confidences ===")
    for group in ("example", "paraphrase", "foreign"):
        rows = sorted((c, t, w, l) for t, w, g, l, c in measured if g == group)
        print(f"  -- {group} ({len(rows)})")
        for confidence, text, want, label in rows:
            verdict = "ok" if label == (want or UNKNOWN) else "WRONG"
            print(f"     {confidence:.3f}  {text:<30} -> {label:<12} {verdict}")


def report_sweep(measured: list[tuple[str, str, str, str, float]]) -> None:
    """A miss costs one Gemini call; a wrong fire costs a wrong action AND the case."""
    library = [row for row in measured if row[2] != "foreign"]
    foreign = [row for row in measured if row[2] == "foreign"]

    print("\n=== 2. threshold sweep ===")
    print("   thr    hit   miss   wrongfire   hit%")
    for threshold in THRESHOLDS:
        hit = sum(1 for _, want, _, label, c in library if label == want and c >= threshold)
        wrong = sum(1 for _, want, _, label, c in library if label != want and c >= threshold)
        fires = sum(1 for *_, label, c in foreign if label != UNKNOWN and c >= threshold)
        print(
            f"  {threshold:.2f}  {hit:5d}  {len(library) - hit - wrong:5d}   "
            f"{wrong + fires:9d}   {hit / len(library) * 100:4.0f}%"
        )
    print("  pick the knee: the lowest threshold past which hits stop paying for wrong fires")


def report_per_skill(skills: list[Skill], measured: list[tuple[str, str, str, str, float]]) -> None:
    print("\n=== 3. per skill (`Skill.threshold` overrides the default) ===")
    for skill in skills:
        own = sorted(c for _, want, _, label, c in measured if want == skill.id == label)
        pull = sorted(
            (c, t) for t, want, _, label, c in measured if label == skill.id and want != skill.id
        )
        floor = f"{own[0]:.2f}" if own else "  - "
        ceiling = f"{pull[-1][0]:.2f} ({pull[-1][1]})" if pull else "-"
        print(f"  {skill.id:<12} own from {floor}   highest phrase it should not take: {ceiling}")
    print("  overlapping ranges mean no per-skill number separates them either — and these")
    print("  move with the option set, so they are printed, not written into the seed files")


def main() -> None:
    from backend.brain.engine import LayaEngine

    start = time.perf_counter()
    engine = LayaEngine()
    print(f"checkpoint loaded in {time.perf_counter() - start:.1f}s")

    for name, skills in registries().items():
        print(f"\n{'#' * 70}\n# registry: {name}\n{'#' * 70}")
        measured = [
            (text, want, group, *ask(engine, skills, text)) for text, want, group in probe(skills)
        ]
        report_confidences(measured)
        report_sweep(measured)
        report_per_skill(skills, measured)


if __name__ == "__main__":
    main()
