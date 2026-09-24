#!/usr/bin/env python3
"""
probe_miner_match.py — run one cluster through the real teacher, the real skill
generator and the real backtest, and print why `match_rate` comes out where it
does.

Why this exists (JEB-1547). On live data the miner never published anything: a
cluster of paraphrases of one command scored 0.25..0.67 against
`MINER_MIN_MATCH=0.8`, and the draft it scored was a *refusal* — `{set_face,
say}` plus "я не умею показывать фокусы" — because that is what the teacher had
answered and `teacher_log` is the miner's only raw material. Lowering the bar
would have published exactly that skill. This script is what the two fixes were
measured with, and it is here so the next person touching either number measures
instead of guessing:

  * the teacher improvises out of the library instead of declining, and says so
    in `handled` (`backend/teacher/prompt.py`, `backend/teacher/schema.py`);
  * `match_rate` counts routing coverage, and the teacher-plan comparison is
    reported as `agreement` next to it instead of gating
    (`backend/miner/backtest.py`).

Both numbers are printed for every cluster, so the argument for the split can be
re-run rather than believed.

Needs `GEMINI_API_KEY`, the Laya weights and therefore a network on first run —
it is a tool, not a test. CI covers all of this with `FakeEngine` and
`FakeGeminiClient`; what CI cannot cover is what the live model answers.

    python scripts/probe_miner_match.py

Costs a handful of `gemini-2.5-flash-lite` calls: one teacher call per command
plus one generate call per cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.brain.router import RouterHit, default_threshold, route
from backend.brain.skill import Skill, load_seed_skills
from backend.miner.backtest import backtest, min_match_rate
from backend.miner.case import Case
from backend.miner.generate import GeminiSkillGenerator
from backend.state import RobotState
from backend.teacher.client import GeminiTeacher

#: The state every probe command is asked in: unremarkable, so the answer is
#: about the command and not about a tired robot refusing to jump.
STATE = RobotState(mood=50.0, energy=50.0, fullness=50.0, face="curious")

#: One intent per cluster, phrased the way a user would phrase it. "фокус" is the
#: cluster from the live QA run the issue was written from; "сальто" is the one
#: that scored 0.25 there.
CLUSTERS: dict[str, list[str]] = {
    "фокус": [
        "покажи фокус",
        "сделай фокус",
        "фокус покажи",
        "удиви фокусом",
        "а фокус умеешь?",
    ],
    "сальто": [
        "сделай сальто",
        "покажи сальто",
        "сальто умеешь?",
        "крутани сальто",
        "хочу увидеть сальто",
    ],
}

#: Commands the library genuinely cannot express. They are here to check the
#: other half of the fix: the teacher must decline these *and* mark them, so they
#: never reach the miner.
UNTEACHABLE: list[str] = [
    "какая сегодня погода",
    "закажи пиццу",
    "переведи слово кот на английский",
]


def action_set(plan: list[dict]) -> set[str]:
    return {step["action"] for step in plan}


def teach(teacher: GeminiTeacher, skills: list[Skill], texts: list[str]) -> list[Case]:
    """Ask the teacher about every command, exactly as `/api/chat` would."""
    cases = []
    for index, text in enumerate(texts, start=1):
        result = teacher.explain(text, STATE, skills)
        names = sorted(action_set(result.raw_plan))
        flag = "handled" if result.handled else "DECLINED"
        print(f"  {text!r:26} {flag:9} {names} {result.reply[:48]!r}")
        cases.append(
            Case(id=index, user_text=text, state=STATE, actions=result.raw_plan)
            if result.handled and not result.error
            else None
        )
    return [case for case in cases if case is not None]


def report_teacher_spread(cases: list[Case]) -> None:
    """The ceiling the old `match_rate` had, before the candidate is even drafted.

    The old criterion demanded the candidate reproduce *each* case's action set,
    so the best any single plan could score was the share of the most common set.
    """
    sets = [frozenset(case.action_names) for case in cases]
    if not sets:
        print("  no mineable case survived — nothing to draft from")
        return
    modal = max(set(sets), key=sets.count)
    print(f"  {len(set(sets))} distinct action set(s) over {len(sets)} case(s)")
    print(
        f"  most common {sorted(modal)} in {sets.count(modal)}/{len(sets)} "
        f"-> a single plan could score at most {sets.count(modal) / len(sets):.2f} "
        "under set-equality"
    )


def report_backtest(engine, active: list[Skill], candidate: Skill, cases: list[Case]) -> None:
    print(f"  candidate {candidate.id!r} / {candidate.description!r}")
    for rule in candidate.rules:
        print(f"    when={rule.when or '{}'} -> {[s['action'] for s in rule.actions]}")

    for case in cases:
        # `use_examples=False`, exactly as the backtest routes: a candidate's
        # `examples` *are* the cluster under test, so the router's step-0 lookup
        # would answer every line below from the draft itself and print 1.00
        # against a `match_rate` computed from the head (JEB-1548).
        outcome = route(
            engine, [*active, candidate], case.user_text, case.state, use_examples=False
        )
        picked = outcome.skill_id if isinstance(outcome, RouterHit) else "miss"
        routed = picked == candidate.id
        got = sorted(action_set(outcome.raw_plan)) if isinstance(outcome, RouterHit) else []
        agrees = routed and set(got) == case.action_names
        print(
            f"  {case.user_text!r:26} -> {picked:12} @{outcome.confidence:.2f} "
            f"routed={routed!s:5} agrees={agrees!s:5} "
            f"candidate={got} teacher={sorted(case.action_names)}"
        )

    report = backtest(engine, active, candidate, cases)
    print(
        f"  match_rate {report.match_rate:.2f} ({report.matched}/{report.total}), "
        f"agreement {report.agreement:.2f}, regression {report.regression}"
    )
    print(
        f"  publishable={report.publishable} "
        f"(MINER_MIN_MATCH={min_match_rate():.2f}, ROUTER_THRESHOLD={default_threshold():.2f})"
    )


def main() -> None:
    from backend.brain.engine import LayaEngine

    engine = LayaEngine()
    active = load_seed_skills()
    teacher = GeminiTeacher()
    generator = GeminiSkillGenerator()
    print(f"library: {[skill.id for skill in active]}")

    print("\n=== 1. commands the library cannot express ===")
    kept = teach(teacher, active, UNTEACHABLE)
    print(f"  mineable after the `handled` filter: {len(kept)}/{len(UNTEACHABLE)} (want 0)")

    for name, texts in CLUSTERS.items():
        print(f"\n=== 2. cluster {name!r}: what the teacher answered ===")
        cases = teach(teacher, active, texts)
        print(f"  mineable: {len(cases)}/{len(texts)}")
        report_teacher_spread(cases)

        print(f"\n=== 3. cluster {name!r}: the draft, case by case ===")
        candidate = generator.propose(cases, active)
        if candidate is None:
            print("  the generator produced nothing — see the log above")
            continue
        report_backtest(engine, active, candidate, cases)


if __name__ == "__main__":
    main()
