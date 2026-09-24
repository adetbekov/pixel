#!/usr/bin/env python3
"""
probe_miner_match.py — run whole clusters through the real teacher, the real
skill generator and the real backtest, and print why each of the backtest's
numbers comes out where it does.

Why this exists (JEB-1547). On live data the miner never published anything: a
cluster of paraphrases of one command scored 0.25..0.67 against
`MINER_MIN_MATCH=0.8`, and the draft it scored was a *refusal* — `{set_face,
say}` plus "я не умею показывать фокусы" — because that is what the teacher had
answered and `teacher_log` is the miner's only raw material. Lowering the bar
would have published exactly that skill. This script is what the fixes were
measured with, and it is here so the next person touching any of these numbers
measures instead of guessing:

  * the teacher improvises out of the library instead of declining, and says so
    in `handled` (`backend/teacher/prompt.py`, `backend/teacher/schema.py`);
  * `match_rate` is what production will do with the cluster after acceptance,
    the `examples` lookup included; the head's own answer is reported as
    `generalization` and the teacher-plan comparison as `agreement`, and neither
    gates (`backend/miner/backtest.py`, JEB-1562).

All three are printed for every cluster, per run, so the argument for the split
can be re-run rather than believed — and so can the threshold: the last block
prints the share of clusters that publish, which is what `MINER_MIN_MATCH` is
calibrated against.

Needs `GEMINI_API_KEY`, the Laya weights and therefore a network on first run —
it is a tool, not a test. CI covers all of this with `FakeEngine` and
`FakeGeminiClient`; what CI cannot cover is what the live model answers.

    python scripts/probe_miner_match.py [runs]

`runs` defaults to 1. Every run re-teaches and re-drafts from scratch, because
that is where the spread lives: Gemini rewrites the draft's `id` and
`description` each time, and the `choice` head scores exactly those.

The three clusters are deliberately two near-synonymous intents plus one that is
not, and the pool carries three one-off `LEFTOVERS` besides, because check 3
treats the two halves differently: a claim on a mineable neighbour's phrase is
temporary and not a veto, a claim on a leftover is permanent and is (JEB-1579).

Costs a handful of `gemini-2.5-flash-lite` calls per run: one teacher call per
command plus one generate call per cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.brain.router import RouterHit, default_threshold, normalize, pick_skill, route
from backend.brain.skill import Skill, load_seed_skills
from backend.miner.backtest import Backtest, backtest, min_match_rate
from backend.miner.case import Case
from backend.miner.cluster import min_cluster_size
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
    # A third intent, and deliberately one the library expresses with `say` and
    # `set_face` rather than movement: the threshold has to be calibrated on more
    # than one shape of cluster (JEB-1562).
    "похвала": [
        "похвали меня",
        "скажи что-нибудь приятное",
        "сделай мне комплимент",
        "скажи что я молодец",
        "хочу похвалу",
    ],
}

#: One-off commands the teacher *can* express, each a different intent, none of
#: them repeated. In a real pool these are the rows the grouper leaves in
#: sub-`MINER_MIN_CLUSTER` groups, so nothing will ever claim them back — which
#: is exactly what makes them the over-broad control set (JEB-1579). A draft that
#: wins one of these keeps it for good; a draft that wins a *neighbouring
#: cluster's* phrase only keeps it until that cluster is mined.
LEFTOVERS: list[str] = [
    "спой песню",
    "расскажи анекдот",
    "посчитай до десяти",
]

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


def report_backtest(
    engine,
    active: list[Skill],
    candidate: Skill,
    cases: list[Case],
    outsiders: list[list[Case]],
) -> Backtest:
    """Print the two routers side by side, then the numbers the backtest reports.

    Two columns per case, because JEB-1562 is the difference between them:
    `prod` is `route` as `/api/chat` will call it after acceptance — step 0, the
    exact `examples` lookup, included — and `head` is `pick_skill`, the `choice`
    head with nothing to look up. The generator is told to copy the cluster into
    `examples` word for word, so `prod` is normally 1.00 on every line while
    `head` is whatever the draft's own wording earns this run.
    """
    print(f"  candidate {candidate.id!r} / {candidate.description!r}")
    for rule in candidate.rules:
        print(f"    when={rule.when or '{}'} -> {[s['action'] for s in rule.actions]}")

    trial = [*active, candidate]
    for case in cases:
        outcome = route(engine, trial, case.user_text, case.state)
        picked = outcome.skill_id if isinstance(outcome, RouterHit) else "miss"
        covered = picked == candidate.id
        got = sorted(action_set(outcome.raw_plan)) if isinstance(outcome, RouterHit) else []
        agrees = covered and set(got) == case.action_names

        head, head_confidence = pick_skill(engine, trial, case.user_text)
        head_id = head.id if head is not None else "miss"
        drafted = {normalize(example) for example in candidate.examples}
        listed = "listed" if normalize(case.user_text) in drafted else "-"
        print(
            f"  {case.user_text!r:30} prod={picked:12} @{outcome.confidence:.2f} {listed:6} "
            f"head={head_id:12} @{head_confidence:.2f} agrees={agrees!s:5} "
            f"candidate={got} teacher={sorted(case.action_names)}"
        )

    report = backtest(engine, active, candidate, cases, outsiders)
    print(
        f"  match_rate {report.match_rate:.2f} ({report.matched}/{report.total}), "
        f"generalization {report.generalization:.2f}, agreement {report.agreement:.2f}"
    )
    # What the candidate took from the *mineable* neighbours: no longer a veto
    # (JEB-1579), and printed because it is the thing the veto used to be.
    neighbours = [
        case
        for group in outsiders
        if len(group) >= min_cluster_size()
        for case in group
    ]
    claimed = [
        (case.user_text, confidence)
        for case, (picked, confidence) in (
            (case, pick_skill(engine, trial, case.user_text)) for case in neighbours
        )
        if picked is not None and picked.id == candidate.id
    ]
    for text, confidence in claimed:
        print(f"  claims {text!r} from a mineable neighbour @{confidence:.2f} — not a veto")
    print(f"  regression {report.regression}, overreach {report.overreach}")
    print(
        f"  publishable={report.publishable} "
        f"(MINER_MIN_MATCH={min_match_rate():.2f}, ROUTER_THRESHOLD={default_threshold():.2f})"
    )
    return report


def report_calibration(results: dict[str, list[Backtest | None]]) -> None:
    """The threshold's own measurement: how often each cluster publishes.

    `MINER_MIN_MATCH` is not a matter of taste — it is the share of publishable
    clusters it produces on live data, which is what this block prints. A number
    the good clusters cannot reach stops the learning loop dead (JEB-1547), and a
    number every draft clears makes checks 4 and 5 the only gates there are.
    """
    print(f"\n=== calibration: MINER_MIN_MATCH={min_match_rate():.2f} ===")
    published = 0
    total = 0
    for name, reports in results.items():
        for label, get in (
            ("match_rate", lambda r: f"{r.match_rate:.2f}"),
            ("generalization", lambda r: f"{r.generalization:.2f}"),
            ("agreement", lambda r: f"{r.agreement:.2f}"),
            ("publishable", lambda r: str(r.publishable)),
        ):
            cells = " ".join(f"{(get(r) if r is not None else 'n/a'):>6}" for r in reports)
            print(f"  {name:10} {label:14} {cells}")
        published += sum(1 for r in reports if r is not None and r.publishable)
        total += len(reports)
    print(f"  publishable clusters: {published}/{total}")


def main() -> None:
    from backend.brain.engine import LayaEngine

    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    engine = LayaEngine()
    active = load_seed_skills()
    teacher = GeminiTeacher()
    generator = GeminiSkillGenerator()
    print(f"library: {[skill.id for skill in active]}, runs: {runs}")

    print("\n=== 1. commands the library cannot express ===")
    kept = teach(teacher, active, UNTEACHABLE)
    print(f"  mineable after the `handled` filter: {len(kept)}/{len(UNTEACHABLE)} (want 0)")

    results: dict[str, list[Backtest | None]] = {name: [] for name in CLUSTERS}
    for run in range(1, runs + 1):
        # Every cluster is taught before any of them is drafted, because each
        # candidate's over-broad control set is the other clusters — and every run
        # re-teaches and re-drafts from scratch, since the draft's wording is where
        # the spread the head reacts to comes from.
        taught: dict[str, list[Case]] = {}
        print(f"\n=== run {run}, the pool's leftovers ===")
        taught["leftovers"] = teach(teacher, active, LEFTOVERS)
        for name, texts in CLUSTERS.items():
            print(f"\n=== run {run}, cluster {name!r}: what the teacher answered ===")
            cases = teach(teacher, active, texts)
            print(f"  mineable: {len(cases)}/{len(texts)}")
            report_teacher_spread(cases)
            taught[name] = cases

        for name, cases in taught.items():
            if name == "leftovers":
                continue
            print(f"\n=== run {run}, cluster {name!r}: the draft, case by case ===")
            candidate = generator.propose(cases, active)
            if candidate is None:
                print("  the generator produced nothing — see the log above")
                results[name].append(None)
                continue
            # The rest of the pool, grouped as a real run's grouper would leave
            # it: the other clusters, plus the leftovers one per group, because
            # check 3 vetoes on group size (JEB-1579, `backtest.leftovers`).
            outsiders = [group for other, group in taught.items() if other not in (name, "leftovers")]
            outsiders += [[case] for case in taught["leftovers"]]
            results[name].append(report_backtest(engine, active, candidate, cases, outsiders))

    report_calibration(results)


if __name__ == "__main__":
    main()
