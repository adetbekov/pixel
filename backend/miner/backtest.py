"""Does the candidate actually work — and did it break anything that did?

The user is asked to approve a skill, so the skill has to have been tried first.
Three checks, and a proposal needs all of them:

**1. Does it cover its own cluster?** Every case is re-routed against
``active + candidate``. A match means the router picked *the candidate* above
its threshold **and** did what the teacher did. Action names only: order and
arguments are compared nowhere, because the teacher never says the same sentence
twice. ``match_rate`` is matches over cluster size, and it is deliberately
measured against the *current* library — a candidate that steals "покорми" shows
up as its own low ``match_rate``, which is a reason to drop the candidate, not
to lower the bar.

**2. Did it break the skills that already worked?** This is the check a match
rate cannot give you. Stage 2 measured that the *order of the options* in the
router's ``choice`` question moves the answer: all 24 orderings of the four
starter skills spread the hit rate over 7/10 … 9/10, and alphabetical order
pushed "покорми" under its threshold outright. A mined skill appends to that
option list, so it changes the question for every existing skill — and a
candidate can score 1.0 on its own cluster while knocking ``feed`` off the map.

So one control phrase per active skill (``examples[0]``, so the control set
grows with the library) is routed through the same trial registry, and if a
single one stops reaching its own skill the proposal is not published at all.

**3. Is it over-broad?** (JEB-1548.) Checks 1 and 2 only ever look at phrases
the library already claims, so neither can see a candidate reaching for commands
that are in nobody's ``examples`` yet. Measured on the live checkpoint: an
accepted ``show_trick`` ("показать фокус") pulled "покажи сальто" to itself at
0.78 while every ``examples[0]`` control passed cleanly. The pool has the right
control set for free — the *other* cases in ``teacher_log``, the ones the
grouper put in other clusters. They are commands a real user typed that this
candidate is not for, so a candidate that wins any of them is drafted too wide
and is not proposed.

None of the three may go through the router's exact-``examples`` lookup: a
candidate's ``examples`` *are* the cluster under test, so the lookup would answer
checks 1 and 3 from its own draft and score every candidate 1.0. Check 1 needs
the plan the router builds, so it calls ``route(..., use_examples=False)``.
Checks 2 and 3 only ever read *which skill* won, so they call
:func:`backend.brain.router.pick_skill` — pass 1 on its own, half the forward
passes, and no lookup to switch off.

They run cheapest-first, and check 3 runs last: see :func:`backtest`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..brain.engine import DecisionEngine
from ..brain.router import RouterHit, RouterOutcome, pick_skill, route
from ..brain.skill import Skill
from .case import Case

DEFAULT_MIN_MATCH = 0.8


def min_match_rate() -> float:
    return float(os.environ.get("MINER_MIN_MATCH", DEFAULT_MIN_MATCH))


@dataclass(frozen=True)
class Backtest:
    match_rate: float
    matched: int
    total: int
    #: ``None`` when every active skill still routes to itself; otherwise the
    #: phrase that moved and where it went, so the log says what broke.
    regression: str | None = None
    #: ``None`` when the candidate left the rest of the pool alone; otherwise the
    #: outside-cluster command it took and the confidence it took it at.
    overreach: str | None = None

    @property
    def publishable(self) -> bool:
        return (
            self.regression is None
            and self.overreach is None
            and self.match_rate >= min_match_rate()
        )


def check_regressions(
    engine: DecisionEngine, trial: list[Skill], active: list[Skill]
) -> str | None:
    """First control phrase that stops reaching its own skill, if any."""
    for skill in active:
        if not skill.examples:
            continue
        phrase = skill.examples[0]
        picked, confidence = pick_skill(engine, trial, phrase)
        if picked is not None and picked.id == skill.id:
            continue
        went = picked.id if picked is not None else "miss"
        return f'"{phrase}" ({skill.id}) -> {went} @ {confidence:.2f}'
    return None


def check_overreach(
    engine: DecisionEngine, trial: list[Skill], candidate: Skill, outsiders: list[Case]
) -> str | None:
    """First pool command from *outside* the cluster that the candidate takes.

    The only unbounded loop in this module — the pool has no ``LIMIT`` — so
    :func:`backtest` calls it last and only when nothing else has already
    rejected the candidate.
    """
    for case in outsiders:
        picked, confidence = pick_skill(engine, trial, case.user_text)
        if picked is not None and picked.id == candidate.id:
            return f'"{case.user_text}" (another cluster) -> {candidate.id} @ {confidence:.2f}'
    return None


def backtest(
    engine: DecisionEngine,
    active: list[Skill],
    candidate: Skill,
    cases: list[Case],
    outsiders: list[Case] | None = None,
) -> Backtest:
    """Run the candidate against its own cluster, the library and the rest of the pool.

    The candidate is appended, never inserted: that is where ``load_skills``
    puts a mined skill (``ORDER BY rowid``), so the trial registry has to have
    the same option order the live one will have after acceptance.

    **Cheapest check first, and the unbounded one last.** Checks 1 and 2 cost one
    forward pass per cluster case and per active skill — both small and both
    fixed. Check 3 costs one per *unmined row in the pool*, and the pool does not
    shrink for a candidate that fails, so a run that rejects everything would pay
    it over and over. A candidate that has already lost on ``match_rate`` or on a
    regression cannot be published whatever check 3 says, so it is not run
    (JEB-1548 review): ``overreach`` stays ``None``, which reads the same as "it
    took nothing" and is why ``publishable`` still has to test all three.
    """
    trial = [*active, candidate]

    matched = sum(
        1
        for case in cases
        if _covers(
            route(engine, trial, case.user_text, case.state, use_examples=False), candidate, case
        )
    )
    rate = matched / len(cases) if cases else 0.0
    regression = check_regressions(engine, trial, active)

    overreach = None
    if regression is None and rate >= min_match_rate():
        overreach = check_overreach(engine, trial, candidate, outsiders or [])

    return Backtest(
        match_rate=rate,
        matched=matched,
        total=len(cases),
        regression=regression,
        overreach=overreach,
    )


def _covers(outcome: RouterOutcome, candidate: Skill, case: Case) -> bool:
    # `route` already applies the skill's threshold, so a RouterHit *is* a
    # confident pick — there is no second comparison to make here.
    if not isinstance(outcome, RouterHit) or outcome.skill_id != candidate.id:
        return False
    return {step["action"] for step in outcome.raw_plan} == case.action_names
