"""Does the candidate actually work — and did it break anything that did?

The user is asked to approve a skill, so the skill has to have been tried first.
Two checks, and a proposal needs both:

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
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..brain.engine import DecisionEngine
from ..brain.router import RouterHit, RouterOutcome, route
from ..brain.skill import Skill
from ..state import RobotState
from .case import Case

DEFAULT_MIN_MATCH = 0.8

#: Controls check *which skill* was picked, and pass 1 of the router sees the
#: command alone — so the state here only has to be unremarkable.
CONTROL_STATE = RobotState(mood=50.0, energy=50.0, fullness=50.0, face="curious")


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

    @property
    def publishable(self) -> bool:
        return self.regression is None and self.match_rate >= min_match_rate()


def check_regressions(
    engine: DecisionEngine, trial: list[Skill], active: list[Skill]
) -> str | None:
    """First control phrase that stops reaching its own skill, if any."""
    for skill in active:
        if not skill.examples:
            continue
        phrase = skill.examples[0]
        outcome = route(engine, trial, phrase, CONTROL_STATE)
        if isinstance(outcome, RouterHit) and outcome.skill_id == skill.id:
            continue
        went = outcome.skill_id if isinstance(outcome, RouterHit) else "miss"
        return f'"{phrase}" ({skill.id}) -> {went} @ {outcome.confidence:.2f}'
    return None


def backtest(
    engine: DecisionEngine, active: list[Skill], candidate: Skill, cases: list[Case]
) -> Backtest:
    """Run the candidate against its own cluster and against the library.

    The candidate is appended, never inserted: that is where ``load_skills``
    puts a mined skill (``ORDER BY rowid``), so the trial registry has to have
    the same option order the live one will have after acceptance.
    """
    trial = [*active, candidate]

    matched = sum(
        1
        for case in cases
        if _covers(route(engine, trial, case.user_text, case.state), candidate, case)
    )
    rate = matched / len(cases) if cases else 0.0
    return Backtest(
        match_rate=rate,
        matched=matched,
        total=len(cases),
        regression=check_regressions(engine, trial, active),
    )


def _covers(outcome: RouterOutcome, candidate: Skill, case: Case) -> bool:
    # `route` already applies the skill's threshold, so a RouterHit *is* a
    # confident pick — there is no second comparison to make here.
    if not isinstance(outcome, RouterHit) or outcome.skill_id != candidate.id:
        return False
    return {step["action"] for step in outcome.raw_plan} == case.action_names
