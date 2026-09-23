"""Does the candidate actually work — and did it break anything that did?

The user is asked to approve a skill, so the skill has to have been tried first.
Two checks, and a proposal needs both:

**1. Does it cover its own cluster?** Every case is re-routed against
``active + candidate``, and a match means the router picked *the candidate* above
its threshold. ``match_rate`` is matches over cluster size, and it is deliberately
measured against the *current* library — a candidate that steals "покорми" shows
up as its own low ``match_rate``, which is a reason to drop the candidate, not
to lower the bar.

``match_rate`` used to demand a second thing of every case: that the candidate's
plan name the same actions the teacher's did. That is what ``agreement`` reports
now, and it no longer gates. Three measurements, all on live data, in the order
they were taken (JEB-1547, ``scripts/probe_miner_match.py``):

* Set equality is unreachable when the teacher varies. Five phrasings of "покажи
  фокус" produced four different action sets, so the most any single plan could
  score was 3/5 = 0.60 — under a bar of 0.80, before the candidate was drafted.
  Teaching the teacher to improvise rather than decline
  (:mod:`backend.teacher.prompt`) narrowed that to two sets and a 0.60 ceiling on
  the same cluster. Still under the bar.
* One action of difference scores a case zero. On that same cluster the candidate
  came back with ``{jump, spin, wave, set_face, say}`` and three of the five
  teacher plans had no ``wave`` — so coverage was 0.60 and agreement 0.20. The
  measure has no notion of "close": it is exact set equality or nothing, and the
  teacher does not repeat itself.
* And it is anti-correlated with quality. The highest-scoring candidate on a
  cluster of refusals is the one that *reproduces the refusal*: it matches every
  case's ``{set_face, say}`` and scores 1.0. A candidate that actually does the
  thing scores 0. Gating on this selected for the one skill nobody wants — one
  that answers "я не умею" for ever, instantly, from Laya, with no route left to
  the teacher.

So the teacher-plan comparison is kept, computed and logged, because a cluster
whose cases disagree is worth knowing about — but as a description of the
*teacher's* variance, not a test of the candidate. What keeps the candidate
honest instead: refusals never enter the pool at all
(:func:`backend.miner.case._parse`), the regression check below, and the user,
who is shown the skill before it is activated.

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
    #: Share of the cluster the router sends to the candidate above its
    #: threshold. This is the gate.
    match_rate: float
    matched: int
    total: int
    #: Share of the cluster whose *teacher* plan the candidate also reproduces,
    #: action names only. Reported, never gated — see the module docstring. It
    #: says how much the teacher varied across the cluster, which is a fact about
    #: the raw material rather than about this candidate.
    agreement: float = 0.0
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

    matched = 0
    agreed = 0
    for case in cases:
        outcome = route(engine, trial, case.user_text, case.state)
        if not _routes_to(outcome, candidate):
            continue
        matched += 1
        if _does_what_the_teacher_did(outcome, case):
            agreed += 1

    total = len(cases)
    return Backtest(
        match_rate=matched / total if total else 0.0,
        matched=matched,
        total=total,
        # Over the whole cluster, not over the routed cases: a case the router
        # never reaches cannot agree with anything, and dividing by a shrinking
        # denominator would make a candidate that covers one case out of five
        # look like perfect agreement.
        agreement=agreed / total if total else 0.0,
        regression=check_regressions(engine, trial, active),
    )


def _routes_to(outcome: RouterOutcome, candidate: Skill) -> bool:
    # `route` already applies the skill's threshold, so a RouterHit *is* a
    # confident pick — there is no second comparison to make here.
    return isinstance(outcome, RouterHit) and outcome.skill_id == candidate.id


def _does_what_the_teacher_did(outcome: RouterHit, case: Case) -> bool:
    """Action names only: order and arguments are compared nowhere, because the
    teacher never says the same sentence twice."""
    return {step["action"] for step in outcome.raw_plan} == case.action_names
