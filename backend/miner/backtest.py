"""Does the candidate actually work — and did it break anything that did?

The user is asked to approve a skill, so the skill has to have been tried first.
Three checks, and a proposal needs all of them:

**1. Does it cover its own cluster?** ``match_rate`` is the share of the cluster
that stops reaching Gemini once the user accepts this skill. Every case is
re-routed through ``active + candidate`` by exactly the call ``/api/chat`` will
make after acceptance — :func:`backend.brain.router.route`, step 0 (the exact
``examples`` lookup) included — and a match means the router landed on *the
candidate* above its threshold. That is the whole definition: the product metric
("share of commands handled without Gemini") restricted to this cluster, which is
why it is the one number that gates.

It is deliberately measured against the *current* library, so a candidate that
steals "покорми" shows up as its own low ``match_rate`` — a reason to drop the
candidate, not to lower the bar.

The same name used to mean two other things. Neither of them does now.

**It was a measurement of the ``choice`` head** (``use_examples=False``,
JEB-1548). The lookup was switched off because a candidate's ``examples`` *are*
the cluster under test, so leaving it on would score every candidate 1.0 — true,
and the conclusion drawn from it was that the head's answer is the interesting
one. It is not the answer production gives. The generator is told to copy the
cluster into ``examples`` word for word (:mod:`backend.miner.generate`), so after
acceptance those phrases route at 1.0 through step 0 and the head is never asked.
Measured on the live checkpoint (JEB-1562, ``scripts/probe_miner_match.py``): the
"фокус" cluster scored 0.60 in three runs out of three and was refused on two
phrases — "сделай фокус" @0.55 and "а фокус умеешь?" @0.52 — that step 0 routes
at 1.0. On five paraphrases of one command the same probe read 0.60 / 1.00 / 0.80
/ 0.40 / 0.80 / 0.40 over six runs, because the head scores the ``id`` and
``description`` Gemini rewrites on every draft: ``do_salto`` passed where
``do_somersault`` failed on the same five phrases. A publish gate that is a coin
flip on the draft's wording is not a gate.

What the head does is still worth knowing — it is what a *sixth* phrasing, the one
nobody has typed yet, will get — so it is reported as ``generalization`` and does
not gate. Rejecting a candidate on it is dominated: a skill whose ``examples``
cover the cluster and whose description is weak still takes those commands off
Gemini and leaves the unseen paraphrases exactly where they already are, while
refusing it leaves the whole cluster on Gemini. What keeps a candidate honest is
checks 2 and 3, not this number.

**It also demanded that the candidate's plan name the same actions the teacher's
did.** That is what ``agreement`` reports now, and it no longer gates either.
Three measurements, all on live data, in the order they were taken (JEB-1547,
``scripts/probe_miner_match.py``):

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

Three numbers, then, and only ``match_rate`` gates. The other two are questions
about the cluster and the draft — will an unlisted phrasing route, and did the
teacher answer the same way twice — and both are read out of the log line in
:func:`backend.miner.run._propose`.

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
candidate is not for, so a candidate that wins one of them is drafted too wide
and is not proposed.

Not *any* of them, though, and that distinction is JEB-1579. A cluster that gets
a draft on this run copies its own phrases into its own ``examples`` word for
word, so the moment it is accepted step 0 routes them to it at 1.0 and no head
score can take them away. A claim on those phrases is therefore temporary by
construction — it lasts until the neighbour is mined — and treating it as fatal
is what made two near-synonymous clusters kill each other. Measured live, three
clusters × three runs on the ``multilingual`` checkpoint
(``scripts/probe_miner_match.py``): "фокус" took "покажи сальто" @0.78, "сальто"
took "сделай фокус" @0.98, both were refused, both stayed in the pool —
``mined=1`` is set only on publication (:func:`backend.miner.run._save`) — and
the next run redrew the same two drafts and refused them again. Six refusals in
nine, all of them this, and neither intent could ever be learned while the other
one existed.

So ``outsiders`` is **not** the rest of the pool: it is the pool commands that
*nothing is going to claim*, and those are the only ones that veto. Re-measured
the same way, three clusters × three runs with three one-off leftovers in the
pool: publishable went from 1 of 3 to **8 of 9**, all six neighbour claims were
reported and none vetoed ("фокус" took "покажи сальто" @0.78 in all three runs,
"сальто" took "сделай фокус" @0.95…0.99), and the check kept its teeth — one of
the nine was refused for taking "спой песню", a leftover nothing will claim,
@0.98. An earlier pass of the same probe read 7 of 9 with two such refusals
(@0.78 and @0.99); the difference between the two is Gemini's draft wording,
which is exactly the spread JEB-1562 measured, and not the rule.

**Which commands those are is decided in one place, and it is not here**
(JEB-1579 review). "Will anything claim this?" is exactly "will this cluster be
drafted on this run?", and that question is already answered by
:func:`backend.miner.run._worth_drafting` — size is only its first term. A
cluster whose case set the user has already rejected, and a cluster that has
spent its draft budget (:mod:`backend.miner.attempts`), are both skipped before
the generator is called and stay ``mined=0`` for good, so their phrases never
reach anyone's ``examples`` and a claim on them is permanent. Deciding it here
off ``len(group)`` alone was a second, weaker copy of that predicate, and the
two disagreed on exactly the clusters already known to be unlearnable — which
re-opened JEB-1548 for them. The module that knows both terms builds the list;
this one takes it and asks nothing about how it was built. Same rule, and the
same reason, as :func:`backend.miner.case.is_mineable` refusing to restate what
``_parse`` decides.

One case is deliberately *not* in the control set: a cluster that **is** drafted
this run and then fails its own backtest. Whether it fails is not knowable when
the candidate before it is judged, and a draft that fails once is still on its
way to a skill — it only becomes permanent once its refusals reach the budget,
and then it is stuck, and then it *is* a control. The gap closes itself.

The trade is deliberate and asymmetric: a candidate accepted over a neighbour's
phrase mis-routes that phrase for one mining run, against an 8-primitive library
where every wrong answer is still a jump or a sentence, and the user can 👎 it
into :mod:`backend.feedback`'s auto-disable — while a candidate refused over it
is refused for ever and its whole intent stays on Gemini, which is the number
the project is measured on. It also costs less than it did: a vetoed candidate
now pays one forward pass per *unclaimed* row instead of per unmined row.

What stops the two clusters becoming one skill each for one intent — the other
reading of the same symptom, and the other half of what JEB-1579 asked — is
nothing here, on purpose. Merging them belongs to the grouper
(:mod:`backend.miner.cluster`), which is one Gemini call at 0.875 purity; a gate
that merges on a routing collision would be a second, worse grouper built out of
head scores, and it would still deadlock on the next pair. Two skills for two
phrasings of one intent costs one extra option in the router's ``choice`` list
and is visible to the user on the proposal card; never learning either costs the
metric.

Which router each check calls follows from what it is asking, and they do not all
ask the same thing. Check 1 asks what production will answer, so it calls
:func:`backend.brain.router.route` with the lookup on and reads the plan it
builds. ``generalization`` asks what the head alone would answer, so it calls
:func:`backend.brain.router.pick_skill`, which has no lookup — and one forward
pass instead of two, because the winner is all it needs. Checks 2 and 3 only ever
read *which skill* won, so they call ``pick_skill`` for the same reason.

They run cheapest-first, and check 3 runs last: see :func:`backtest`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..brain.engine import DecisionEngine
from ..brain.router import RouterHit, example_index, normalize, pick_skill, route
from ..brain.skill import Skill
from .case import Case

#: Measured on the live checkpoint, three clusters × three runs (JEB-1562,
#: ``scripts/probe_miner_match.py``): ``match_rate`` came out 1.00 in 9 runs of 9,
#: because the generator copies the cluster into ``examples`` word for word. So on
#: live data this number rejects nothing and every value up to 1.0 publishes the
#: same three clusters — what it is for is the draft that does not even claim its
#: own cluster, and 0.8 tolerates exactly one phrase in five that the draft failed
#: to list *and* the head then missed. It is deliberately left where it was: the
#: same 0.8 was unreachable while it meant the head's answer instead (0.40…1.00
#: across runs on one cluster), and moving the number would read as the fix when
#: the fix is what it measures.
DEFAULT_MIN_MATCH = 0.8


def min_match_rate() -> float:
    return float(os.environ.get("MINER_MIN_MATCH", DEFAULT_MIN_MATCH))


@dataclass(frozen=True)
class Backtest:
    #: Share of the cluster the live router — step 0 included — sends to the
    #: candidate above its threshold. This is the gate, and it is what the user
    #: is promised when the proposal card says a share.
    match_rate: float
    matched: int
    total: int
    #: Share of the cluster the ``choice`` head reaches on its own, with no
    #: ``examples`` to read: what a phrasing the draft never listed would get.
    #: Reported, never gated — see the module docstring.
    generalization: float = 0.0
    #: Share of the cluster whose *teacher* plan the candidate also reproduces,
    #: action names only. Reported, never gated — see the module docstring. It
    #: says how much the teacher varied across the cluster, which is a fact about
    #: the raw material rather than about this candidate.
    agreement: float = 0.0
    #: ``None`` when every active skill still routes to itself; otherwise the
    #: phrase that moved and where it went, so the log says what broke.
    regression: str | None = None
    #: ``None`` when the candidate left the unclaimed pool commands alone;
    #: otherwise the one it took and the confidence it took it at. A claim on a
    #: phrase a neighbour is about to list is not this — see the module docstring.
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
    """First unclaimed pool command the candidate takes, if any.

    ``outsiders`` is already the control set — the pool commands no cluster is
    going to claim (see the module docstring, and
    :func:`backend.miner.run._worth_drafting`, which decides it). This function
    does not filter it further: a second opinion about what belongs here is the
    defect JEB-1579's review found.

    The only loop in this module the pool's size decides — and the pool has no
    ``LIMIT`` — so :func:`backtest` calls it last and only when nothing else has
    already rejected the candidate.
    """
    for case in outsiders:
        picked, confidence = pick_skill(engine, trial, case.user_text)
        if picked is not None and picked.id == candidate.id:
            return f'"{case.user_text}" (nothing will claim it) -> {candidate.id} @ {confidence:.2f}'
    return None


def backtest(
    engine: DecisionEngine,
    active: list[Skill],
    candidate: Skill,
    cases: list[Case],
    outsiders: list[Case] | None = None,
) -> Backtest:
    """Run the candidate against its own cluster, the library and the rest of the pool.

    ``outsiders`` is check 3's control set, already chosen by the caller: the
    pool commands nothing is going to claim, never the whole pool. See the module
    docstring for why, and :func:`backend.miner.run._worth_drafting` for the
    predicate that picks them.

    The candidate is appended, never inserted: that is where ``load_skills``
    puts a mined skill (``ORDER BY rowid``), so the trial registry has to have
    the same option order the live one will have after acceptance.

    **Cheapest check first, and the unbounded one last.** Check 1 plus
    ``generalization`` cost one forward pass per cluster case — step 0 answers the
    routing for free and a mined skill has no ``questions``, so pass 2 is free too
    — and check 2 one per active skill; both small and both fixed. Check 3 costs
    one per *unclaimed row in the pool*, and the pool does not
    shrink for a candidate that fails, so a run that rejects everything would pay
    it over and over. A candidate that has already lost on ``match_rate`` or on a
    regression cannot be published whatever check 3 says, so it is not run
    (JEB-1548 review): ``overreach`` stays ``None``, which reads the same as "it
    took nothing" and is why ``publishable`` still has to test all three.

    Since ``match_rate`` became a measure of production rather than of the head
    (JEB-1562) it is no longer what rejects most drafts — a draft that lists its
    cluster covers it — so check 3 now runs for nearly every candidate instead of
    almost none. What keeps that affordable is JEB-1579 narrowing its control set
    to the unclaimed rows: the clusters being drafted alongside this one, which
    are most of the pool whenever the miner has anything to do, are neither
    scanned nor charged for. It stays last so a regression still stops it.
    """
    trial = [*active, candidate]
    # The phrases step 0 will answer, which costs no forward pass to know. `route`
    # runs the same lookup internally but does not report whether it used it, and
    # that is exactly what decides whether `generalization` needs its own pass.
    listed = set(example_index(trial))

    matched = 0
    generalized = 0
    agreed = 0
    for case in cases:
        # Check 1: exactly the call `/api/chat` makes, lookup included.
        outcome = route(engine, trial, case.user_text, case.state)
        # `route` already applies the skill's threshold, so a RouterHit *is* a
        # confident pick — there is no second comparison to make here.
        covers = isinstance(outcome, RouterHit) and outcome.skill_id == candidate.id
        if covers:
            matched += 1
            if _does_what_the_teacher_did(outcome, case):
                agreed += 1

        if normalize(case.user_text) in listed:
            # Step 0 answered this one, so `route` never asked the head — and the
            # head is the whole question `generalization` asks.
            generalized += int(_head_reaches(engine, trial, candidate, case.user_text))
        else:
            # Nothing to look up, so the route above already *is* the head's answer.
            generalized += int(covers)

    total = len(cases)
    rate = matched / total if total else 0.0
    regression = check_regressions(engine, trial, active)

    overreach = None
    if regression is None and rate >= min_match_rate():
        overreach = check_overreach(engine, trial, candidate, outsiders or [])

    return Backtest(
        match_rate=rate,
        matched=matched,
        total=total,
        generalization=generalized / total if total else 0.0,
        # Over the whole cluster, not over the routed cases: a case the router
        # never reaches cannot agree with anything, and dividing by a shrinking
        # denominator would make a candidate that covers one case out of five
        # look like perfect agreement.
        agreement=agreed / total if total else 0.0,
        regression=regression,
        overreach=overreach,
    )


def _head_reaches(
    engine: DecisionEngine, trial: list[Skill], candidate: Skill, text: str
) -> bool:
    """Would the ``choice`` head land on the candidate with no ``examples`` to read?"""
    picked, _ = pick_skill(engine, trial, text)
    return picked is not None and picked.id == candidate.id


def _does_what_the_teacher_did(outcome: RouterHit, case: Case) -> bool:
    """Action names only: order and arguments are compared nowhere, because the
    teacher never says the same sentence twice."""
    return {step["action"] for step in outcome.raw_plan} == case.action_names
