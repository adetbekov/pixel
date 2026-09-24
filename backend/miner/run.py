"""One mining run, start to finish.

The whole run is :func:`mine_once` and nothing else, so *who* calls it is the
only thing that decides whether anyone waits for it. The automatic trigger does
not: it goes through :mod:`backend.miner.worker`, off the request path. The
manual ``POST /api/mine`` calls this directly, because it wants the count back.

What a run costs, measured on the live checkpoint
(``scripts/calibrate_miner_sim.py``, section 6): a run is roughly one teacher
grouping call plus a few forward passes per cluster case and per active skill.
What a concurrent chat waits for depends on which part of the run holds the
engine lock, and the two parts differ by 4x (JEB-1599,
``scripts/bench_router_under_mining.py``, live checkpoint, window 40):

* the **backtest** takes ``LayaEngine._lock`` once per forward pass and holds
  it 260 ms at p50, so a chat that arrives here waits one pass, not the series;
* the **grouping step**, when the teacher answers ``[]`` and
  :func:`backend.miner.cluster.group_texts` falls back to local vectors, is one
  ``engine.embed`` over the whole window under a single acquisition — held
  1092 ms at p50 (572 ms at window 20). A chat that lands inside that waits all
  of it, and the wait grows with ``MINER_POOL_WINDOW`` linearly.

So the window is a latency knob on the fallback path and only a run-length knob
on the teacher-grouped one.

The one term that scales with the *pool* rather than the cluster is the
backtest's over-broad check, and :func:`backend.miner.backtest.backtest`
only reaches it for a candidate no regression has already rejected. Since
``match_rate`` became a measure of what production will do (JEB-1562) that is
nearly every candidate, where it used to be almost none — but it no longer scans
the whole pool: JEB-1579 narrowed its control set to the rows nothing is going to
claim, and on a pool the miner has work to do in, most rows belong to a cluster
that is being drafted right now. The pool still does not shrink for a candidate
that was rejected — what bounds it is ``MINER_POOL_WINDOW``: a run drafts from
the newest N mineable cases, so the unclaimed rows an older pool keeps
accumulating no longer make every run slower than the last.

Which rows those are is :func:`_worth_drafting`, and there is deliberately no
second opinion about it: a cluster gets a draft, or nothing will ever list its
phrases. Both halves of the answer are computed once per run, before the loop.

The other thing that does not shrink is the *Gemini* bill of a cluster that
keeps failing, and that one is now bounded: a case set that has been drafted and
refused ``MINER_MAX_ATTEMPTS`` times is skipped before ``generator.propose`` and
counted as stuck (:mod:`backend.miner.attempts`), until a new case joins it —
which makes it a different cluster and retires the old signature.

Two triggers, one body: every ``MINER_BATCH``-th new mineable case, and
``POST /api/mine``. A second concurrent run is refused rather than queued — it
would re-read the same pool and race the first one to the same proposals.

Locking. Gemini calls and routing happen with ``db.lock`` released: the reads
come first, the writes come last, and nothing network-shaped happens in between
while holding a non-reentrant lock.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass

from .. import db
from ..brain.engine import DecisionEngine, get_engine
from ..brain.skill import Skill, load_skills
from ..state import iso, utcnow
from . import attempts
from .backtest import backtest
from .case import Case, is_mineable, load_pool, pool_ids, pool_size
from .cluster import group_texts, min_cluster_size
from .generate import SkillGenerator, get_generator

log = logging.getLogger(__name__)

DEFAULT_BATCH = 5

_lock = threading.Lock()


@dataclass(frozen=True)
class MineResult:
    started: bool
    proposals: int


def batch_size() -> int:
    return int(os.environ.get("MINER_BATCH", DEFAULT_BATCH))


def mining_due(conn: sqlite3.Connection, case_id: int) -> bool:
    """True when the case that just arrived is the ``MINER_BATCH``-th mineable one.

    ``case_id`` is the ``teacher_log`` row this request wrote, and asking about
    it first is what makes the remainder below safe. The remainder alone is a
    *level*, not an event: it stays true for as long as the pool stays put, and
    the pool stays put on everything the miner cannot use. A failed call and a
    declined answer are kept in ``teacher_log`` and never enter the pool
    (:func:`backend.miner.case._parse`), and ``mined = 1`` is set only in
    :func:`_save`, so a cluster that fails its backtest stays unmined for good —
    exactly the "фокус" cluster at ``match_rate`` 0.60 that JEB-1547 measured.
    Park the pool on a multiple of ``MINER_BATCH`` that way and every later
    "какая погода" would re-fire the trigger, putting a synchronous ``group_texts``
    plus one ``generator.propose`` round trip inside a ``POST /api/chat`` the user
    is waiting on — an unbounded Gemini bill, one call per declined message.
    Gating on the arrival turns it back into an edge: no new mineable case, no run.

    Counted, not accumulated: a run that mines nothing leaves the pool where it
    was, so the next mineable case brings the count to the following multiple and
    the trigger fires again instead of going quiet forever.
    """
    if not is_mineable(conn, case_id):
        return False
    size = pool_size(conn)
    return size > 0 and size % batch_size() == 0


def mine_once() -> MineResult:
    """Mine the pool once. ``started=False`` means another run holds the lock."""
    if not _lock.acquire(blocking=False):
        log.info("miner: a run is already in progress")
        return MineResult(started=False, proposals=0)
    try:
        return MineResult(started=True, proposals=_mine())
    finally:
        _lock.release()


def _mine() -> int:
    generator = get_generator()
    if generator is None:
        # No API key: nothing writes `teacher_log` either, so this is the
        # ordinary key-less configuration and not an error.
        return 0
    try:
        engine = get_engine()
    except RuntimeError:
        log.warning("miner: no decision engine — a candidate could not be backtested")
        return 0

    conn = db.get_conn()
    with db.lock:
        cases = load_pool(conn)
        # The ledger is keyed by case ids and swept against the pool, not against
        # the window this run drafts from: a case the window left behind is still
        # unmined, and its cluster's spent budget has to survive with it.
        mineable = pool_ids(conn)
        active = load_skills(conn, only_active=True)
        taken = _taken_ids(conn)
        rejected = _rejected_signatures(conn)

    if len(cases) < min_cluster_size():
        return 0

    groups = group_texts(engine, [case.user_text for case in cases], generator.group)
    clusters = [[cases[index] for index in group] for group in groups]
    signatures = [tuple(sorted(case.id for case in cluster)) for cluster in clusters]

    # The ledger is pruned against *this run's* grouping, not just against the
    # pool, so a signature a larger cluster has grown past stops being counted as
    # stuck (JEB-1579 review). Read after the prune, so `refused` is what the
    # ledger says now.
    with db.lock:
        attempts.forget(conn, mineable, signatures)
        refused = attempts.load(conn)

    # Whether a cluster gets a draft this run is also the answer to "will anything
    # ever claim its phrases", so it is computed once, up front, and the same list
    # decides both what is proposed and what check 3 controls against. Two
    # predicates for one question is what JEB-1579's review found: `leftovers`
    # used to re-derive this from `len(group)` alone and disagreed on the clusters
    # already known to be unlearnable.
    draftable = [
        _worth_drafting(cluster, signature, rejected, refused)
        for cluster, signature in zip(clusters, signatures, strict=True)
    ]

    created = 0
    for position, cluster in enumerate(clusters):
        if not draftable[position]:
            continue
        # Check 3's control set, free of charge: the pool commands no cluster is
        # going to claim. A cluster that *is* being drafted lists its own phrases
        # in its own `examples`, so step 0 takes them back on acceptance and a
        # claim on them lasts one run; one that is not drafted — too small, user-
        # rejected, or out of draft budget — keeps `mined=0` for good.
        outsiders = [
            case
            for number, other in enumerate(clusters)
            if number != position and not draftable[number]
            for case in other
        ]
        signature = signatures[position]
        attempt = _propose(engine, generator, cluster, outsiders, active, taken, signature)
        if attempt.proposal is not None:
            taken.add(attempt.proposal.skill.id)
            with db.lock:
                _save(conn, attempt.proposal)
                attempts.clear(conn, signature)
            created += 1
        elif attempt.reason is not None:
            # A draft was paid for and refused. Counted so the next run can stop
            # paying for the same one; the write is here rather than in `_propose`
            # to keep every network call outside `db.lock`.
            with db.lock:
                attempts.record(conn, signature, attempt.reason)
    return created


@dataclass(frozen=True)
class MinedProposal:
    id: str
    skill: Skill
    match_rate: float
    #: Stored for the card, not for the gate: `match_rate` is ~1.00 whenever the
    #: draft lists its own cluster, so this is the number of the two that actually
    #: varies between drafts, and the user decides on the card (JEB-1581).
    generalization: float
    sample_ids: list[int]


@dataclass(frozen=True)
class Attempt:
    """What one drafted cluster produced, and whether it is worth remembering.

    ``proposal`` is a published draft; ``reason`` is a draft that was generated,
    backtested and refused, which is what :mod:`backend.miner.attempts` counts;
    neither is a draft that never arrived, which says nothing about the cluster.
    """

    proposal: MinedProposal | None = None
    reason: str | None = None


def _worth_drafting(
    cluster: list[Case],
    signature: tuple[int, ...],
    rejected: set[tuple[int, ...]],
    refused: dict[tuple[int, ...], int],
) -> bool:
    """Everything that can be decided about a cluster before paying for a draft.

    The two "no" answers below are about the same thing — this exact set of cases
    has been through the mill already — and both are keyed on the case ids rather
    than on the cluster, because the grouper is a Gemini call and does not hand
    back the same grouping twice.

    **This is also check 3's control-set predicate, and deliberately the only
    copy of it** (JEB-1579 review). ``False`` here means the cluster gets no draft
    this run, which means its phrases reach no ``examples``, which means a
    candidate that wins one of them keeps it: exactly the permanent claim
    :func:`backend.miner.backtest.check_overreach` exists to refuse. Size is only
    the first term — a user-rejected case set never comes back
    (:func:`_rejected_signatures`) and a stuck one waits for a case that may never
    arrive, so reading size alone called both of them claimable and re-opened
    JEB-1548 for the two kinds of cluster already known to be unlearnable.
    """
    if len(cluster) < min_cluster_size():
        return False

    if signature in rejected:
        # The user already said no to exactly these cases. Re-proposing them on
        # the very next run is how a suggestion panel becomes noise.
        log.info("miner: cluster %s was already rejected", list(signature))
        return False

    spent = refused.get(signature, 0)
    if spent >= attempts.max_attempts():
        # Stuck: this exact case set has had its draft budget and no new case has
        # joined it since, so the only thing another run could change is Gemini's
        # wording. Skipped *before* `generator.propose`, which is the whole point
        # — the bill this removes is one generate call per run, for ever.
        log.info(
            "miner: cluster %s is stuck — %d drafts refused and no new case since; not redrawing",
            list(signature),
            spent,
        )
        return False
    return True


def _propose(
    engine: DecisionEngine,
    generator: SkillGenerator,
    cluster: list[Case],
    outsiders: list[list[Case]],
    active: list[Skill],
    taken: set[str],
    signature: tuple[int, ...],
) -> Attempt:
    """One drafted cluster -> one proposal, or the reason it did not become one."""
    skill = generator.propose(cluster, active)
    if skill is None:
        # Not counted against the budget. A draft that never arrived is a verdict
        # on Gemini — a dead key, an exhausted quota, three malformed answers —
        # and not on this case set, and a failed call bills no tokens either.
        # `generator.propose` has its own bounded retry inside the one call.
        return Attempt()
    if skill.id in taken:
        log.warning("miner: candidate id %r is already in use", skill.id)
        return Attempt(reason=f"candidate id {skill.id!r} is already in use")

    report = backtest(engine, active, skill, cluster, outsiders)
    if report.regression is not None:
        log.warning(
            "miner: %r rejected — it breaks an active skill: %s", skill.id, report.regression
        )
        return Attempt(reason=f"breaks an active skill: {report.regression}")
    if report.overreach is not None:
        log.warning("miner: %r rejected — it is drafted too wide: %s", skill.id, report.overreach)
        return Attempt(reason=f"drafted too wide: {report.overreach}")
    if not report.publishable:
        log.info(
            "miner: %r rejected — match_rate %.2f (generalization %.2f, agreement %.2f)"
            " on %d cases",
            skill.id,
            report.match_rate,
            report.generalization,
            report.agreement,
            report.total,
        )
        return Attempt(
            reason=(
                f"match_rate {report.match_rate:.2f} on {report.total} cases"
                f" (generalization {report.generalization:.2f},"
                f" agreement {report.agreement:.2f})"
            )
        )

    # `generalization` well under `match_rate` means the cluster is covered but a
    # *sixth* phrasing will still cost a Gemini call; it is stored below and shown
    # on the card, so the log line is no longer the only place it is read
    # (JEB-1581). `agreement` still is: a low one means the teacher improvised
    # differently every time, which is a fact about the raw material and would
    # read as a verdict on the skill next to "принять?" (JEB-1547, JEB-1562).
    log.info(
        "miner: %r proposed — match_rate %.2f (generalization %.2f, agreement %.2f) on %d cases",
        skill.id,
        report.match_rate,
        report.generalization,
        report.agreement,
        report.total,
    )
    return Attempt(
        proposal=MinedProposal(
            id=str(uuid.uuid4()),
            skill=skill,
            match_rate=round(report.match_rate, 3),
            generalization=round(report.generalization, 3),
            sample_ids=list(signature),
        )
    )


def _taken_ids(conn: sqlite3.Connection) -> set[str]:
    """Skill ids the candidate may not reuse — library plus pending proposals.

    A **disabled** skill does not hold its id. Stage 5 turns a badly-rated skill
    off and hands its cases back to the pool exactly so the miner can try again,
    and the generator is only ever shown the *active* library — so the next draft
    for the same cluster of phrases picks the same obvious id. Counting that id
    as taken would drop every retry into a log line: quietly, every run, forever.
    `accept_proposal` overwrites the disabled row when the retry is accepted.
    """
    # `IS NOT`, not `!=`: a row with a NULL status must read as taken, not free.
    ids = {row["id"] for row in conn.execute("SELECT id FROM skills WHERE status IS NOT 'disabled'")}
    for row in conn.execute("SELECT skill_json FROM skill_proposals WHERE status = 'pending'"):
        try:
            ids.add(json.loads(row["skill_json"])["id"])
        except (KeyError, TypeError, ValueError):
            continue
    return ids


def _rejected_signatures(conn: sqlite3.Connection) -> set[tuple[int, ...]]:
    """The case sets the user has already turned down.

    The signature is read back off the rejected proposal rather than stored in a
    column of its own: ``sample_ids`` already *is* the set of cases, so a second
    copy could only ever disagree with it.
    """
    signatures = set()
    for row in conn.execute("SELECT sample_ids FROM skill_proposals WHERE status = 'rejected'"):
        try:
            signatures.add(tuple(sorted(json.loads(row["sample_ids"]))))
        except (TypeError, ValueError):
            continue
    return signatures


def _save(conn: sqlite3.Connection, proposal: MinedProposal) -> None:
    """Write the proposal and take its cases out of the pool — in that order.

    ``mined=1`` is set only here, so a cluster that failed generation or the
    backtest stays available: more cases may arrive and make it work.
    """
    conn.execute(
        "INSERT INTO skill_proposals"
        " (id, skill_json, match_rate, generalization, sample_ids, status, created_at)"
        " VALUES (?, ?, ?, ?, ?, 'pending', ?)",
        (
            proposal.id,
            proposal.skill.model_dump_json(),
            proposal.match_rate,
            proposal.generalization,
            json.dumps(proposal.sample_ids),
            iso(utcnow()),
        ),
    )
    conn.executemany(
        "UPDATE teacher_log SET mined = 1, cluster_id = ? WHERE id = ?",
        [(proposal.id, case_id) for case_id in proposal.sample_ids],
    )
    conn.commit()
