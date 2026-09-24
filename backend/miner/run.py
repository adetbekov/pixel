"""One mining run, start to finish.

Synchronous inside the request that triggers it, and that is a prototype
decision, not an architectural one — which is why the whole run is
:func:`mine_once` and nothing else: moving it to a background worker is a change
of caller, not of this module.

What it costs the request it runs inside, measured on the live checkpoint
(``scripts/calibrate_miner_sim.py``, section 6): a run is roughly one teacher
grouping call plus a few forward passes per cluster case and per active skill,
and it holds ``LayaEngine._lock`` for all of them, so every other request's
router waits. The one term that scales with the *pool* rather than the cluster
is the backtest's over-broad check, and :func:`backend.miner.backtest.backtest`
only reaches it for a candidate no regression has already rejected. Since
``match_rate`` became a measure of what production will do (JEB-1562) that is
nearly every candidate, where it used to be almost none: one pass-1 per unmined
row, per candidate. The pool has no ``LIMIT`` and does not shrink for a candidate
that was rejected, so this is the term to watch as the pool grows.

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
from .backtest import backtest
from .case import Case, is_mineable, load_pool, pool_size
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
        active = load_skills(conn, only_active=True)
        taken = _taken_ids(conn)
        rejected = _rejected_signatures(conn)

    if len(cases) < min_cluster_size():
        return 0

    groups = group_texts(engine, [case.user_text for case in cases], generator.group)
    created = 0
    for group in groups:
        cluster = [cases[index] for index in group]
        # The rest of the pool is the over-broad control set, free of charge:
        # real commands this candidate is not for (see backtest, check 3).
        member = set(group)
        outsiders = [case for index, case in enumerate(cases) if index not in member]
        proposal = _propose(engine, generator, cluster, outsiders, active, taken, rejected)
        if proposal is None:
            continue
        taken.add(proposal.skill.id)
        with db.lock:
            _save(conn, proposal)
        created += 1
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


def _propose(
    engine: DecisionEngine,
    generator: SkillGenerator,
    cluster: list[Case],
    outsiders: list[Case],
    active: list[Skill],
    taken: set[str],
    rejected: set[tuple[int, ...]],
) -> MinedProposal | None:
    """One cluster -> one proposal, or ``None`` and the reason in the log."""
    if len(cluster) < min_cluster_size():
        return None

    signature = tuple(sorted(case.id for case in cluster))
    if signature in rejected:
        # The user already said no to exactly these cases. Re-proposing them on
        # the very next run is how a suggestion panel becomes noise.
        log.info("miner: cluster %s was already rejected", list(signature))
        return None

    skill = generator.propose(cluster, active)
    if skill is None:
        return None
    if skill.id in taken:
        log.warning("miner: candidate id %r is already in use", skill.id)
        return None

    report = backtest(engine, active, skill, cluster, outsiders)
    if report.regression is not None:
        log.warning(
            "miner: %r rejected — it breaks an active skill: %s", skill.id, report.regression
        )
        return None
    if report.overreach is not None:
        log.warning("miner: %r rejected — it is drafted too wide: %s", skill.id, report.overreach)
        return None
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
        return None

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
    return MinedProposal(
        id=str(uuid.uuid4()),
        skill=skill,
        match_rate=round(report.match_rate, 3),
        generalization=round(report.generalization, 3),
        sample_ids=list(signature),
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
