"""👍/👎 and what a 👎 costs a skill.

This is the half of the learning loop that runs backwards. The miner adds
skills; nothing until now ever took one away, so a skill that Gemini worded
badly would keep winning the router forever and every command it stole would be
answered wrong — fast, cheaply, and wrong.

The rule, deliberately boring:

* a rating is recounted from the table, never accumulated in a counter, because
  ``/api/feedback`` overwrites an earlier vote on the same interaction;
* a skill is disabled only after ``SKILL_MIN_RATED`` ratings — without that
  floor a single 👎 on a brand-new skill's first use kills it;
* disabling is not deleting. The row stays, ``status`` becomes ``disabled``, and
  ``/api/skills`` keeps showing it with the reason.

Disabling also hands the skill's raw material back: the cases the accepted
proposal was mined from return to ``teacher_log`` unmined, and any *rejected*
proposal covering the same case set is retired so it cannot block the re-mine.
Returning the pool while leaving that block in place is a silent hole — the
cases come back and no proposal ever appears again.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3

from .state import iso, utcnow

log = logging.getLogger(__name__)

DEFAULT_DISLIKE_LIMIT = 0.30
DEFAULT_MIN_RATED = 5

#: Written to ``skills.disabled_reason``. The only automatic reason there is.
DISLIKE_REASON = "dislike_rate"


def dislike_limit() -> float:
    return float(os.environ.get("SKILL_DISLIKE_LIMIT", DEFAULT_DISLIKE_LIMIT))


def min_rated() -> int:
    return int(os.environ.get("SKILL_MIN_RATED", DEFAULT_MIN_RATED))


def record(conn: sqlite3.Connection, *, interaction_id: int, value: int) -> bool:
    """Store one vote and, on a 👎, re-judge the skill behind it.

    Returns ``False`` when no such interaction exists. Re-rating the same
    interaction overwrites the previous value — one row per interaction, always.
    """
    cursor = conn.execute(
        "UPDATE interactions SET feedback = ? WHERE id = ?", (value, interaction_id)
    )
    if cursor.rowcount == 0:
        conn.commit()
        return False
    conn.commit()

    row = conn.execute(
        "SELECT skill_id FROM interactions WHERE id = ?", (interaction_id,)
    ).fetchone()
    skill_id = row["skill_id"] if row else None
    # A vote on a Gemini answer (`skill_id IS NULL`) has nothing to disable — it
    # stays in the metrics and stays the miner's raw material.
    #
    # Every vote triggers the recount, not only a 👎: a 👍 can be the rating that
    # finally lifts the skill over `SKILL_MIN_RATED`, and with 4 dislikes already
    # behind it the verdict is due right then. A 👍 can never disable a skill that
    # was already past the floor — it can only lower the ratio.
    if skill_id:
        review_skill(conn, skill_id)
    return True


def review_skill(conn: sqlite3.Connection, skill_id: str) -> bool:
    """Recount the skill's ratings; disable it if they are bad enough.

    Returns ``True`` when this call turned the skill off.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS rated, SUM(feedback IS -1) AS dislikes"
        " FROM interactions WHERE skill_id = ? AND feedback IS NOT NULL",
        (skill_id,),
    ).fetchone()
    rated = int(row["rated"] or 0)
    dislikes = int(row["dislikes"] or 0)
    if rated < min_rated() or dislikes / rated <= dislike_limit():
        return False
    return disable_skill(conn, skill_id, DISLIKE_REASON)


def disable_skill(conn: sqlite3.Connection, skill_id: str, reason: str) -> bool:
    """Turn a skill off and give its mined cases back to the pool.

    The router needs no cache invalidation: ``/api/chat`` reads the library with
    ``load_skills(only_active=True)`` on every request, under the same
    ``db.lock`` this runs under, so the very next command is routed without the
    disabled skill — it is simply not among the options any more.

    Seed skills are disabled by the same rule as mined ones. A starter skill the
    user keeps disliking is exactly as wrong as a mined one.
    """
    cursor = conn.execute(
        "UPDATE skills SET status = 'disabled', disabled_at = ?, disabled_reason = ?"
        " WHERE id = ? AND status = 'active'",
        (iso(utcnow()), reason, skill_id),
    )
    if cursor.rowcount == 0:
        # Already disabled, or never existed. Nothing to release either way.
        conn.commit()
        return False

    for proposal_id, signature in _accepted_clusters(conn, skill_id):
        conn.execute(
            "UPDATE teacher_log SET mined = 0, cluster_id = NULL WHERE cluster_id = ?",
            (proposal_id,),
        )
        _lift_rejections(conn, signature)
    conn.commit()
    log.info("feedback: skill %r disabled (%s)", skill_id, reason)
    return True


def _accepted_clusters(
    conn: sqlite3.Connection, skill_id: str
) -> list[tuple[str, tuple[int, ...]]]:
    """``(proposal_id, case signature)`` of the proposals this skill came from.

    A seed skill was never proposed, so this is empty and there is nothing to
    return to the pool — which is the correct answer, not a missing case.
    """
    clusters = []
    for row in conn.execute(
        "SELECT id, skill_json, sample_ids FROM skill_proposals WHERE status = 'accepted'"
    ):
        try:
            if json.loads(row["skill_json"])["id"] != skill_id:
                continue
            signature = tuple(sorted(json.loads(row["sample_ids"])))
        except (KeyError, TypeError, ValueError):
            continue
        clusters.append((row["id"], signature))
    return clusters


def _lift_rejections(conn: sqlite3.Connection, signature: tuple[int, ...]) -> None:
    """Retire the rejected proposals that cover exactly this set of cases.

    ``backend/miner/run.py`` reads its "already turned down" signatures off rows
    with ``status = 'rejected'``, so moving the status is what unblocks them. The
    rows are kept as history rather than deleted; any status other than
    ``rejected`` is invisible to that read.
    """
    for row in conn.execute(
        "SELECT id, sample_ids FROM skill_proposals WHERE status = 'rejected'"
    ).fetchall():
        try:
            rejected = tuple(sorted(json.loads(row["sample_ids"])))
        except (TypeError, ValueError):
            continue
        if rejected == signature:
            conn.execute(
                "UPDATE skill_proposals SET status = 'retired' WHERE id = ?", (row["id"],)
            )
