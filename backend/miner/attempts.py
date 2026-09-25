"""How many times this exact set of cases has been drafted and refused.

A cluster that fails its backtest is left in the pool on purpose — ``mined=1``
is set only on publication (:func:`backend.miner.run._save`) — so more cases can
arrive and make it work. What nobody wrote down until JEB-1579 is the other half
of that: while no case arrives, the *same* cluster is regrouped, redrafted and
refused on every single run, and each of those redraws is a
``models.generate_content`` call the user pays for. Live, three runs of three
clusters, that was six paid drafts for two intents that could not be learned at
all. The failure was invisible too — one ``log.warning`` per run, indistinguishable
from a cluster that is simply still ripening.

So refusals are counted, and the counter is keyed by the **case set**, not by the
cluster: the grouper is a Gemini call and re-groups the same commands slightly
differently run to run, but the ``teacher_log`` ids are stable, and the ids are
what decides whether there is anything new to learn from. A new case joining the
cluster is a new signature with a fresh budget, which is exactly the event that
makes another draft worth paying for — and it retires the old signature, which
:func:`forget` is where that happens.

``MINER_MAX_ATTEMPTS`` (default 3) is what the budget is, and redraws are worth
paying for because the draft's wording is the thing that varies between two runs
over an identical case set — JEB-1562 measured that spread on the live
checkpoint, 0.40 / 0.80 / 0.40 / 1.00 / 0.80 / 0.60 for one cluster over six
runs. Three is measured rather than picked: across the live probe's passes the
"сальто" cluster was refused for taking "спой песню" on two drafts out of three
(@0.99 and @0.78) and on one out of three (@0.98), and the draft that published
was never the first — so a budget of 3 got it through where 1 would have lost
it. It is a floor on patience, not a proof: a cluster that needs a fourth wording
waits for a new case instead, which is the cheaper way to buy one. Past the
budget the cluster is *stuck*: it is skipped
before the generator is called, it costs nothing per run, and it is counted in
``GET /api/metrics`` as ``clusters_stuck`` so "this one is not ripening, it is
failing" is something you can see without opening the database.

Nothing here ever removes a case from the pool. A stuck cluster is still routed,
still logged, still grouped; the only thing withheld is the paid draft.

**A stuck cluster is not a dormant one, and check 3 has to treat it as such.**
Its phrases never reach anyone's ``examples``, so nothing will ever claim them
back through step 0 — which makes them exactly the commands
:func:`backend.miner.backtest.check_overreach` must control against. That
follows from being skipped, not from being small, and is why the control set is
built from :func:`backend.miner.run._worth_drafting` rather than from cluster
size (JEB-1579 review).
"""

from __future__ import annotations

import json
import os
import sqlite3

from ..state import iso, utcnow

DEFAULT_MAX_ATTEMPTS = 3

#: A signature is the cluster's ``teacher_log`` ids, sorted — the same shape
#: ``skill_proposals.sample_ids`` stores, so the two can be compared by eye.
Signature = tuple[int, ...]


def max_attempts() -> int:
    """At least one, whatever the environment says.

    ``MINER_MAX_ATTEMPTS=0`` would make ``spent >= max_attempts()`` true for every
    cluster on its first run, so mining would switch itself off entirely and say
    so only at ``INFO`` — a whole subsystem disabled by a value that reads like
    "no retries" (JEB-1579 review). The floor makes the worst setting mean "draft
    once, never redraw", which is what that value is trying to say.
    """
    return max(1, int(os.environ.get("MINER_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)))


def _key(signature: Signature) -> str:
    return json.dumps(sorted(signature))


def load(conn: sqlite3.Connection) -> dict[Signature, int]:
    """Every case set that has been refused, and how often."""
    counts: dict[Signature, int] = {}
    for row in conn.execute("SELECT signature, attempts FROM mining_attempts"):
        try:
            counts[tuple(sorted(json.loads(row["signature"])))] = int(row["attempts"])
        except (TypeError, ValueError):
            continue
    return counts


def record(conn: sqlite3.Connection, signature: Signature, reason: str) -> None:
    """Count one refusal of this case set and keep the last reason for it."""
    conn.execute(
        "INSERT INTO mining_attempts (signature, attempts, last_reason, last_at)"
        " VALUES (?, 1, ?, ?)"
        " ON CONFLICT(signature) DO UPDATE SET attempts = attempts + 1,"
        " last_reason = excluded.last_reason, last_at = excluded.last_at",
        (_key(signature), reason, iso(utcnow())),
    )
    conn.commit()


def clear(conn: sqlite3.Connection, signature: Signature) -> None:
    """This case set finally produced a proposal — it is not stuck any more."""
    conn.execute("DELETE FROM mining_attempts WHERE signature = ?", (_key(signature),))
    conn.commit()


def forget(conn: sqlite3.Connection, pool_ids: set[int], live: list[Signature]) -> None:
    """Drop the counters that can no longer describe a cluster.

    Two ways a row dies, and ``clusters_stuck`` is only honest if both are swept.

    **Its cases left the pool.** A signature can only recur while every one of
    its cases is still unmined — publication and acceptance set ``mined=1`` and
    the grouper can never hand that id back. Dropping the row also means an
    accepted-then-rejected cluster (whose cases go back to ``mined=0``) starts
    from a clean budget rather than from whatever the old draft spent.

    **A larger cluster grew past it.** ``live`` is this run's grouping, and a row
    that is a *proper* subset of one of those signatures describes a cluster that
    no longer exists: the new case that arrived is exactly what bought the budget
    back, so the old set can never be grouped again. Without this the stuck
    counter kept the superseded signature — it is still a subset of the pool —
    and every intermediate case set a growing cluster passed through accumulated,
    turning "how many clusters are stuck now" into "how many ever were"
    (JEB-1579 review). Equality is kept on purpose: the same case set regrouped
    is the same cluster, and its budget is meant to survive.
    """
    doomed = [
        row["signature"]
        for row in conn.execute("SELECT signature FROM mining_attempts")
        if _is_dead(row["signature"], pool_ids, live)
    ]
    if not doomed:
        return
    conn.executemany("DELETE FROM mining_attempts WHERE signature = ?", [(s,) for s in doomed])
    conn.commit()


def _is_dead(signature: str, pool_ids: set[int], live: list[Signature]) -> bool:
    try:
        ids = set(json.loads(signature))
    except (TypeError, ValueError):
        # Unreadable signature: it can never match a cluster again, so it is gone.
        return True
    if not ids <= pool_ids:
        return True
    return any(ids < set(group) for group in live)


def stuck_count(conn: sqlite3.Connection) -> int:
    """Case sets that have spent their whole draft budget — the metric.

    Accurate only because :func:`forget` runs first on every mining run: the row
    count answers "how many clusters are stuck" exactly when every row still
    describes a cluster that could be grouped today.
    """
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM mining_attempts WHERE attempts >= ?", (max_attempts(),)
        ).fetchone()["n"]
    )
