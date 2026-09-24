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
makes another draft worth paying for.

``MINER_MAX_ATTEMPTS`` (default 3) is what the budget is, and redraws are worth
paying for because the draft's wording is the thing that varies between two runs
over an identical case set — JEB-1562 measured that spread on the live
checkpoint, 0.40 / 0.80 / 0.40 / 1.00 / 0.80 / 0.60 for one cluster over six
runs. Three is measured rather than picked: on the live probe the "сальто"
cluster was refused twice for taking "спой песню" (@0.99, then @0.78) and
published on the third draft, so a budget of 3 was exactly enough for it and 1
would have lost it. It is a floor on patience, not a proof — a cluster that needs
a fourth wording waits for a new case instead, which is the cheaper way to buy
one. Past the budget the cluster is *stuck*: it is skipped
before the generator is called, it costs nothing per run, and it is counted in
``GET /api/metrics`` as ``clusters_stuck`` so "this one is not ripening, it is
failing" is something you can see without opening the database.

Nothing here ever removes a case from the pool. A stuck cluster is still routed,
still logged, still grouped; the only thing withheld is the paid draft.
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
    return int(os.environ.get("MINER_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS))


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


def forget_gone(conn: sqlite3.Connection, pool_ids: set[int]) -> None:
    """Drop the counters whose cases have left the pool.

    A signature only ever recurs while every one of its cases is still unmined:
    publication and acceptance set ``mined=1`` and the grouper can never hand
    that id back. So a row that is no longer a subset of the pool is dead weight,
    and dropping it also means an accepted-then-rejected cluster (whose cases go
    back to ``mined=0``) starts from a clean budget rather than from whatever the
    old draft spent.
    """
    doomed = [
        row["signature"]
        for row in conn.execute("SELECT signature FROM mining_attempts")
        if not _still_pooled(row["signature"], pool_ids)
    ]
    if not doomed:
        return
    conn.executemany("DELETE FROM mining_attempts WHERE signature = ?", [(s,) for s in doomed])
    conn.commit()


def _still_pooled(signature: str, pool_ids: set[int]) -> bool:
    try:
        return set(json.loads(signature)) <= pool_ids
    except (TypeError, ValueError):
        # Unreadable signature: it can never match a cluster again, so it is gone.
        return False


def stuck_count(conn: sqlite3.Connection) -> int:
    """Case sets that have spent their whole draft budget — the metric."""
    return int(
        conn.execute(
            "SELECT COUNT(*) AS n FROM mining_attempts WHERE attempts >= ?", (max_attempts(),)
        ).fetchone()["n"]
    )
