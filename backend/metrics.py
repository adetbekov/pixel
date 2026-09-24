"""The numbers that say whether Pixel is learning.

All of it is queries over ``interactions``, ``skills`` and — for the one number
about the miner rather than about the router — ``mining_attempts``. There is no
metrics table, because a second copy of a count can only ever disagree with the
rows it was derived from.

Two rules the SQL encodes:

* **Button clicks are not commands.** ``engine='button'`` never enters
  ``laya_share``: the buttons have no router and no teacher, so counting them
  would inflate the headline metric with clicks and hide the real trend.
* **Every ratio is guarded.** An empty database answers ``0.0``, not a
  ``ZeroDivisionError`` and not ``NaN%`` in the panel.

``laya_share`` over all time is the project's headline; the same share over the
last 24h is what shows the trend, since a long tail of early Gemini calls hides
the improvement in the lifetime number for a long while.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Any

from .miner.attempts import stuck_count
from .state import iso, utcnow

#: The engines a user *command* can be answered by. Buttons are excluded on
#: purpose — see the module docstring.
ROUTED_ENGINES = ("laya", "gemini")

WINDOW = timedelta(hours=24)


def _share(laya: int, gemini: int) -> float:
    routed = laya + gemini
    return round(laya / routed, 3) if routed else 0.0


def collect(conn: sqlite3.Connection) -> dict[str, Any]:
    """Every field of ``GET /api/metrics``, in one pass over the tables."""
    since = iso(utcnow() - WINDOW)

    by_engine = {
        row["engine"]: row
        for row in conn.execute(
            "SELECT engine, COUNT(*) AS n, AVG(latency_ms) AS avg_ms"
            " FROM interactions GROUP BY engine"
        )
    }
    recent = {
        row["engine"]: row["n"]
        for row in conn.execute(
            # `iso()` always renders UTC with a fixed date-time prefix, so string
            # comparison orders the instants — the optional `.ffffff` only ever
            # appears after the part that differs.
            "SELECT engine, COUNT(*) AS n FROM interactions WHERE ts >= ? GROUP BY engine",
            (since,),
        )
    }
    skills = conn.execute(
        "SELECT SUM(status = 'active') AS active, SUM(status = 'disabled') AS disabled FROM skills"
    ).fetchone()
    # `teacher_log` has no timestamp of its own, so the age of a teacher call
    # comes from the interaction it belongs to.
    teacher_calls = conn.execute(
        "SELECT COUNT(*) AS n FROM teacher_log t"
        " JOIN interactions i ON i.id = t.interaction_id"
        " WHERE i.ts >= ?",
        (since,),
    ).fetchone()["n"]

    laya = by_engine.get("laya")
    gemini = by_engine.get("gemini")
    laya_n = laya["n"] if laya else 0
    gemini_n = gemini["n"] if gemini else 0

    return {
        "laya_share": _share(laya_n, gemini_n),
        "laya_share_24h": _share(recent.get("laya", 0), recent.get("gemini", 0)),
        "avg_latency_laya_ms": round(laya["avg_ms"], 1) if laya else 0.0,
        "avg_latency_gemini_ms": round(gemini["avg_ms"], 1) if gemini else 0.0,
        "skills_active": int(skills["active"] or 0),
        "skills_disabled": int(skills["disabled"] or 0),
        "total_commands": laya_n + gemini_n,
        "gemini_calls_24h": int(recent.get("gemini", 0)),
        "teacher_calls_24h": int(teacher_calls),
        # Clusters the miner has given up redrawing (JEB-1579). Zero is the
        # normal reading; anything else is "the pool has a pattern in it that the
        # backtest will not pass", which used to live only in a log line.
        "clusters_stuck": stuck_count(conn),
    }
