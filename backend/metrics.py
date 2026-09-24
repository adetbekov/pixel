"""The numbers that say whether Pixel is learning.

All of it is queries over ``interactions`` and ``skills`` — there is no metrics
table, because a second copy of a count can only ever disagree with the rows it
was derived from.

Two rules the SQL encodes:

* **Button clicks are not commands.** ``engine='button'`` never enters
  ``laya_share``: the buttons have no router and no teacher, so counting them
  would inflate the headline metric with clicks and hide the real trend.
* **Every ratio is guarded.** An empty database answers ``0.0``, not a
  ``ZeroDivisionError`` and not ``NaN%`` in the panel.

``laya_share`` over all time is the project's headline; the same share over the
last 24h is what shows the trend, since a long tail of early Gemini calls hides
the improvement in the lifetime number for a long while.

The averages are windowed, not lifetime (JEB-1574). A lifetime average is the
one number history can poison permanently: a batch of rows logged under the old
teacher timeout (~100 s each) held ``avg_latency_gemini_ms`` at 98 s while live
calls answered in 1.6 s, and a single fresh call moved it by 10 s. Latency is
"how fast is the robot *now*", so it reads over the same 24h as the counts
beside it.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from typing import Any

from .state import iso, utcnow

#: The engines a user *command* can be answered by. Buttons are excluded on
#: purpose — see the module docstring.
ROUTED_ENGINES = ("laya", "gemini")

WINDOW = timedelta(hours=24)


def _share(laya: int, gemini: int) -> float:
    routed = laya + gemini
    return round(laya / routed, 3) if routed else 0.0


def _recent_n(recent: dict[str, Any], engine: str) -> int:
    row = recent.get(engine)
    return int(row["n"]) if row else 0


def _avg_latency(recent: dict[str, Any], engine: str) -> float:
    """Average latency inside the window, or ``0.0`` when nothing is there to average.

    ``latency_ms`` is nullable, so ``AVG`` over rows that all miss it is ``NULL``.
    """
    row = recent.get(engine)
    return round(row["avg_ms"], 1) if row and row["avg_ms"] is not None else 0.0


def collect(conn: sqlite3.Connection) -> dict[str, Any]:
    """Every field of ``GET /api/metrics``, in one pass over the tables."""
    since = iso(utcnow() - WINDOW)

    totals = {
        row["engine"]: row["n"]
        for row in conn.execute("SELECT engine, COUNT(*) AS n FROM interactions GROUP BY engine")
    }
    recent = {
        row["engine"]: row
        for row in conn.execute(
            # `iso()` always renders UTC with a fixed date-time prefix, so string
            # comparison orders the instants — the optional `.ffffff` only ever
            # appears after the part that differs.
            "SELECT engine, COUNT(*) AS n, AVG(latency_ms) AS avg_ms"
            " FROM interactions WHERE ts >= ? GROUP BY engine",
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

    laya_n = totals.get("laya", 0)
    gemini_n = totals.get("gemini", 0)

    return {
        "laya_share": _share(laya_n, gemini_n),
        "laya_share_24h": _share(_recent_n(recent, "laya"), _recent_n(recent, "gemini")),
        "avg_latency_laya_ms": _avg_latency(recent, "laya"),
        "avg_latency_gemini_ms": _avg_latency(recent, "gemini"),
        "skills_active": int(skills["active"] or 0),
        "skills_disabled": int(skills["disabled"] or 0),
        "total_commands": laya_n + gemini_n,
        "gemini_calls_24h": _recent_n(recent, "gemini"),
        "teacher_calls_24h": int(teacher_calls),
    }
