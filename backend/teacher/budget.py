"""How many teacher calls today's bucket has left, and what to say when it has none.

The free tier gives **20 requests a day** per (project, model) for
`models/gemini-2.5-flash-lite`, measured on the stand on 2026-09-24 (JEB-1600)::

    Quota exceeded for metric:
    generativelanguage.googleapis.com/generate_content_free_tier_requests,
    limit: 20, model: gemini-2.5-flash-lite

Twenty is small enough that a stranger who found `https://pixel.yeldos.dev` can
spend the day's learning before the owner types a word (JEB-1623). This module
stops that a couple of calls short of the API's own ceiling, so the outcome is a
robot that says it is done learning for today rather than a `429` the user reads
as Pixel being stupid.

**The window is a bucket, not a rolling 24 hours.** Google resets at midnight
Pacific, i.e. ``07:00Z``, and that is the difference JEB-1600 tripped over:
``teacher_calls_24h`` read ``38`` against a ceiling of ``20``, because a rolling
window spans two buckets. A cap counted the rolling way would refuse calls the
API would have accepted, and the morning after a busy evening would start locked.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from ..state import iso, utcnow
from .client import TeacherResult

#: Two below the free tier's 20, so the nightly live-contract gate (JEB-1552)
#: still has its two calls to make after a day of ordinary use.
DEFAULT_DAILY_CAP = 18

#: The hour the free-tier bucket rolls over, UTC — midnight Pacific.
RESET_HOUR_UTC = 7

CAP_REPLY = "На сегодня я больше не могу учиться — давай продолжим завтра"

#: `sad`, matching the quota outage next door in `client.py`, and for its reason:
#: `sleepy` is what a low-energy robot already wears (`TIRED_PLAN`), and a cap is
#: not tiredness. The machine-readable difference between "we stopped" and
#: "Google stopped us" travels on `teacher_status`, not on the face.
CAP_PLAN: list[dict[str, Any]] = [
    {"action": "set_face", "args": {"face": "sad"}},
    {"action": "say", "args": {"text": CAP_REPLY}},
]

#: Carried to the UI on `Reply` / `HistoryItem`, and the thing that separates a
#: budget we imposed from a `429` Google imposed (`QUOTA_STATUS`) and from a plan
#: that failed to parse (no status at all).
CAP_STATUS = "daily_cap"

#: What `teacher_log.state_json.error` starts with when the cause was this cap.
#: Greppable, and it names the status verbatim so the two channels agree.
CAP_ERROR = f"teacher unavailable: {CAP_STATUS}"


def daily_cap() -> int:
    return int(os.environ.get("TEACHER_DAILY_CAP", DEFAULT_DAILY_CAP))


def bucket_start(now: datetime | None = None) -> datetime:
    """The most recent ``07:00Z``, at or before ``now``."""
    now = now or utcnow()
    start = now.replace(hour=RESET_HOUR_UTC, minute=0, second=0, microsecond=0)
    if start > now:
        start -= timedelta(days=1)
    return start


def calls_used(conn: sqlite3.Connection, now: datetime | None = None) -> int:
    """Teacher calls charged to today's bucket.

    `teacher_log` has no timestamp of its own, so the age of a call comes from
    the interaction it belongs to — the same join `/api/metrics` makes.

    Rows this cap itself wrote are excluded: no request left the process for
    them, so charging them to the bucket would count a refusal as a call and
    drift the number away from what Google is actually counting. A `429` row is
    *not* excluded — that one did spend a unit.
    """
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM teacher_log t"
        " JOIN interactions i ON i.id = t.interaction_id"
        " WHERE i.ts >= ? AND (i.teacher_status IS NULL OR i.teacher_status != ?)",
        (iso(bucket_start(now)), CAP_STATUS),
    ).fetchone()
    return int(row["n"])


def exhausted(conn: sqlite3.Connection, now: datetime | None = None) -> bool:
    """Has today's bucket run out? A cap of ``0`` switches the teacher off."""
    return calls_used(conn, now) >= daily_cap()


def cap_result() -> TeacherResult:
    """What the teacher hands back when it was never asked.

    Shaped exactly like `_quota_exhausted` next door so `/api/chat` has one tail:
    a real plan, a reply the user can act on, `handled=False` so the miner never
    learns a budget stop as a skill, and `error` set so the row stays out of the
    pool.
    """
    return TeacherResult(
        raw_plan=list(CAP_PLAN),
        reply=CAP_REPLY,
        raw_response="",
        error=f"{CAP_ERROR}: reached TEACHER_DAILY_CAP={daily_cap()}",
        handled=False,
        teacher_status=CAP_STATUS,
    )
