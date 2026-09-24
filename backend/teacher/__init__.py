"""The teacher: Gemini answers the commands Laya missed.

Laya always goes first. Only a :class:`~backend.brain.router.RouterMiss` reaches
this package, and what comes back is not code — it is a plan built from
:mod:`backend.actions` and nothing else. Every call is written to ``teacher_log``,
which is the only input stage 4's skill miner has.
"""

from .client import (
    FALLBACK_PLAN,
    QUOTA_PLAN,
    QUOTA_REPLY,
    QUOTA_STATUS,
    GeminiTeacher,
    Teacher,
    TeacherResult,
    build_teacher,
    get_teacher,
    set_teacher,
)
from .log import log_case

__all__ = [
    "FALLBACK_PLAN",
    "QUOTA_PLAN",
    "QUOTA_REPLY",
    "QUOTA_STATUS",
    "GeminiTeacher",
    "Teacher",
    "TeacherResult",
    "build_teacher",
    "get_teacher",
    "log_case",
    "set_teacher",
]
