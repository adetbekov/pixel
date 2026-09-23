"""SQLite access. Single-user prototype: one connection, one module-level lock.

The whole schema is created up front — including the tables stages 2-4 will fill
— so no stage needs a *table*. Columns a later stage turned out to need are added
by :func:`migrate`, which runs on every connect and is a no-op once applied.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path

from .state import START_ENERGY, START_FACE, START_FULLNESS, START_MOOD, iso, utcnow

DEFAULT_DB_PATH = "./pixel.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS robot_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    mood REAL,
    energy REAL,
    fullness REAL,
    face TEXT,
    last_tick_at TEXT
);

CREATE TABLE IF NOT EXISTS skills (
    id TEXT PRIMARY KEY,
    json TEXT,
    status TEXT,
    origin TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    user_text TEXT,
    engine TEXT,
    skill_id TEXT,
    confidence REAL,
    latency_ms INTEGER,
    actions_json TEXT,
    reply_text TEXT,
    feedback INTEGER
);

CREATE TABLE IF NOT EXISTS teacher_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER,
    state_json TEXT,
    raw_response TEXT,
    actions_json TEXT,
    mined INTEGER DEFAULT 0,
    cluster_id TEXT
);

CREATE TABLE IF NOT EXISTS skill_proposals (
    id TEXT PRIMARY KEY,
    skill_json TEXT,
    match_rate REAL,
    sample_ids TEXT,
    status TEXT,
    created_at TEXT
);
"""

#: Columns added after the schema above was frozen, as ``table -> column -> type``.
#: Applied on every connect, so a ``pixel.db`` written by an earlier stage is
#: upgraded in place instead of having to be deleted.
MIGRATIONS: dict[str, dict[str, str]] = {
    # Stage 5 turns a skill off when its dislike rate is too high. `status` alone
    # cannot say *when* that happened or *why*, and the skills panel has to tell
    # the user both — an unexplained disappearance reads as a bug.
    "skills": {
        "disabled_at": "TEXT",
        "disabled_reason": "TEXT",
    },
}

lock = threading.Lock()

_conn: sqlite3.Connection | None = None


def migrate(conn: sqlite3.Connection) -> None:
    """Add every missing column from :data:`MIGRATIONS`. Idempotent by design.

    SQLite has no ``ADD COLUMN IF NOT EXISTS``, so the columns already present
    are read from ``PRAGMA table_info`` first. Running this twice is a no-op,
    which is what lets it sit on the ordinary connect path.
    """
    for table, columns in MIGRATIONS.items():
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for column, column_type in columns.items():
            if column not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
    conn.commit()


def db_path() -> str:
    return os.environ.get("PIXEL_DB_PATH", DEFAULT_DB_PATH)


def connect(path: str | None = None) -> sqlite3.Connection:
    """Open a connection with the schema in place and the state row seeded."""
    path = path or db_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    migrate(conn)
    conn.execute(
        "INSERT OR IGNORE INTO robot_state (id, mood, energy, fullness, face, last_tick_at)"
        " VALUES (1, ?, ?, ?, ?, ?)",
        (START_MOOD, START_ENERGY, START_FULLNESS, START_FACE, iso(utcnow())),
    )
    conn.commit()
    return conn


def init(path: str | None = None) -> sqlite3.Connection:
    global _conn
    close()
    _conn = connect(path)
    return _conn


def get_conn() -> sqlite3.Connection:
    if _conn is None:
        return init()
    return _conn


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
