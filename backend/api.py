"""HTTP contract — frozen at stage 1.

Every later stage (router, teacher, miner, metrics) writes against these shapes.
Fields are only ever added, never renamed. Endpoints that belong to a later
stage already exist here as stubs so the stage-1 frontend is complete.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from . import db, feedback
from .actions import Action, validate_plan
from .brain.engine import get_engine
from .brain.router import RouterHit, route
from .brain.skill import load_skills, parse_skill
from .metrics import collect as collect_metrics
from .miner import mine_once, mining_due
from .state import apply_actions, iso, read_state, utcnow
from .teacher import get_teacher, log_case

router = APIRouter(prefix="/api")

LOW_ENERGY_FOR_PLAY = 20.0

#: How much chat `GET /api/history` redraws after a reload, and the ceiling a
#: caller may ask for — the chat is a conversation, not an archive.
DEFAULT_HISTORY = 30
MAX_HISTORY = 200

# A chat call costs a forward pass behind the engine's single lock, so an unbounded
# body is one caller stalling every other chat. Laya's context is 1024 tokens and a
# real command is a handful of words; stage 3 trims to the same length before Gemini.
MAX_CHAT_TEXT = 500

BUTTON_PLANS: dict[str, list[dict[str, Any]]] = {
    "feed": [
        {"action": "eat", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": "Ням!"}},
    ],
    "pet": [
        {"action": "wave", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": "Приятно!"}},
    ],
    "play": [
        {"action": "jump", "args": {}},
        {"action": "spin", "args": {}},
        {"action": "set_face", "args": {"face": "happy"}},
        {"action": "say", "args": {"text": "Ура!"}},
    ],
}

# A tired robot refuses to play. Stage 4's mined "play" skill reuses this rule.
TIRED_PLAN: list[dict[str, Any]] = [
    {"action": "set_face", "args": {"face": "sleepy"}},
    {"action": "say", "args": {"text": "Я устал, давай позже"}},
]

# What a router miss answers with when the teacher is off (no GEMINI_API_KEY).
MISS_PLAN: list[dict[str, Any]] = [
    {"action": "set_face", "args": {"face": "curious"}},
    {"action": "say", "args": {"text": "Я пока не понял"}},
]


class StateOut(BaseModel):
    mood: float
    energy: float
    fullness: float
    face: str


class ActionOut(BaseModel):
    action: str
    args: dict[str, Any] = Field(default_factory=dict)


class Reply(BaseModel):
    interaction_id: int
    reply: str
    actions: list[ActionOut]
    engine: Literal["laya", "gemini", "button"]
    skill_id: str | None
    confidence: float | None
    latency_ms: int
    state: StateOut


class ChatIn(BaseModel):
    # `strip_whitespace` runs before `min_length`, so a body of nothing but
    # spaces is a 422 from Pydantic rather than a forward pass behind the
    # engine's lock. It also means the engine and the log see the same trimmed
    # text, and `max_length` counts what is actually sent to the model.
    text: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHAT_TEXT),
    ]


class ActionIn(BaseModel):
    name: Literal["feed", "pet", "play"]


class FeedbackIn(BaseModel):
    interaction_id: int
    value: Literal[1, -1]


class Ok(BaseModel):
    ok: bool = True


class MineOut(BaseModel):
    started: bool = True
    proposals: int = 0


class SkillCard(BaseModel):
    id: str
    name: str
    description: str
    status: str
    origin: str
    uses: int = 0
    likes: int = 0
    dislikes: int = 0
    #: Why a disabled skill is off — `"dislike_rate"` is the only automatic one.
    disabled_reason: str | None = None


class HistoryItem(BaseModel):
    """One past interaction, enough to redraw its chat bubble and its vote."""

    interaction_id: int
    user_text: str
    reply: str
    engine: str
    skill_id: str | None = None
    confidence: float | None = None
    latency_ms: int = 0
    feedback: int | None = None


class Proposal(BaseModel):
    id: str
    skill: dict[str, Any]
    match_rate: float
    sample_ids: list[int]
    status: str
    created_at: str


class Metrics(BaseModel):
    laya_share: float
    avg_latency_laya_ms: float
    avg_latency_gemini_ms: float
    skills_active: int
    total_commands: int
    #: Teacher calls in the last 24h — Gemini is the only thing here that costs
    #: money, so the spend is visible without opening the console. Measured live
    #: on 2026-09-24 (JEB-1508) against `models/gemini-2.5-flash-lite`: ~480 in +
    #: ~101 out tokens per call, **$0.0004 a call**, i.e. ~$0.40 per 1000 —
    #: divide this count by 2500 for dollars. Re-measure if the prompt or the
    #: model changes.
    teacher_calls_24h: int = 0
    #: The same headline share over the last 24h: the lifetime number moves
    #: slowly once there is history behind it, this one shows today's trend.
    laya_share_24h: float = 0.0
    gemini_calls_24h: int = 0
    skills_disabled: int = 0


def _log_interaction(
    conn,
    *,
    user_text: str,
    engine: str,
    skill_id: str | None,
    confidence: float | None,
    latency_ms: int,
    actions: list[Action],
    reply_text: str,
) -> int:
    cursor = conn.execute(
        "INSERT INTO interactions"
        " (ts, user_text, engine, skill_id, confidence, latency_ms, actions_json, reply_text,"
        "  feedback)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)",
        (
            iso(utcnow()),
            user_text,
            engine,
            skill_id,
            confidence,
            latency_ms,
            json.dumps([a.to_dict() for a in actions], ensure_ascii=False),
            reply_text,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def _reply_text(actions: list[Action], fallback: str) -> str:
    for action in actions:
        if action.action == "say":
            return action.args["text"]
    return fallback


def execute_plan(
    *,
    user_text: str,
    raw_plan: list[dict[str, Any]],
    engine: str,
    started: float,
    skill_id: str | None = None,
    confidence: float | None = None,
    fallback_reply: str = "Готово!",
) -> dict:
    """The single execution path: validate, apply, log, and shape a ``Reply``.

    Stages 2-4 route the router's and the teacher's plans through here too, so
    the frozen response shape is built in exactly one place.
    """
    conn = db.get_conn()
    with db.lock:
        actions = validate_plan(raw_plan)
        state = apply_actions(conn, actions)
        latency_ms = int((time.perf_counter() - started) * 1000)
        reply = _reply_text(actions, fallback_reply)
        interaction_id = _log_interaction(
            conn,
            user_text=user_text,
            engine=engine,
            skill_id=skill_id,
            confidence=confidence,
            latency_ms=latency_ms,
            actions=actions,
            reply_text=reply,
        )
    return {
        "interaction_id": interaction_id,
        "reply": reply,
        "actions": [a.to_dict() for a in actions],
        "engine": engine,
        "skill_id": skill_id,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "state": state.to_dict(),
    }


@router.get("/state", response_model=StateOut)
def get_state() -> dict:
    conn = db.get_conn()
    with db.lock:
        return read_state(conn).to_dict()


@router.post("/action", response_model=Reply)
def post_action(payload: ActionIn) -> dict:
    started = time.perf_counter()
    conn = db.get_conn()
    with db.lock:
        state = read_state(conn)

    raw_plan = BUTTON_PLANS[payload.name]
    if payload.name == "play" and state.energy < LOW_ENERGY_FOR_PLAY:
        raw_plan = TIRED_PLAN

    return execute_plan(
        user_text=payload.name,
        raw_plan=raw_plan,
        engine="button",
        started=started,
    )


@router.post("/chat", response_model=Reply)
def post_chat(payload: ChatIn) -> dict:
    """Laya first, always; Gemini only on a miss.

    The order is the product, not an implementation detail: asking both, or
    asking Gemini first, would destroy the one metric that says whether Pixel is
    learning anything. `started` is taken before the router so `latency_ms`
    covers the miss the user waited through as well as the Gemini call.
    """
    started = time.perf_counter()
    try:
        engine = get_engine()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    conn = db.get_conn()
    # `db.lock` is a plain Lock and `execute_plan` takes it itself, so this read
    # has to be finished and released before the router runs — exactly the order
    # `post_action` uses. The engine's own lock is never nested with this one.
    with db.lock:
        state = read_state(conn)
        skills = load_skills(conn, only_active=True)

    outcome = route(engine, skills, payload.text, state)
    if isinstance(outcome, RouterHit):
        return execute_plan(
            user_text=payload.text,
            raw_plan=outcome.raw_plan,
            engine="laya",
            started=started,
            skill_id=outcome.skill_id,
            confidence=outcome.confidence,
        )

    teacher = get_teacher()
    if teacher is None:
        # No API key: Pixel runs on Laya alone and says so politely.
        return execute_plan(
            user_text=payload.text,
            raw_plan=MISS_PLAN,
            engine="laya",
            started=started,
            confidence=outcome.confidence,
        )

    # Deliberately outside every lock: `db.lock` is not reentrant and this is a
    # network call that can sit for the full TIMEOUT_S.
    result = teacher.explain(payload.text, state, skills)
    reply = execute_plan(
        user_text=payload.text,
        raw_plan=result.raw_plan,
        engine="gemini",
        started=started,
        confidence=outcome.confidence,
        fallback_reply=result.reply,
    )

    with db.lock:
        log_case(
            conn,
            interaction_id=reply["interaction_id"],
            user_text=payload.text,
            # The state the teacher reasoned about, before its own plan moved it.
            state=state,
            confidence=outcome.confidence,
            raw_response=result.raw_response,
            actions=reply["actions"],
            error=result.error,
        )
        due = mining_due(conn)

    # Every MINER_BATCH-th miss, the miner runs before this response returns.
    # It is seconds on a pool this size, and the user who just taught Pixel
    # something is the one most likely to be looking at the skills panel.
    if due:
        mine_once()
    return reply


@router.post("/feedback", response_model=Ok)
def post_feedback(payload: FeedbackIn) -> dict:
    """Store the vote — and, on a 👎, let it cost the skill behind it.

    Re-rating the same interaction overwrites the earlier value; the skill's
    health is recounted from the table afterwards, so a flipped vote flips the
    verdict with it. See `backend/feedback.py`.
    """
    conn = db.get_conn()
    with db.lock:
        known = feedback.record(
            conn, interaction_id=payload.interaction_id, value=payload.value
        )
    if not known:
        raise HTTPException(status_code=404, detail="unknown interaction_id")
    return {"ok": True}


@router.get("/history", response_model=list[HistoryItem])
def get_history(limit: int = DEFAULT_HISTORY) -> list[dict]:
    """The last interactions, oldest first — what the chat redraws after a reload.

    Without this the votes survive the reload in the database but vanish from
    the screen, which reads as "my 👎 was not saved".
    """
    limit = max(1, min(limit, MAX_HISTORY))
    conn = db.get_conn()
    with db.lock:
        rows = conn.execute(
            "SELECT id, user_text, reply_text, engine, skill_id, confidence, latency_ms,"
            " feedback FROM interactions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {
            "interaction_id": row["id"],
            "user_text": row["user_text"],
            "reply": row["reply_text"],
            "engine": row["engine"],
            "skill_id": row["skill_id"],
            "confidence": row["confidence"],
            "latency_ms": row["latency_ms"],
            "feedback": row["feedback"],
        }
        for row in reversed(rows)
    ]


@router.get("/skills", response_model=list[SkillCard])
def get_skills() -> list[dict]:
    """Every skill in the library, with its usage counted from `interactions`."""
    conn = db.get_conn()
    with db.lock:
        skills = load_skills(conn)
        reasons = {
            row["id"]: row["disabled_reason"]
            for row in conn.execute("SELECT id, disabled_reason FROM skills")
        }
        stats = conn.execute(
            # `IS` rather than `=` so an un-rated interaction (feedback NULL)
            # counts as 0 instead of turning the whole SUM into NULL.
            #
            # `i.ts >= s.created_at` is the same incarnation boundary
            # `backend/feedback.py` judges a skill over: a skill re-mined and
            # accepted under an id it once held must not open its card on the
            # previous life's "👎 4 · использован 10 раз". The join also replaces
            # the old `skill_id IS NOT NULL` — a Gemini answer joins to no skill.
            "SELECT i.skill_id AS skill_id,"
            " COUNT(*) AS uses,"
            " SUM(i.feedback IS 1) AS likes,"
            " SUM(i.feedback IS -1) AS dislikes"
            " FROM interactions i JOIN skills s ON s.id = i.skill_id"
            " WHERE i.ts >= s.created_at GROUP BY i.skill_id"
        ).fetchall()

    unused = {"uses": 0, "likes": 0, "dislikes": 0}
    counts = {row["skill_id"]: {key: row[key] for key in unused} for row in stats}
    return [
        {
            "id": skill.id,
            "name": skill.name,
            "description": skill.description,
            "status": skill.status,
            "origin": skill.origin,
            "disabled_reason": reasons.get(skill.id),
            **counts.get(skill.id, unused),
        }
        for skill in skills
    ]


@router.get("/proposals", response_model=list[Proposal])
def get_proposals() -> list[dict]:
    """Skills the miner wants to add — pending ones only.

    The card's example phrases live in `skill.examples`; there is no second
    `examples` field, and the frozen `Proposal` shape says so.
    """
    conn = db.get_conn()
    with db.lock:
        rows = conn.execute(
            "SELECT id, skill_json, match_rate, sample_ids, status, created_at"
            " FROM skill_proposals WHERE status = 'pending' ORDER BY created_at, rowid"
        ).fetchall()
    return [
        {
            "id": row["id"],
            "skill": json.loads(row["skill_json"]),
            "match_rate": row["match_rate"],
            "sample_ids": json.loads(row["sample_ids"]),
            "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


@router.post("/proposals/{proposal_id}/accept", response_model=Ok)
def accept_proposal(proposal_id: str) -> dict:
    """The one and only way a mined skill enters the library.

    Nothing has to be reloaded and nothing has to be restarted: `/api/chat`
    reads the library with `load_skills` on every request, so the next command
    is already routed against the new skill. It lands at the end of the option
    list — which is exactly the order the backtest tried it in.

    A **disabled** skill's id is free, so a retry of a skill stage 5 turned off
    overwrites that row instead of colliding with its primary key. The miner is
    the other half of that rule (`backend/miner/run.py`, `_taken_ids`): without
    both, the retry is either refused before it is drafted or accepted into a
    500.
    """
    conn = db.get_conn()
    with db.lock:
        row = conn.execute(
            "SELECT skill_json FROM skill_proposals WHERE id = ? AND status = 'pending'",
            (proposal_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="unknown or already decided proposal")

        skill = parse_skill(json.loads(row["skill_json"]))
        if skill is None:
            raise HTTPException(status_code=422, detail="the proposed skill no longer validates")
        # Only a row that says `disabled` may be overwritten — a NULL status is
        # not a free id, it is a row nobody can vouch for.
        existing = conn.execute(
            "SELECT status FROM skills WHERE id = ?", (skill.id,)
        ).fetchone()
        if existing is not None and existing["status"] != "disabled":
            raise HTTPException(status_code=409, detail=f"skill {skill.id} already exists")

        skill = skill.model_copy(update={"status": "active", "origin": "mined"})
        # `disabled_at` / `disabled_reason` are written back as NULL rather than
        # left behind: the row is a different skill now, and a fresh card
        # carrying the old one's "отключён из-за дизлайков" is a lie.
        conn.execute(
            "INSERT OR REPLACE INTO skills"
            " (id, json, status, origin, created_at, disabled_at, disabled_reason)"
            " VALUES (?, ?, ?, ?, ?, NULL, NULL)",
            (skill.id, skill.model_dump_json(), skill.status, skill.origin, iso(utcnow())),
        )
        conn.execute("UPDATE skill_proposals SET status = 'accepted' WHERE id = ?", (proposal_id,))
        conn.commit()
    return {"ok": True}


@router.post("/proposals/{proposal_id}/reject", response_model=Ok)
def reject_proposal(proposal_id: str) -> dict:
    """Turn the proposal down and put its cases back in the pool.

    The rejected row is kept, because its `sample_ids` are what stop the miner
    proposing the very same cluster again on the next run.
    """
    conn = db.get_conn()
    with db.lock:
        cursor = conn.execute(
            "UPDATE skill_proposals SET status = 'rejected' WHERE id = ? AND status = 'pending'",
            (proposal_id,),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="unknown or already decided proposal")
        conn.execute(
            "UPDATE teacher_log SET mined = 0, cluster_id = NULL WHERE cluster_id = ?",
            (proposal_id,),
        )
        conn.commit()
    return {"ok": True}


@router.post("/mine", response_model=MineOut)
def post_mine() -> dict:
    """Mine now. An empty pool, or no API key, is `0` proposals — not an error."""
    result = mine_once()
    return {"started": result.started, "proposals": result.proposals}


@router.get("/metrics", response_model=Metrics)
def get_metrics() -> dict:
    """Every number the learning panel shows. The SQL lives in `backend/metrics.py`."""
    conn = db.get_conn()
    with db.lock:
        return collect_metrics(conn)
