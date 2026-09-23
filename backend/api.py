"""HTTP contract — frozen at stage 1.

Every later stage (router, teacher, miner, metrics) writes against these shapes.
Fields are only ever added, never renamed. Endpoints that belong to a later
stage already exist here as stubs so the stage-1 frontend is complete.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import db
from .actions import Action, validate_plan
from .brain.engine import get_engine
from .brain.router import RouterHit, route
from .brain.skill import load_skills, parse_skill
from .miner import mine_once, mining_due
from .state import apply_actions, iso, read_state, utcnow
from .teacher import get_teacher, log_case

router = APIRouter(prefix="/api")

LOW_ENERGY_FOR_PLAY = 20.0

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
    text: str = Field(min_length=1, max_length=MAX_CHAT_TEXT)


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
    #: money, so the spend is visible without opening the console.
    teacher_calls_24h: int = 0


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
    conn = db.get_conn()
    with db.lock:
        cursor = conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (payload.value, payload.interaction_id),
        )
        conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="unknown interaction_id")
    return {"ok": True}


@router.get("/skills", response_model=list[SkillCard])
def get_skills() -> list[dict]:
    """Every skill in the library, with its usage counted from `interactions`."""
    conn = db.get_conn()
    with db.lock:
        skills = load_skills(conn)
        stats = conn.execute(
            # `IS` rather than `=` so an un-rated interaction (feedback NULL)
            # counts as 0 instead of turning the whole SUM into NULL.
            "SELECT skill_id,"
            " COUNT(*) AS uses,"
            " SUM(feedback IS 1) AS likes,"
            " SUM(feedback IS -1) AS dislikes"
            " FROM interactions WHERE skill_id IS NOT NULL GROUP BY skill_id"
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
        if conn.execute("SELECT 1 FROM skills WHERE id = ?", (skill.id,)).fetchone():
            raise HTTPException(status_code=409, detail=f"skill {skill.id} already exists")

        skill = skill.model_copy(update={"status": "active", "origin": "mined"})
        conn.execute(
            "INSERT INTO skills (id, json, status, origin, created_at) VALUES (?, ?, ?, ?, ?)",
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
    conn = db.get_conn()
    with db.lock:
        rows = conn.execute(
            "SELECT engine, COUNT(*) AS n, AVG(latency_ms) AS avg_ms"
            " FROM interactions GROUP BY engine"
        ).fetchall()
        skills_active = conn.execute(
            "SELECT COUNT(*) AS n FROM skills WHERE status = 'active'"
        ).fetchone()["n"]
        # `teacher_log` has no timestamp column, so the age comes from the
        # interaction it belongs to. `iso()` always renders UTC with a fixed
        # date-time prefix, so comparing the strings orders the instants — the
        # optional `.ffffff` only ever appears after the part that differs.
        since = iso(utcnow() - timedelta(hours=24))
        teacher_calls = conn.execute(
            "SELECT COUNT(*) AS n FROM teacher_log t"
            " JOIN interactions i ON i.id = t.interaction_id"
            " WHERE i.ts >= ?",
            (since,),
        ).fetchone()["n"]

    by_engine = {row["engine"]: row for row in rows}
    laya = by_engine.get("laya")
    gemini = by_engine.get("gemini")
    laya_n = laya["n"] if laya else 0
    gemini_n = gemini["n"] if gemini else 0
    total = sum(row["n"] for row in rows)
    routed = laya_n + gemini_n

    return {
        # The headline learning metric: share of understood commands Laya handled alone.
        "laya_share": round(laya_n / routed, 3) if routed else 0.0,
        "avg_latency_laya_ms": round(laya["avg_ms"], 1) if laya else 0.0,
        "avg_latency_gemini_ms": round(gemini["avg_ms"], 1) if gemini else 0.0,
        "skills_active": int(skills_active),
        "total_commands": int(total),
        "teacher_calls_24h": int(teacher_calls),
    }
