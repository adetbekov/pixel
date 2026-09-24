"""FastAPI application: API routes first, static frontend mounted last."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import db
from .api import router
from .brain.engine import LayaEngine, set_engine
from .brain.skill import seed_db
from .miner import build_generator, set_generator, stop_worker
from .teacher import build_teacher, set_teacher

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = db.init()
    seeded = seed_db(conn)
    if seeded:
        log.info("seeded %d starter skills", seeded)

    # A cold load costs seconds and the first request must not pay it. Set
    # PIXEL_SKIP_MODEL=1 to serve the UI and the buttons without weights —
    # /api/chat then answers 503 instead of blocking on a download.
    if os.environ.get("PIXEL_SKIP_MODEL") == "1":
        log.warning("PIXEL_SKIP_MODEL=1 — /api/chat is disabled")
    else:
        set_engine(LayaEngine())

    # A missing GEMINI_API_KEY is a supported setup, not a crash: Pixel runs on
    # Laya alone and a router miss answers with the polite stub.
    teacher = build_teacher()
    set_teacher(teacher)
    if teacher is None:
        log.warning("GEMINI_API_KEY is not set — the teacher is off, Laya only")

    # The miner reads what the teacher wrote, so the two share a key and switch
    # off together: no teacher means an empty `teacher_log` and nothing to mine.
    set_generator(build_generator())

    yield
    # First, because a run in flight still reads the database and the engine.
    # It is a daemon thread, so a run that outlasts the timeout loses its work
    # and nothing else — its cases are still `mined=0` on the next start.
    stop_worker()
    set_generator(None)
    set_teacher(None)
    set_engine(None)
    db.close()


app = FastAPI(title="Pixel", version="0.1.0", lifespan=lifespan)
app.include_router(router)

# Must be the last registration: mounted at "/", StaticFiles would otherwise
# swallow every /api/* route.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
