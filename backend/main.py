"""FastAPI application: API routes first, static frontend mounted last."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from . import db
from .api import router

FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init()
    yield
    db.close()


app = FastAPI(title="Pixel", version="0.1.0", lifespan=lifespan)
app.include_router(router)

# Must be the last registration: mounted at "/", StaticFiles would otherwise
# swallow every /api/* route.
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
