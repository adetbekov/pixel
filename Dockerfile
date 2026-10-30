# Pixel — single-process image. FastAPI serves the API and the static frontend.
FROM python:3.11-slim

# uv, pinned: the lockfile format and resolver behaviour move with it.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    # The install is a function of the commit, not of a download cache in the layer.
    UV_NO_CACHE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    HF_HOME=/data/hf \
    PIXEL_DB_PATH=/data/pixel.db

WORKDIR /app

# Dependencies first, in their own layer, from the lockfile alone — so a code change
# never re-downloads laya and its ~200 MB CPU torch wheel. `--frozen` refuses to
# resolve: the image installs exactly what `uv.lock` records, which is what CI's
# `lint + tests` ran against. `--extra engine` is the laya + torch extra that only
# this image installs; `pyproject.toml` pins both and routes torch to the CPU index
# (JEB-1516: the pin lives in one file, and CI's `image build` job asserts the built
# image really carries that version and not a CUDA wheel).
#
# `readme = "README.md"` in pyproject.toml, so the project build below needs it.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra engine

COPY backend/ backend/
COPY frontend/ frontend/

# What keeps `backend/main.py`'s `parents[1] / "frontend"` resolvable is the layout
# above — `COPY backend/` and `COPY frontend/` as siblings under `WORKDIR /app` —
# not the install. uvicorn's default `--app-dir ""` puts the cwd first on `sys.path`,
# so /app/backend shadows any site-packages copy either way: JEB-1530 measured the
# same image with a non-editable install still serving `/api/state`, `/` and
# `/app.js` with 200.
#
# Drop `COPY frontend/` and the UI really does 404 while the API still answers 200.
# That is the regression CI's smoke step catches.
RUN uv sync --frozen --no-dev --extra engine

# Weights (~650 MB) and the SQLite file both live here. Without a volume every
# restart re-downloads the weights and drops the skill library.
VOLUME ["/data"]

EXPOSE 8000

# Exactly one worker. Each worker loads its own copy of the model (memory × N),
# and backend/db.py guards a single connection with one module-level lock —
# a second process breaks that guarantee.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
