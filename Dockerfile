# Pixel — single-process image. FastAPI serves the API and the static frontend.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    PIXEL_DB_PATH=/data/pixel.db

WORKDIR /app

# `laya` depends on torch, and the default PyPI wheel is the CUDA build (~1 GB).
# Pixel runs on CPU, so seed the CPU wheel first — in its own layer, so a code
# change never rebuilds it — and let the install below see torch as satisfied.
#
# The pin itself lives in requirements-torch.txt — the single file the `test` job
# installs from as well, so the two can no longer drift the way JEB-1516 found them
# drifted. This layer rebuilds only when that file changes. CI's `image build` job
# asserts the built image really carries that version and is not a CUDA wheel.
COPY requirements-torch.txt ./
RUN pip install -r requirements-torch.txt

# `readme = "README.md"` in pyproject.toml, so the build needs it.
COPY pyproject.toml README.md ./
COPY backend/ backend/
COPY frontend/ frontend/

# Editable on purpose: `backend/main.py` resolves the static frontend as
# `parents[1] / "frontend"`. A non-editable install moves the package into
# site-packages, that path stops existing, and the UI silently 404s while the
# API still answers.
RUN pip install -e .

# Weights (~650 MB) and the SQLite file both live here. Without a volume every
# restart re-downloads the weights and drops the skill library.
VOLUME ["/data"]

EXPOSE 8000

# Exactly one worker. Each worker loads its own copy of the model (memory × N),
# and backend/db.py guards a single connection with one module-level lock —
# a second process breaks that guarantee.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
