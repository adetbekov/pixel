# Pixel

Pixel is a self-learning robot pet that lives on a web page. You talk to it with text or quick-action
buttons — feed it, pet it, play with it, ask it to dance — and it keeps a character and a state
(mood, energy, fullness) that drifts over time and reacts to what you do.

The point is that Pixel learns. A fast local router (Laya) recognises known commands in a few
hundred milliseconds on CPU. Anything unknown goes to a teacher model (Gemini), which answers using a fixed library
of physical actions. When several similar unknown commands pile up, a skill miner proposes a new
skill, validates it against past examples, and — after the user approves it — adds it to the skill
library. Over time Laya handles more and more on its own and Gemini is called less.

Stack: Python, FastAPI, Laya (local), Gemini API, SQLite, HTML/JS frontend with an SVG robot.

Tracking issue: JEB-1495.

## The fast path

`POST /api/chat` costs at most two forward passes of one local model:

1. **Which skill?** One `choice` question over every active skill plus a mandatory `unknown`
   option. Below the skill's threshold (`ROUTER_THRESHOLD`, default `0.6`), or on `unknown`, the
   router reports a miss and hands it to the teacher.
2. **How should it behave?** All of the chosen skill's questions in a single batched call. A skill
   with no questions skips this pass.

A skill is JSON, never code: it combines the fixed action library (`backend/actions.py`) through
`when -> actions` rules, first match wins. A skill naming an action outside the library is refused
at load time and logged; the rest of the library keeps serving.

Laya is reached only through `DecisionEngine` (`backend/brain/engine.py`) — the one module that
imports `laya`. The checkpoint is `convaiinnovations/laya` + `subfolder="multilingual"`; the English
root checkpoint answers Cyrillic confidently and wrongly.

## The slow path — the teacher

Only a router miss reaches Gemini (`backend/teacher/`). The order is the product: Laya first,
always, because "share of commands handled without Gemini" is the metric the whole project is
measured on. `latency_ms` on a teacher reply covers the router miss as well as the Gemini call —
it is what the user actually waited.

Gemini writes no code. It returns a `{reply, actions}` plan, and that plan passes two independent
checks: the response schema (`backend/teacher/schema.py`) for the shape, and `validate_plan`
(`backend/actions.py`) for membership of the action library. The second is the one that matters —
a schema cannot stop `{"action": "hack_nasa"}` in a string field. A plan that is invalid, empty or
carries a blank `reply` is retried once with the reason, and a second failure answers with a fixed
fallback plan; a timeout or an API error does the same. The user never sees a traceback.

Two clocks bound the wait: 8 s per call (`TIMEOUT_S`) and 12 s across both attempts
(`TOTAL_DEADLINE_S`), the retry getting whatever is left. The SDK surface the teacher calls is
pinned (`google-genai>=2.25,<3`) and asserted against the installed package by
`test_the_sdk_still_has_the_surface_we_call` — a renamed argument would otherwise reach production
as a fallback plan and a log line.

Every teacher call writes a row to `teacher_log` — state, original command, router confidence, the
raw model response and the validated plan. That table is the only input stage 4's skill miner has,
so fallbacks are logged too. `raw_response` is never returned over the API.

Without `GEMINI_API_KEY` the app still starts: the teacher is off, a miss answers with a polite
stub, and `/api/metrics` shows a Gemini share of zero. `GEMINI_TEACHER_MODEL` overrides the model
(default `models/gemini-2.5-flash-lite`).

## Learning — the skill miner

Gemini answering a command is not learning; it costs money every single time. Learning is the moment
a *pattern* in those answers becomes a skill and the command stops reaching Gemini at all. That is
`backend/miner/`, and it runs on every `MINER_BATCH`-th unmined case (default 5) or on
`POST /api/mine`:

1. **Cluster.** Sentence vectors come from the Laya checkpoint already in memory
   (`DecisionEngine.embed`) — local, free, and still no `import laya` outside `engine.py`.
   Single-link agglomerative clustering on cosine >= `MINER_SIM` (0.75), which is connected
   components of the similarity graph, in numpy. Clusters under `MINER_MIN_CLUSTER` (3) stay in the
   pool and ripen. Without embeddings the commands are grouped by one Gemini call instead — a
   degraded path, not the default one.
2. **Generate.** One call to `GEMINI_MINER_MODEL` (default `models/gemini-2.5-flash-lite`,
   $0.30 / $2.50 per 1M) per cluster. Offline, nobody waiting, and what comes back is a schema that
   will route thousands of later commands; raise the model through the env var if drafts start
   failing validation, never loosen the validation. A mined skill carries no
   `questions` and branches only on the robot's own state; it is assembled into a real `Skill`, and
   that is where `validate_plan` refuses anything outside the action library.
3. **Backtest.** Every case of the cluster is re-routed against `active + candidate`, and a match
   means the router picked the candidate *and* did what the teacher did (action names only — the
   teacher never phrases a reply the same way twice). `match_rate` must reach `MINER_MIN_MATCH`
   (0.8).
4. **Regression check.** A match rate cannot see the damage a new option does to the old ones: stage
   2 measured all 24 orderings of the four starter skills spreading the hit rate over 7/10…9/10, and
   alphabetical order pushing "покорми" under its threshold outright. So one control phrase per
   active skill (`examples[0]`, so the set grows with the library) is routed through the same trial
   registry, and one phrase leaving its own skill kills the proposal.
5. **Propose.** `GET /api/proposals` shows the card. **The miner never activates anything** — only
   `POST /api/proposals/{id}/accept` adds the skill, and it takes effect in the same process, since
   `/api/chat` reads the library on every request. `reject` puts the cases back in the pool and
   remembers the case set, so the same cluster is not offered again.

A mined `description` is a hard 60 characters and a candidate over it is rejected, not trimmed: the
description *is* the router's option label, and stage 2 measured long ones dropping routing from 6/6
to 2/6 — for every skill, not just the new one.

```
pip install -e ".[dev]"
uvicorn backend.main:app --reload     # first run downloads the weights (~650 MB) into HF_HOME
PIXEL_SKIP_MODEL=1 uvicorn backend.main:app   # UI and buttons only; /api/chat answers 503
```

Tests never touch the model or the network — they run against `FakeEngine` and `FakeGeminiClient`
(`tests/fakes.py`).

## Deploy

Live at **https://pixel.yeldos.dev** — one container on the NAS (Portainer stack `pixel`), TLS and
access control at Nginx Proxy Manager. The app has **no auth and no rate limit**, and both
`/api/chat` and `/api/mine` spend `GEMINI_API_KEY`, so the proxy host carries an Access List (HTTP
Basic or IP allow-list). That list is the only thing standing between a loop script and the key's
quota — do not publish the host without it.

`Dockerfile` + `docker-compose.yml` are the whole deployment. Redeploy after a merge is two steps:

```
docker build -t pixel:latest .                    # on the NAS, from a fresh checkout of dev
docker compose -p pixel up -d --force-recreate     # or: redeploy the `pixel` stack in Portainer
```

- **Volume `pixel_pixel_data` → `/data`** — mandatory. It holds the SQLite DB (`PIXEL_DB_PATH`, WAL
  mode) *and* the ~650 MB Laya checkpoint (`HF_HOME`). Lose it and the robot forgets every mined
  skill and re-downloads the weights on the next start.
- **Memory** — 3 GB limit; 2 GB is the floor (mmBERT-base, 322M, resident in the process).
- **Network** — `npm_network` (external, owned by the Nginx Proxy Manager stack). No published
  port; port 8000 is reachable only from the proxy.
- **Env** — see `.env.example`; values live in the Portainer stack env, never in the repo. A missing
  `GEMINI_API_KEY` is supported: Pixel starts and runs on Laya alone.
- **Cold start** is slow by design (~70 s: weights download), warm start 8-10 s. The healthcheck
  allows a 180 s `start_period` — shorten it and the orchestrator kills the download and restarts
  into the same download.

## Pipeline

PRs target `dev`; `main` is the release branch. CI runs lint + tests, the frontend lint and an
`image build` gate on every PR, and emits `check_suite`, which is the review hand-off gate.

`image build` builds the same `Dockerfile` the NAS redeploy builds, so a broken image fails the PR
instead of the stand. It then asks the built image what it actually installed: torch must equal the
pin in `requirements-torch.txt` — the one file the image *and* the test job install from — it must
not be a CUDA wheel, and `laya` must land inside the range in `pyproject.toml`. Then it smoke-runs
the container with `PIXEL_SKIP_MODEL=1` and asserts 200 from `/api/state`, `/` and `/app.js`. The
last one is the editable-install regression: a non-editable `pip install .` moves the package into
site-packages, `/frontend` stops existing and the whole UI 404s while the API still answers.
Nothing is pushed to a registry — the image is a gate, not a deploy.
