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

## The loop, in one line

```
command -> Laya (router) -> hit: a skill answers in ms
                         -> miss: Gemini answers and the case is logged
                                -> miner clusters the cases, drafts a skill, backtests it
                                       -> user accepts -> the skill is in the library
                                              -> the same command never reaches Gemini again
```

Two things close the loop back: 👎 on an answer, which switches a skill off once its dislike rate is
high enough and returns its cases to the miner's pool, and `laya_share` — the share of commands
handled without Gemini, which is the number the whole project is measured on.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # CPU torch first if you are on a GPU-less box, see CI
cp .env.example .env               # then fill GEMINI_API_KEY, or leave it empty
uvicorn backend.main:app --reload  # http://127.0.0.1:8000
```

The first run downloads the Laya weights (~650 MB) into `HF_HOME`. Two shortcuts while developing:

```bash
PIXEL_SKIP_MODEL=1 uvicorn backend.main:app   # no weights: UI and buttons work, /api/chat is 503
python3 -m pytest -q                          # never touches the model or the network
```

The frontend is served by the same app at `/`; `frontend/index.html?mock=1` runs it against the
fixtures in `frontend/mocks/` with no backend at all.

Nothing here needs a migration step: `backend/db.py` creates the schema and applies its column
migrations on every connect, so an existing `pixel.db` from an earlier stage keeps working.

## Environment

All of it is optional except the Gemini key, and an empty key is a supported configuration — Pixel
then runs on Laya alone. Defaults are in `.env.example`.

| Key | Default | What it does |
| --- | --- | --- |
| `GEMINI_API_KEY` | — | Empty = teacher and miner are off. A router miss answers with a polite stub. |
| `GEMINI_TEACHER_MODEL` | `models/gemini-2.5-flash-lite` | Model that answers router misses. |
| `GEMINI_MINER_MODEL` | `models/gemini-2.5-flash-lite` | Model that drafts new skills, offline. |
| `PIXEL_DB_PATH` | `./pixel.db` | SQLite file. |
| `LAYA_MODEL` | `multilingual` | Laya subfolder. The English root checkpoint answers Cyrillic confidently and wrongly. |
| `LAYA_DEVICE` | `cpu` | Where the local model runs. |
| `PIXEL_SKIP_MODEL` | `0` | `1` = start without Laya; `/api/chat` answers 503. |
| `ROUTER_THRESHOLD` | `0.6` | Confidence a skill needs to win the router. Below it, the command is a miss. |
| `MINER_BATCH` | `5` | Mine on every N-th unmined case. |
| `MINER_SIM` | `0.88` | Cosine similarity that joins two commands into one cluster. Narrow usable band — see `backend/miner/cluster.py`. |
| `MINER_MIN_CLUSTER` | `3` | Cases below this never become a skill. |
| `MINER_MIN_MATCH` | `0.8` | Share of its cluster a candidate must reproduce to be proposed. |
| `SKILL_DISLIKE_LIMIT` | `0.30` | Dislike share above which a skill is switched off (strictly greater). |
| `SKILL_MIN_RATED` | `5` | Ratings required before that rule applies at all. |

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
   Single-link agglomerative clustering on cosine >= `MINER_SIM` (0.88), which is connected
   components of the similarity graph, in numpy. The default is measured against the real
   checkpoint: these vectors are anisotropic, so the band that separates "same request" from
   "different request" sits high and is narrow. Clusters under `MINER_MIN_CLUSTER` (3) stay in the
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

Tests never touch the model or the network — they run against `FakeEngine` and `FakeGeminiClient`
(`tests/fakes.py`).

## Unlearning — feedback and metrics

The miner only ever adds. Without the other direction, a skill Gemini worded badly keeps winning the
router forever and every command it steals is answered wrong — fast, cheaply, and wrong. So
`POST /api/feedback` is not just a counter (`backend/feedback.py`):

* a vote is stored on the interaction and **overwrites** an earlier vote on the same one, so the
  skill's health is always recounted from the table, never accumulated;
* once a skill has at least `SKILL_MIN_RATED` (5) ratings **and** more than `SKILL_DISLIKE_LIMIT`
  (30%) of them are 👎, it is disabled. The floor is what stops a single 👎 on a new skill's first
  use from killing it; 4 dislikes in 10 disables, 2 in 3 does not;
* disabling takes effect on the very next command — `/api/chat` reads the library per request, so
  the skill is simply not among the router's options any more and the command goes to Gemini;
* the skill is **not** deleted. It stays in `GET /api/skills` with `status="disabled"` and
  `disabled_reason="dislike_rate"`, because a skill that silently disappears reads as a bug;
* its mined cases go back to `teacher_log` unmined, and **every** lock on re-mining that cluster
  comes off with them — the *rejected* proposal covering exactly that case set is retired, and the
  skill's **id** stops counting as taken (`_taken_ids` in `backend/miner/run.py`). Both matter: the
  generator is only ever shown the *active* library, so its next draft for the same phrases picks
  the same obvious id, and one lock left on drops the retry into a `log.warning` on every run,
  forever. Accepting the retry overwrites the disabled row and clears `disabled_at` /
  `disabled_reason`; an id held by an **active** skill is still a 409;
* a re-accepted skill is judged on its **current life only**. `skills` and `interactions` have no
  foreign key, so the dead incarnation's rows keep the id; `skills.created_at` — rewritten on every
  accept — is the boundary (`i.ts >= s.created_at`), used by both the health query and the card
  counts. Without it a freshly re-approved skill inherits the 4/10 that killed it and dies on its
  first new rating, a 👍 included, with a card already reading "👎 4";
* seed skills get no exemption. A starter skill the user keeps disliking is exactly as wrong.

A 👎 on a Gemini answer has no skill to disable; it stays in the metrics and stays raw material.

`GET /api/metrics` is all queries, no metrics table — a second copy of a count can only disagree
with the rows it came from. `laya_share` = `laya / (laya + gemini)`; **button clicks are not
commands** and never enter it, or the headline number would be inflated with clicks. Every ratio is
guarded, so an empty database answers `0.0` and the panel says "пока нет данных" rather than `NaN%`.
Alongside the lifetime share the panel shows the same share over 24h — the lifetime number moves
slowly once there is history behind it, so the windowed one is where the learning is visible.

The panel refreshes on load, after every reply, after a vote, and after accept/reject — on events,
never on a timer. `GET /api/history` redraws the last interactions with their votes after a page
reload; without it the 👎 survives in the database but vanishes from the screen.

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

`image build` covers both halves of the deployment — `Dockerfile` **and** `docker-compose.yml`.

It starts with the compose file, because that check costs under a second and needs no image:
`docker compose config` parses it, validates it against the schema and resolves every
`${VAR:-default}`, then `.github/scripts/verify_compose.py` asserts the stack runs the image the
redeploy above actually builds. Both are load-bearing on a stand where `pull_policy: never` means a
tag nobody builds is not an error but a silently stale container, and where the compose file is
otherwise first executed on production.

Then it builds the same `Dockerfile` the NAS redeploy builds, so a broken image fails the PR
instead of the stand. It asks the built image what it actually installed: torch must equal the
pin in `requirements-torch.txt` — the one file the image *and* the test job install from — it must
not be a CUDA wheel, and `laya` must land inside the range in `pyproject.toml`. Then it smoke-runs
the container with `PIXEL_SKIP_MODEL=1` and asserts 200 from `/api/state`, `/` and `/app.js`. The
last two are the missing-frontend regression: an image that lost `frontend/` — a dropped `COPY`, a
bad `.dockerignore` — 404s the whole UI while `/api/state` still answers 200. It is *not* an
editable-install gate: JEB-1530 measured an image built with plain `pip install .` serving all
three paths, because `WORKDIR /app` plus uvicorn's default `--app-dir ""` make `/app/backend`
shadow the site-packages copy either way. Nothing is pushed to a registry — the image is a gate,
not a deploy.

One gate runs **outside** the PR: `Live Gemini Contract`, nightly at 03:17 UTC and on
`workflow_dispatch`. Everything above runs against `tests/fakes.py::FakeGeminiClient`, which returns
bare JSON whatever it is asked — so the teacher and miner paths stay green on fixtures while the
live path is dead. That is not hypothetical: `interactions.create` + `response_format` does not hold
structured output on `models/gemini-2.5-flash-lite` (the answer comes back in a ```` ```json ````
fence), and both halves of the project shipped that call and had to be moved to
`models.generate_content` + `response_schema`, both times found by hand on a live stand.

So the nightly calls `GeminiTeacher._call` and `GeminiSkillGenerator._call` for real, once each, and
parses the answers with `TeacherPlan` / `SkillDraft` — no fence-stripping, because the strict parser
is what makes the break visible. Two `flash-lite` calls a day, on the order of $0.001. It needs the
`GEMINI_API_KEY` repository secret, which is why it never runs on a PR: a fork PR cannot have it,
and a green CI must not depend on a key. It is not a required check on any branch — it can go red
because Google changed something, and that must never block a merge. A failing nightly opens (or
comments on) an issue labelled `live-gemini-contract`.
