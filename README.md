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
   router reports a miss and runs nothing — stage 3 hands that miss to Gemini.
2. **How should it behave?** All of the chosen skill's questions in a single batched call. A skill
   with no questions skips this pass.

A skill is JSON, never code: it combines the fixed action library (`backend/actions.py`) through
`when -> actions` rules, first match wins. A skill naming an action outside the library is refused
at load time and logged; the rest of the library keeps serving.

Laya is reached only through `DecisionEngine` (`backend/brain/engine.py`) — the one module that
imports `laya`. The checkpoint is `convaiinnovations/laya` + `subfolder="multilingual"`; the English
root checkpoint answers Cyrillic confidently and wrongly.

```
pip install -e ".[dev]"
uvicorn backend.main:app --reload     # first run downloads the weights (~650 MB) into HF_HOME
PIXEL_SKIP_MODEL=1 uvicorn backend.main:app   # UI and buttons only; /api/chat answers 503
```

Tests never touch the model or the network — they run against a `FakeEngine` (`tests/fakes.py`).

## Pipeline

PRs target `dev`; `main` is the release branch. CI runs lint + tests on every PR and
emits `check_suite`, which is the review hand-off gate.
