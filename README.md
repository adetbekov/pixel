# Pixel

Pixel is a self-learning robot pet that lives on a web page. You talk to it with text or quick-action
buttons — feed it, pet it, play with it, ask it to dance — and it keeps a character and a state
(mood, energy, fullness) that drifts over time and reacts to what you do.

The point is that Pixel learns. A fast local router (Laya) recognises known commands in tens of
milliseconds. Anything unknown goes to a teacher model (Gemini), which answers using a fixed library
of physical actions. When several similar unknown commands pile up, a skill miner proposes a new
skill, validates it against past examples, and — after the user approves it — adds it to the skill
library. Over time Laya handles more and more on its own and Gemini is called less.

Stack: Python, FastAPI, Laya (local), Gemini API, SQLite, HTML/JS frontend with an SVG robot.

Tracking issue: JEB-1495.

## Pipeline

PRs target `dev`; `main` is the release branch. CI runs lint + tests on every PR and
emits `check_suite`, which is the review hand-off gate.
