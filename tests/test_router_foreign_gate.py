"""The live-router gate's own judgement, checked without the weights.

`.github/scripts/check_router_foreign_commands.py` only runs the real checkpoint
in its own workflow, so what can rot unnoticed here is the verdict: a gate that
cannot fail looks exactly like one that passes. These tests drive it through
`FakeEngine` with the QA reproduction from JEB-1547 — two of five `сальто`
phrases taken by `play` — and assert it goes red for both reasons it names.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from backend.brain.router import normalize
from backend.brain.skill import load_seed_skills
from backend.miner.run import batch_size

from .fakes import FakeEngine

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "check_router_foreign_commands.py"

spec = importlib.util.spec_from_file_location("check_router_foreign_commands", SCRIPT)
gate = importlib.util.module_from_spec(spec)
# Registered before it executes: `@dataclass` resolves annotations through
# `sys.modules[cls.__module__]`, and a module loaded by path is not there yet.
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

ABSORBED = {"покажи сальто": "play", "крутани сальто": "play"}


def run_gate(engine: FakeEngine) -> list[str]:
    return gate.judge(gate.route_topics(engine, load_seed_skills()))


def test_everything_reaching_the_teacher_is_green():
    assert run_gate(FakeEngine(pick="unknown", confidence=0.9)) == []


def test_the_qa_reproduction_is_red_for_both_reasons():
    """`8dccba3`: 0.605 and 0.631 into `play`, 3 of 5 left, batch never reached."""
    findings = run_gate(FakeEngine(confidence=0.95, routes=ABSORBED))

    absorbed = [line for line in findings if "absorbed by 'play'" in line]
    assert [line.split("'")[1] for line in absorbed] == list(ABSORBED)
    assert any(line.startswith("сальто: 3/5 reached the teacher") for line in findings)
    # The other topics were not touched, so they say nothing.
    assert all(line.startswith("сальто:") for line in findings)


def test_a_pick_under_the_threshold_is_a_miss_not_a_finding(monkeypatch):
    """The gate reads `ROUTER_THRESHOLD` from the code it guards, not a copy."""
    monkeypatch.setenv("ROUTER_THRESHOLD", "0.7")
    assert run_gate(FakeEngine(confidence=0.65, routes=ABSORBED)) == []

    monkeypatch.setenv("ROUTER_THRESHOLD", "0.6")
    assert run_gate(FakeEngine(confidence=0.65, routes=ABSORBED)) != []


def test_the_batch_is_the_miners_not_the_gates(monkeypatch):
    """Raise `MINER_BATCH` past a topic's size and a clean run goes red."""
    monkeypatch.setenv("MINER_BATCH", "6")
    findings = run_gate(FakeEngine(pick="unknown"))
    assert len(findings) == len(gate.FOREIGN_TOPICS)
    assert all("5/5 reached the teacher, MINER_BATCH=6" in line for line in findings)


def test_every_topic_is_one_full_batch():
    """Fewer phrases than the batch and the gate could never be green."""
    for topic, phrases in gate.FOREIGN_TOPICS.items():
        assert len(set(phrases)) >= batch_size(), topic


def test_no_foreign_phrase_is_a_starter_example():
    """A phrase a skill lists is that skill's by the lookup — not a foreign command."""
    listed = {normalize(example) for skill in load_seed_skills() for example in skill.examples}
    for phrases in gate.FOREIGN_TOPICS.values():
        assert not listed & {normalize(text) for text in phrases}
