from datetime import timedelta

import pytest

from backend.actions import validate_plan
from backend.state import (
    START_ENERGY,
    START_FULLNESS,
    START_MOOD,
    apply_actions,
    read_state,
    utcnow,
)

# The seeded last_tick_at is a few microseconds older than the test's clock, so
# every absolute expectation is compared with a tolerance rather than exactly.
TOL = 0.01


def test_initial_state(conn):
    state = read_state(conn)
    assert state.mood == pytest.approx(START_MOOD, abs=TOL)
    assert state.energy == pytest.approx(START_ENERGY, abs=TOL)
    assert state.fullness == pytest.approx(START_FULLNESS, abs=TOL)
    assert state.face == "curious"


def test_decay_after_an_hour(conn):
    # Time is injected, never slept through.
    later = utcnow() + timedelta(minutes=60)
    state = read_state(conn, now=later)

    assert state.fullness == 0.0  # 60 - 1.0 * 60, clamped at 0
    assert state.energy == pytest.approx(START_ENERGY - 0.7 * 60, abs=TOL)
    # 70 - 0.5 * 60 = 40, then -10 because fullness dropped below 15.
    assert state.mood == pytest.approx(30.0, abs=TOL)


def test_decay_is_clamped_after_a_week(conn):
    state = read_state(conn, now=utcnow() + timedelta(days=7))
    assert (state.mood, state.energy, state.fullness) == (0.0, 0.0, 0.0)
    assert state.face == "sleepy"


def test_low_energy_forces_sleepy_face(conn):
    # 80 energy at 0.7/min crosses 15 just before 93 minutes.
    state = read_state(conn, now=utcnow() + timedelta(minutes=95))
    assert state.energy < 15
    assert state.face == "sleepy"


def test_decay_is_persisted_between_reads(conn):
    later = utcnow() + timedelta(minutes=30)
    first = read_state(conn, now=later)
    second = read_state(conn, now=later)
    assert first.to_dict() == second.to_dict()


def test_clock_going_backwards_does_not_raise_the_scales(conn):
    state = read_state(conn, now=utcnow() - timedelta(minutes=60))
    assert state.fullness == pytest.approx(START_FULLNESS, abs=TOL)


def test_apply_actions_uses_effects_table(conn):
    now = utcnow()
    before = read_state(conn, now=now)
    state = apply_actions(
        conn,
        validate_plan(
            [{"action": "eat", "args": {}}, {"action": "set_face", "args": {"face": "happy"}}]
        ),
        now=now,
    )
    assert state.fullness == before.fullness + 30
    assert state.energy == before.energy - 2
    assert state.face == "happy"


def test_effects_are_clamped(conn):
    now = utcnow()
    state = apply_actions(conn, validate_plan([{"action": "sleep", "args": {}}] * 4), now=now)
    assert state.energy == 100.0
