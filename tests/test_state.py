from datetime import timedelta

import pytest

from backend.actions import validate_plan
from backend.state import (
    START_ENERGY,
    START_FULLNESS,
    START_MOOD,
    RobotState,
    apply_actions,
    decay,
    read_state,
    utcnow,
    write_state,
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
    # 70 - 0.5 * 60 = 40, minus 1.0 for each of the 15 minutes spent below a
    # fullness of 15 (60 fullness at 1.0/min crosses it after 45 minutes).
    assert state.mood == pytest.approx(25.0, abs=TOL)


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


@pytest.mark.parametrize("fullness", [10.0, 20.0, 60.0])
def test_decay_is_additive_over_any_split(fullness):
    """The whole point: splitting a span must not cost extra mood.

    `fullness=20` is the interesting one — the hunger threshold is crossed in
    the middle of the span, so a per-step penalty would diverge here even if the
    already-hungry case happened to line up.
    """
    whole = decay(RobotState(70.0, 80.0, fullness, "curious"), 1.5)

    split = RobotState(70.0, 80.0, fullness, "curious")
    for _ in range(6):
        decay(split, 0.25)

    # to_dict rounds to 0.1, which is the tolerance the invariant is stated at.
    assert split.to_dict() == whole.to_dict()


def test_decay_is_additive_across_the_hunger_threshold():
    whole = decay(RobotState(70.0, 80.0, 20.0, "curious"), 30.0)

    split = RobotState(70.0, 80.0, 20.0, "curious")
    for _ in range(30):
        decay(split, 1.0)

    assert split.mood == pytest.approx(whole.mood, abs=0.1)


def test_polling_frequency_does_not_change_the_state(conn):
    """Regression for JEB-1569: six 15-second reads == one 90-second read.

    Driven through `read_state` with an injected clock, not through `decay`, so
    it also pins the invariant that a read is not a write.
    """
    start = utcnow()
    write_state(conn, RobotState(70.0, 80.0, 10.0, "curious"), start)

    polled = None
    for step in range(1, 7):
        polled = read_state(conn, now=start + timedelta(seconds=15 * step))

    once = read_state(conn, now=start + timedelta(seconds=90))

    assert polled.to_dict() == once.to_dict()
