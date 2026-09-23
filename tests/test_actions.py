import pytest

from backend.actions import ACTIONS, InvalidPlan, validate_plan


def test_library_has_exactly_eight_primitives():
    assert set(ACTIONS) == {
        "jump",
        "dance",
        "sleep",
        "eat",
        "spin",
        "wave",
        "say",
        "set_face",
    }


def test_unknown_action_is_dropped_not_raised():
    plan = validate_plan([{"action": "launch_rocket", "args": {}}, {"action": "jump", "args": {}}])
    assert [a.action for a in plan] == ["jump"]


def test_say_text_is_truncated_to_200_chars():
    plan = validate_plan([{"action": "say", "args": {"text": "я" * 500}}])
    assert len(plan[0].args["text"]) == 200


def test_say_without_text_and_bad_face_are_dropped():
    plan = validate_plan(
        [
            {"action": "say", "args": {}},
            {"action": "set_face", "args": {"face": "smug"}},
            {"action": "set_face", "args": {"face": "sad"}},
        ]
    )
    assert [a.to_dict() for a in plan] == [{"action": "set_face", "args": {"face": "sad"}}]


def test_malformed_steps_are_dropped():
    assert validate_plan(["dance", None, 42, {"args": {}}]) == []


def test_plan_longer_than_eight_raises():
    with pytest.raises(InvalidPlan):
        validate_plan([{"action": "jump", "args": {}}] * 9)


def test_non_list_plan_raises():
    with pytest.raises(InvalidPlan):
        validate_plan({"action": "jump"})
