"""The skill format is a gate, not a suggestion: a bad skill is refused, alone."""

import pytest

from backend.brain.skill import load_seed_skills, load_skills, parse_skill, seed_db

WAVE = {"action": "wave"}


def skill(**overrides) -> dict:
    payload = {
        "id": "demo",
        "name": "Demo",
        "description": "демо",
        "rules": [{"when": {}, "actions": [WAVE]}],
    }
    payload.update(overrides)
    return payload


def test_a_valid_skill_parses():
    assert parse_skill(skill()).id == "demo"


def test_an_action_outside_the_library_is_refused(caplog):
    payload = skill(rules=[{"when": {}, "actions": [{"action": "fly"}]}])
    assert parse_skill(payload) is None
    assert "demo" in caplog.text


def test_a_plan_longer_than_the_limit_is_refused():
    assert parse_skill(skill(rules=[{"when": {}, "actions": [WAVE] * 9}])) is None


def test_a_bad_face_argument_is_refused():
    actions = [{"action": "set_face", "args": {"face": "smug"}}]
    assert parse_skill(skill(rules=[{"when": {}, "actions": actions}])) is None


def test_the_last_rule_must_be_the_fallback():
    assert parse_skill(skill(rules=[{"when": {"energy": "low"}, "actions": [WAVE]}])) is None


def test_a_skill_needs_a_rule():
    assert parse_skill(skill(rules=[])) is None


def test_when_must_name_a_question_or_a_state_scale():
    rules = [{"when": {"weather": "rainy"}, "actions": [WAVE]}, {"when": {}, "actions": [WAVE]}]
    assert parse_skill(skill(rules=rules)) is None


def test_a_state_scale_takes_only_low_mid_high():
    rules = [{"when": {"energy": "empty"}, "actions": [WAVE]}, {"when": {}, "actions": [WAVE]}]
    assert parse_skill(skill(rules=rules)) is None


def test_a_noul_condition_takes_only_yes_or_no():
    payload = skill(
        questions={"hungry": {"type": "noul", "instructions": "голоден"}},
        rules=[{"when": {"hungry": "maybe"}, "actions": [WAVE]}, {"when": {}, "actions": [WAVE]}],
    )
    assert parse_skill(payload) is None


def test_a_choice_question_needs_object_criteria():
    questions = {"kind": {"type": "choice", "instructions": "какой", "criteria": ["a", "b"]}}
    assert parse_skill(skill(questions=questions)) is None


def test_a_score_question_needs_at_least_two_levels():
    questions = {"how": {"type": "score", "instructions": "как", "criteria": ["одно"]}}
    assert parse_skill(skill(questions=questions)) is None


def test_every_seed_skill_is_valid_and_answerable():
    skills = load_seed_skills()
    assert len(skills) == 4
    for item in skills:
        assert item.rules[-1].when == {}
        assert set(item.to_questions()) == set(item.questions)


@pytest.mark.parametrize("payload", [skill(), {"id": "broken"}])
def test_one_bad_skill_does_not_take_the_library_down(conn, payload):
    conn.execute(
        "INSERT INTO skills (id, json, status, origin, created_at) VALUES (?, ?, ?, ?, ?)",
        (
            "bad",
            '{"id": "bad", "rules": [{"when": {}, "actions": [{"action": "fly"}]}]}',
            "active",
            "seed",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    conn.commit()
    seed_db(conn)  # the table is not empty, so seeding is a no-op
    assert [item.id for item in load_skills(conn)] == []


def test_seed_db_fills_an_empty_table_once(conn):
    assert seed_db(conn) == 4
    assert seed_db(conn) == 0
    # Insertion order, not alphabetical: the router's option order is measured.
    assert [item.id for item in load_skills(conn, only_active=True)] == [
        "greet",
        "feed",
        "play",
        "sleep",
    ]


def test_status_comes_from_the_row_not_the_json(seeded):
    seeded.execute("UPDATE skills SET status = 'disabled' WHERE id = 'sleep'")
    seeded.commit()
    by_id = {item.id: item.status for item in load_skills(seeded)}
    assert by_id["sleep"] == "disabled"
    assert by_id["greet"] == "active"
