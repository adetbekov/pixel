"""Router behaviour, driven entirely by `FakeEngine` — no weights, no network."""

import pytest

from backend.brain.engine import NoulResult, ScoreResult
from backend.brain.router import UNKNOWN, RouterHit, RouterMiss, route
from backend.brain.skill import Skill, load_seed_skills, load_skills
from backend.state import RobotState

from .fakes import FakeEngine

RESTED = RobotState(mood=70.0, energy=80.0, fullness=50.0, face="curious")
TIRED = RobotState(mood=40.0, energy=10.0, fullness=50.0, face="sleepy")
STUFFED = RobotState(mood=70.0, energy=80.0, fullness=95.0, face="happy")


@pytest.fixture()
def skills():
    return load_seed_skills()


def names(outcome: RouterHit) -> list[str]:
    return [step["action"] for step in outcome.raw_plan]


def test_seed_skills_load_in_the_measured_order(skills):
    assert [skill.id for skill in skills] == ["greet", "feed", "play", "sleep"]


def test_confident_pick_runs_the_skill(skills):
    outcome = route(FakeEngine("greet", 0.91), skills, "привет", RESTED)
    assert isinstance(outcome, RouterHit)
    assert outcome.skill_id == "greet"
    assert outcome.confidence == pytest.approx(0.91)
    assert names(outcome) == ["wave", "set_face", "say"]


def test_unknown_is_a_miss(skills):
    outcome = route(FakeEngine(UNKNOWN, 0.95), skills, "расскажи про квантовую физику", RESTED)
    assert outcome == RouterMiss(0.95)


def test_below_threshold_is_a_miss(skills):
    """A confident-looking label still loses to the threshold."""
    outcome = route(FakeEngine("play", 0.59), skills, "покажи фокус", RESTED)
    assert outcome == RouterMiss(0.59)


def test_threshold_is_per_skill(skills):
    picky = [skill.model_copy(update={"threshold": 0.95}) for skill in skills]
    assert isinstance(route(FakeEngine("greet", 0.9), skills, "привет", RESTED), RouterHit)
    assert isinstance(route(FakeEngine("greet", 0.9), picky, "привет", RESTED), RouterMiss)


def test_no_skills_is_a_miss():
    assert route(FakeEngine("greet", 0.99), [], "привет", RESTED) == RouterMiss(0.0)


def test_state_rule_wins_over_the_question(skills):
    """A tired robot refuses to play even when the engine shouts "бурно"."""
    engine = FakeEngine("play", 0.97, {"intensity": ScoreResult(2.0, 0.9, {"2": 1.0})})
    outcome = route(engine, skills, "поиграем", TIRED)
    assert names(outcome) == ["set_face", "say"]
    assert outcome.raw_plan[0]["args"]["face"] == "sleepy"


def test_first_matching_rule_wins(skills):
    engine = FakeEngine("play", 0.97, {"intensity": ScoreResult(1.8, 0.8, {"2": 0.8})})
    assert names(route(engine, skills, "побесимся", RESTED)) == [
        "jump",
        "spin",
        "dance",
        "set_face",
        "say",
    ]

    calm = FakeEngine("play", 0.97, {"intensity": ScoreResult(0.4, 0.8, {"0": 0.8})})
    assert names(route(calm, skills, "поиграем", RESTED)) == ["jump", "set_face", "say"]


def test_noul_condition(skills):
    yes = FakeEngine("feed", 0.9, {"wants_food": NoulResult(0.88, 0.88)})
    assert names(route(yes, skills, "покорми", RESTED)) == ["eat", "set_face", "say"]

    no = FakeEngine("feed", 0.9, {"wants_food": NoulResult(0.2, 0.8)})
    assert names(route(no, skills, "а ты ел?", RESTED)) == ["eat", "say"]


def test_full_robot_refuses_food(skills):
    engine = FakeEngine("feed", 0.9, {"wants_food": NoulResult(0.99, 0.99)})
    outcome = route(engine, skills, "покорми", STUFFED)
    assert outcome.raw_plan[0]["args"]["face"] == "angry"


def test_a_skill_without_questions_costs_one_pass(skills):
    engine = FakeEngine("greet", 0.9)
    route(engine, skills, "привет", RESTED)
    assert len(engine.calls) == 1


def test_a_skill_with_questions_costs_two_passes_and_batches_them(skills):
    engine = FakeEngine("play", 0.9, {"intensity": ScoreResult(1.0, 0.8, {"1": 0.8})})
    route(engine, skills, "поиграем", RESTED)
    assert len(engine.calls) == 2
    assert list(engine.calls[1]) == ["intensity"]


def test_router_offers_every_skill_in_order_plus_unknown(skills):
    """Option order changes the answer, so it must be the library's, not a set's."""
    engine = FakeEngine("greet", 0.9)
    route(engine, skills, "привет", RESTED)
    assert list(engine.calls[0]["skill"]["criteria"]) == [
        "greet",
        "feed",
        "play",
        "sleep",
        UNKNOWN,
    ]


def test_a_missing_answer_falls_through_to_the_fallback(skills):
    """The engine dropped the question, so its rule must not fire."""
    outcome = route(FakeEngine("feed", 0.9), skills, "покорми", RESTED)
    assert names(outcome) == ["eat", "say"]


def test_disabled_skills_are_not_routed_to(seeded):
    seeded.execute("UPDATE skills SET status = 'disabled' WHERE id = 'play'")
    seeded.commit()
    active = load_skills(seeded, only_active=True)
    assert "play" not in {skill.id for skill in active}
    assert route(FakeEngine("play", 0.99), active, "поиграем", RESTED) == RouterMiss(0.99)


def test_an_unrouteable_label_is_a_miss(skills):
    """The engine answered with something that is not a skill id."""
    assert route(FakeEngine("fly", 0.99), skills, "лети", RESTED) == RouterMiss(0.99)


def test_long_descriptions_are_trimmed():
    skill = Skill(
        id="verbose",
        name="Verbose",
        description="ц" * 500,
        rules=[{"when": {}, "actions": [{"action": "wave"}]}],
    )
    engine = FakeEngine("verbose", 0.9)
    route(engine, [skill], "привет", RESTED)
    assert len(engine.calls[0]["skill"]["criteria"]["verbose"]) == 120
