"""The two checks a candidate has to survive before the user ever sees it."""

import json

import pytest

from backend.brain.skill import Skill, load_seed_skills
from backend.miner.backtest import (
    backtest,
    check_overreach,
    check_regressions,
    min_match_rate,
)
from backend.miner.case import Case
from backend.miner.schema import MAX_DESCRIPTION_LEN, SkillDraft
from backend.state import RobotState

from .fakes import FakeEngine
from .trick_cluster import (
    LATER_TRICK,
    MID_STATE,
    SEED_CONTROLS,
    TEACHER_PLANS,
    TRICK_COMMANDS,
    TRICK_DRAFT,
    TRICK_ROUTES,
    draft_json,
)

TIRED = RobotState(mood=50.0, energy=5.0, fullness=50.0, face="sleepy")


@pytest.fixture()
def active():
    return load_seed_skills()


@pytest.fixture()
def candidate():
    return SkillDraft.model_validate(TRICK_DRAFT).to_skill()


@pytest.fixture()
def cases():
    return [
        Case(id=index + 1, user_text=command, state=MID_STATE, actions=TEACHER_PLANS[index])
        for index, command in enumerate(TRICK_COMMANDS)
    ]


def cases_that_miss() -> list[Case]:
    """A cluster the candidate cannot cover: the router sends these elsewhere."""
    return [
        Case(id=index + 1, user_text=f"покорми {index}", state=MID_STATE, actions=TEACHER_PLANS[0])
        for index in range(5)
    ]


def engine(steals: dict[str, str] | None = None) -> FakeEngine:
    """The library routing as it stands, optionally with a command the candidate steals."""
    return FakeEngine(routes={**TRICK_ROUTES, **(steals or {})})


def test_a_candidate_that_covers_its_cluster_is_publishable(active, candidate, cases):
    report = backtest(engine(), active, candidate, cases)
    assert report.matched == 5
    assert report.match_rate == pytest.approx(1.0)
    assert report.generalization == pytest.approx(1.0)
    assert report.agreement == pytest.approx(1.0)
    assert report.regression is None
    assert report.publishable


def test_a_different_plan_is_agreement_not_coverage(active, candidate, cases):
    """The router reaches the candidate, but a tired robot does something else.

    This is the split JEB-1547 turns on. Coverage is about routing, so it stays
    at 1.0; agreement drops to 0.0 and does not veto the proposal. Gating on the
    old combined number is what made the miner unpublishable on live data: the
    teacher improvises a slightly different plan every time, and nothing in this
    measure has a notion of "close".
    """
    tired = [Case(case.id, case.user_text, TIRED, case.actions) for case in cases]
    report = backtest(engine(), active, candidate, tired)
    assert report.match_rate == pytest.approx(1.0)
    assert report.agreement == pytest.approx(0.0)
    assert report.publishable


def test_a_candidate_the_router_misses_agrees_with_nothing(active, candidate):
    """Agreement is counted over routed cases only — an unrouted case cannot agree."""
    report = backtest(FakeEngine("unknown"), active, candidate, cases_that_miss())
    assert report.match_rate == pytest.approx(0.0)
    assert report.agreement == pytest.approx(0.0)


def test_wording_differences_do_not_break_agreement(active, candidate, cases):
    """Every teacher reply is phrased differently; only the primitives count."""
    assert len({json.dumps(case.actions, ensure_ascii=False) for case in cases}) == len(cases)
    assert backtest(engine(), active, candidate, cases).agreement == pytest.approx(1.0)


def test_a_cluster_the_router_sends_elsewhere_scores_zero(active, candidate):
    report = backtest(engine(), active, candidate, cases_that_miss())
    assert report.match_rate == pytest.approx(0.0)
    assert not report.publishable


def test_a_phrase_the_head_reaches_below_its_threshold_is_not_covered(active, candidate):
    """The threshold still decides every phrase the draft did not list."""
    unlisted = [Case(id=6, user_text=LATER_TRICK, state=MID_STATE, actions=TEACHER_PLANS[0])]
    confident = backtest(engine(), active, candidate, unlisted)
    shy = backtest(FakeEngine(routes=TRICK_ROUTES, confidence=0.4), active, candidate, unlisted)
    assert confident.match_rate == pytest.approx(1.0)
    assert shy.match_rate == pytest.approx(0.0)


def test_a_candidate_that_steals_an_active_command_is_refused(active, candidate, cases):
    """The regression check, and the whole reason it exists.

    `show_trick` covers its own cluster perfectly *and* drags "покорми" away from
    `feed` — stage 2 measured that appending an option does exactly this. A
    match rate cannot see it; a proposal published on match rate alone would
    have broken feeding.
    """
    thief = engine({"покорми": "show_trick"})
    report = backtest(thief, active, candidate, cases)
    assert report.match_rate == pytest.approx(1.0)
    assert report.regression is not None
    assert "покорми" in report.regression
    assert "feed" in report.regression
    assert not report.publishable


def test_a_control_phrase_falling_under_its_threshold_is_a_regression(active, candidate):
    shy = FakeEngine(routes=TRICK_ROUTES, confidence=0.5)
    assert "miss" in check_regressions(shy, [*active, candidate], active)


def test_the_control_set_is_the_first_example_of_every_active_skill(active, candidate):
    """So it grows with the library instead of being a hard-coded four."""
    probe = engine()
    check_regressions(probe, [*active, candidate], active)
    for skill in active:
        assert any(skill.examples[0] in prompt for prompt in probe.prompts)


def test_a_skill_without_examples_is_not_a_control(candidate):
    bare = Skill(
        id="bare",
        name="Bare",
        description="без примеров",
        rules=[{"when": {}, "actions": [{"action": "wave"}]}],
    )
    assert check_regressions(FakeEngine("unknown"), [bare, candidate], [bare]) is None


def test_the_candidate_is_tried_last_in_the_option_list(active, candidate):
    """Option order changes the answer, so the trial order must be the live one."""
    probe = engine()
    backtest(probe, active, candidate, [])
    assert list(probe.calls[0]["skill"]["criteria"]) == [
        "greet",
        "feed",
        "play",
        "sleep",
        "show_trick",
        "unknown",
    ]


def test_the_minimum_match_rate_is_configurable(monkeypatch):
    assert min_match_rate() == 0.8
    monkeypatch.setenv("MINER_MIN_MATCH", "0.5")
    assert min_match_rate() == 0.5


def test_an_over_long_description_never_reaches_the_backtest():
    """Measured, not stylistic: a long option label costs every skill accuracy."""
    long = "навык, который показывает пользователю очень красивый и длинный фокус"
    assert len(long) > MAX_DESCRIPTION_LEN
    with pytest.raises(ValueError):
        SkillDraft.model_validate_json(draft_json(description=long))


def test_a_draft_naming_an_action_outside_the_library_does_not_build():
    draft = SkillDraft.model_validate_json(
        draft_json(rules=[{"when_state": "", "when_band": "", "actions": [{"action": "fly"}]}])
    )
    with pytest.raises(ValueError):
        draft.to_skill()


def test_a_draft_whose_last_rule_is_conditional_does_not_build():
    draft = SkillDraft.model_validate_json(
        draft_json(
            rules=[
                {
                    "when_state": "energy",
                    "when_band": "low",
                    "actions": [{"action": "say", "text": "Устал"}],
                }
            ]
        )
    )
    with pytest.raises(ValueError):
        draft.to_skill()


@pytest.mark.parametrize("bad_id", ["Показать Фокус", "show trick", "1trick", ""])
def test_an_id_that_is_not_a_latin_label_is_refused(bad_id):
    with pytest.raises(ValueError):
        SkillDraft.model_validate_json(draft_json(id=bad_id))


def test_a_when_naming_something_that_is_not_a_state_scale_is_refused():
    with pytest.raises(ValueError):
        SkillDraft.model_validate_json(
            draft_json(
                rules=[
                    {
                        "when_state": "weather",
                        "when_band": "low",
                        "actions": [{"action": "wave"}],
                    },
                    {"when_state": "", "when_band": "", "actions": [{"action": "wave"}]},
                ]
            )
        )


def test_a_built_skill_is_marked_as_mined(candidate):
    assert (candidate.origin, candidate.status) == ("mined", "active")
    assert candidate.questions == {}


def test_a_candidate_that_takes_another_cluster_is_not_published(active, candidate, cases):
    """JEB-1548: `show_trick` pulled "покажи сальто" at 0.78 with every control green."""
    outsider = Case(id=99, user_text="покажи сальто", state=MID_STATE, actions=TEACHER_PLANS[0])
    report = backtest(
        engine({"покажи сальто": "show_trick"}), active, candidate, cases, [outsider]
    )
    assert report.match_rate == pytest.approx(1.0)
    assert report.regression is None
    assert report.overreach == '"покажи сальто" (another cluster) -> show_trick @ 0.90'
    assert not report.publishable


def test_the_rest_of_the_pool_is_left_alone(active, candidate, cases):
    outsider = Case(id=99, user_text="покорми", state=MID_STATE, actions=TEACHER_PLANS[0])
    report = backtest(engine(), active, candidate, cases, [outsider])
    assert report.overreach is None
    assert report.publishable


def test_coverage_counts_the_examples_lookup_because_production_does(active, candidate, cases):
    """JEB-1562, and the whole point of the number.

    The generator copies the cluster into `examples` word for word, so after
    acceptance every one of these five phrases routes to the candidate at 1.0
    through step 0 and the head is never asked. A blind head does not change that
    — and the old `use_examples=False` measurement said it did, which is how a
    cluster that production covers completely got refused.
    """
    # Blind on the cluster, and the seed controls still route, so the only thing
    # under test here is check 1.
    blind = FakeEngine(routes=SEED_CONTROLS, confidence=0.99)
    report = backtest(blind, active, candidate, cases)
    assert report.match_rate == pytest.approx(1.0)
    assert report.publishable
    # And the head's own answer is reported, not thrown away: it is 0.0 here, which
    # is what a *sixth* phrasing of this command would get.
    assert report.generalization == pytest.approx(0.0)


def test_generalization_is_the_head_alone(active, candidate, cases):
    """Two phrasings the head reaches out of five, with coverage flat at 1.0."""
    head = FakeEngine(
        routes={**SEED_CONTROLS, "покажи фокус": "show_trick", "удиви фокусом": "show_trick"}
    )
    report = backtest(head, active, candidate, cases)
    assert report.match_rate == pytest.approx(1.0)
    assert report.generalization == pytest.approx(0.4)


def test_a_phrase_the_draft_forgot_to_list_falls_back_to_the_head(active, candidate, cases):
    """Which is the tolerance MINER_MIN_MATCH buys: one such phrase in five."""
    later = Case(id=6, user_text=LATER_TRICK, state=MID_STATE, actions=TEACHER_PLANS[0])
    blind = FakeEngine(routes=SEED_CONTROLS, confidence=0.99)
    report = backtest(blind, active, candidate, [*cases, later])
    assert report.matched == 5
    assert report.match_rate == pytest.approx(5 / 6)
    assert report.publishable


def test_a_phrase_an_active_skill_already_lists_is_not_the_candidates(active, candidate):
    """Step 0 is first-in-registry-order and a mined skill is appended, so the
    older skill keeps the phrase — coverage reports what production will do even
    when the head would hand it over."""
    stolen = [Case(id=1, user_text="привет", state=MID_STATE, actions=TEACHER_PLANS[0])]
    report = backtest(engine({"привет": "show_trick"}), active, candidate, stolen)
    assert report.match_rate == pytest.approx(0.0)
    assert report.generalization == pytest.approx(1.0)
    assert not report.publishable


def test_the_cluster_costs_one_forward_pass_per_case(active, candidate, cases):
    """Step 0 answers coverage without the head and a mined skill has no
    `questions`, so the only pass a listed case pays for is `generalization`."""
    probe = engine()
    backtest(probe, active, candidate, cases)
    assert len(probe.calls) == len(cases) + len(active)


def test_a_candidate_that_lost_on_match_rate_never_pays_for_the_pool_scan(active, candidate):
    """The pool has no LIMIT and does not shrink for a rejected candidate (JEB-1548 review)."""
    outsiders = [
        Case(id=index, user_text=f"команда {index}", state=MID_STATE, actions=[])
        for index in range(40)
    ]
    thin = engine()
    report = backtest(thin, active, candidate, cases_that_miss(), outsiders)

    assert report.match_rate < min_match_rate()
    assert report.overreach is None
    assert not report.publishable
    assert not any("команда" in prompt for prompt in thin.prompts)


def test_a_regression_also_stops_the_pool_scan(active, candidate, cases):
    """match_rate is a clean 1.0 here, so the regression is what does the stopping."""
    outsiders = [Case(id=99, user_text="покажи сальто", state=MID_STATE, actions=[])]
    greedy = engine({"привет": "show_trick", "покажи сальто": "show_trick"})
    report = backtest(greedy, active, candidate, cases, outsiders)

    assert report.match_rate == pytest.approx(1.0)
    assert report.regression is not None
    assert report.overreach is None
    assert not any("сальто" in prompt for prompt in greedy.prompts)


def test_the_controls_cost_one_forward_pass_each_not_two(active, candidate):
    """`pick_skill`, not `route`: they read the skill id and never need the plan."""
    probe = engine()
    check_overreach(probe, [*active, candidate], candidate, [
        Case(id=1, user_text="покорми", state=MID_STATE, actions=[])
    ])
    assert len(probe.calls) == 1
