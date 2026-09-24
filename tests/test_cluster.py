"""Clustering and the pool it reads from — no weights, no network."""

import inspect
import json

import numpy as np
import pytest

from backend.brain.engine import EmbeddingsUnavailable
from backend.miner.case import is_mineable, load_pool, pool_size
from backend.miner.cluster import (
    DEFAULT_SIM,
    components,
    cosine_matrix,
    group_texts,
    min_cluster_size,
    sim_threshold,
)
from backend.miner.run import mining_due
from backend.state import RobotState

from .fakes import FakeEngine, hash_vector
from .trick_cluster import TEACHER_PLANS, TRICK_COMMANDS, TRICK_EMBEDDINGS, fill_pool

FAR = [0.0, 0.0, 1.0]


@pytest.fixture()
def trick_engine():
    return FakeEngine(embeddings=TRICK_EMBEDDINGS)


def test_similar_commands_land_in_one_cluster(trick_engine):
    assert group_texts(trick_engine, TRICK_COMMANDS) == [[0, 1, 2, 3, 4]]


def test_an_unrelated_command_stays_on_its_own(trick_engine):
    texts = [*TRICK_COMMANDS, "расскажи про квантовую физику"]
    assert group_texts(trick_engine, texts) == [[0, 1, 2, 3, 4], [5]]


def test_the_threshold_is_configurable(monkeypatch):
    """These two sit at cos = 0.9: together by default, apart at 0.95."""
    engine = FakeEngine(embeddings={"a": [1.0, 0.0], "b": [0.9, np.sqrt(1 - 0.9**2)]})
    assert group_texts(engine, ["a", "b"]) == [[0, 1]]
    monkeypatch.setenv("MINER_SIM", "0.95")
    assert group_texts(engine, ["a", "b"]) == [[0], [1]]


def test_the_default_sits_in_the_measured_band(monkeypatch):
    """0.88, calibrated on real Laya vectors in JEB-1509.

    Pinned because it is not a round number anybody would guess back: below
    ~0.87 the clusters come out mixed, past ~0.91 none reach the minimum size.
    """
    monkeypatch.delenv("MINER_SIM", raising=False)
    assert DEFAULT_SIM == 0.88
    assert sim_threshold() == DEFAULT_SIM


def test_single_link_chains_through_a_middle_point():
    """A and C are far apart but both close to B — single-link joins all three.

    Three unit vectors 25° apart: neighbours sit at cos 0.91, the ends at 0.64.
    """
    angles = {"a": 0.0, "b": 25.0, "c": 50.0}
    engine = FakeEngine(
        embeddings={
            name: [np.cos(np.radians(angle)), np.sin(np.radians(angle))]
            for name, angle in angles.items()
        }
    )
    assert group_texts(engine, ["a", "b", "c"]) == [[0, 1, 2]]


def test_components_are_returned_in_input_order():
    similarity = np.array([[1.0, 0.0, 0.9], [0.0, 1.0, 0.0], [0.9, 0.0, 1.0]])
    assert components(similarity, 0.75) == [[0, 2], [1]]


def test_a_zero_vector_is_similar_to_nothing():
    similarity = cosine_matrix([[0.0, 0.0], [1.0, 0.0]])
    assert similarity[0, 1] == pytest.approx(0.0)


def test_hash_vectors_never_reach_the_threshold():
    """The fake's default vectors must not manufacture a cluster by accident."""
    texts = ["привет", "покорми", "поиграй", "пора спать", "станцуй", "прыгни"]
    similarity = cosine_matrix([hash_vector(text) for text in texts])
    off_diagonal = similarity[~np.eye(len(texts), dtype=bool)]
    assert off_diagonal.max() < 0.75


def test_the_teacher_groups_and_the_vectors_are_never_touched():
    """JEB-1548 inverted this: the teacher is the primary grouper, not the plan B."""
    engine = FakeEngine(embeddings=TRICK_EMBEDDINGS)
    calls = []

    def grouper(texts):
        calls.append(texts)
        return [[0, 1], [2]]

    assert group_texts(engine, ["a", "b", "c"], grouper) == [[0, 1], [2]]
    assert calls == [["a", "b", "c"]]
    assert engine.embedded == []


def test_a_teacher_that_grouped_nothing_falls_back_to_the_vectors(trick_engine):
    """`SkillGenerator.group` swallows its own failures and answers `[]`."""
    assert group_texts(trick_engine, TRICK_COMMANDS, lambda texts: []) == [[0, 1, 2, 3, 4]]
    assert trick_engine.embedded == [TRICK_COMMANDS]


def test_neither_grouper_leaves_the_pool_alone():
    class NoEmbeddings(FakeEngine):
        def embed(self, texts):
            raise EmbeddingsUnavailable("laya has no embed_fn_from_agent")

    assert group_texts(NoEmbeddings(), ["a", "b", "c"], lambda texts: []) == []


def test_a_grouping_from_a_model_is_not_trusted():
    """Out-of-range indices are dropped and a repeated one is only used once."""

    def grouper(texts):
        return [[0, 99, 1], [1, 2], [-4]]

    engine = FakeEngine()
    engine.embed = lambda texts: []
    assert group_texts(engine, ["a", "b", "c"], grouper) == [[0, 1], [2]]


def test_no_grouper_and_no_embeddings_clusters_nothing():
    engine = FakeEngine()
    engine.embed = lambda texts: []
    assert group_texts(engine, ["a", "b", "c"]) == []


def test_the_pool_reads_back_the_command_and_the_plan(conn):
    fill_pool(conn)
    cases = load_pool(conn)
    assert [case.user_text for case in cases] == TRICK_COMMANDS
    assert cases[0].action_names == {"spin", "set_face", "say"}
    assert cases[0].state.energy == pytest.approx(60.0)


def test_a_failed_teacher_call_is_not_mining_material(conn):
    """Its plan is "I did not understand" — mining it would teach that."""
    fill_pool(conn, error="TimeoutError: boom")
    assert load_pool(conn) == []


def test_a_declined_answer_is_not_mining_material(conn):
    """The call worked; the teacher said it cannot do this.

    A real plan and a real reply, and the worst thing in the pool: mined, it
    becomes a skill that answers "я не умею" from Laya for ever and never lets
    the command reach the teacher again (JEB-1547).
    """
    fill_pool(conn, handled=False)
    assert load_pool(conn) == []


def test_a_row_written_before_handled_existed_is_still_mineable(conn):
    """A missing verdict is not a refusal — `pixel.db` predates the field."""
    conn.execute(
        "INSERT INTO teacher_log (state_json, actions_json, mined) VALUES (?, ?, 0)",
        (json.dumps({"user_text": "покажи фокус", "state": {}}), json.dumps(TEACHER_PLANS[0])),
    )
    conn.commit()
    assert [case.user_text for case in load_pool(conn)] == ["покажи фокус"]


def test_the_pool_size_counts_only_what_the_miner_could_use(conn):
    """Otherwise the every-MINER_BATCH-th trigger drifts and never recovers: the
    rows `load_pool` drops are never marked `mined`, so they sit in the count for
    ever and the trigger fires on a pool too small to cluster."""
    fill_pool(conn, ["какая погода", "закажи пиццу"], handled=False)
    assert pool_size(conn) == 0

    fill_pool(conn, TRICK_COMMANDS)
    assert pool_size(conn) == len(TRICK_COMMANDS) == len(load_pool(conn))


def test_the_trigger_reads_the_arrival_not_the_pool_level(conn, monkeypatch):
    """A pool parked on a multiple of MINER_BATCH must not fire on every message.

    `mined = 1` is set only when a proposal is saved, so a cluster that fails its
    backtest stays in the pool for good, and a refusal never enters it — the
    level stays true while the pool stands still. Reading it instead of the
    arrival puts one synchronous `generator.propose` round trip inside every
    later declined `/api/chat` (JEB-1547 review).
    """
    monkeypatch.setenv("MINER_BATCH", "5")
    ids = fill_pool(conn, TRICK_COMMANDS)
    assert pool_size(conn) == 5

    assert mining_due(conn, ids[-1]), "the fifth mineable case is what the trigger counts"

    refusal = fill_pool(conn, ["какая погода"], handled=False)[0]
    assert pool_size(conn) == 5, "a refusal does not move the pool"
    assert not mining_due(conn, refusal)

    broken = fill_pool(conn, ["закажи пиццу"], error="TimeoutError: boom")[0]
    assert not mining_due(conn, broken)


def test_a_mined_row_is_not_an_arrival(conn):
    """`is_mineable` answers about the pool, so a row already taken out is not in it."""
    ids = fill_pool(conn, TRICK_COMMANDS)
    assert is_mineable(conn, ids[0])
    conn.execute("UPDATE teacher_log SET mined = 1 WHERE id = ?", (ids[0],))
    conn.commit()
    assert not is_mineable(conn, ids[0])
    assert not is_mineable(conn, 9999)


def test_a_mined_row_is_out_of_the_pool(conn):
    ids = fill_pool(conn)
    conn.execute("UPDATE teacher_log SET mined = 1 WHERE id = ?", (ids[0],))
    conn.commit()
    assert [case.id for case in load_pool(conn)] == ids[1:]


def test_an_unreadable_row_is_skipped_not_fatal(conn):
    fill_pool(conn, ["покажи фокус"])
    conn.execute("INSERT INTO teacher_log (state_json, actions_json, mined) VALUES ('{', '[]', 0)")
    conn.execute(
        "INSERT INTO teacher_log (state_json, actions_json, mined) VALUES (?, '[]', 0)",
        (json.dumps({"user_text": "", "state": {}}),),
    )
    conn.commit()
    assert [case.user_text for case in load_pool(conn)] == ["покажи фокус"]


def test_a_row_without_a_state_falls_back_to_the_starting_one(conn):
    conn.execute(
        "INSERT INTO teacher_log (state_json, actions_json, mined) VALUES (?, ?, 0)",
        (json.dumps({"user_text": "станцуй"}), json.dumps([{"action": "dance", "args": {}}])),
    )
    conn.commit()
    assert load_pool(conn)[0].state == RobotState(70.0, 80.0, 60.0, "curious")


def test_min_cluster_size_is_configurable(monkeypatch):
    assert min_cluster_size() == 3
    monkeypatch.setenv("MINER_MIN_CLUSTER", "2")
    assert min_cluster_size() == 2


def test_laya_still_has_the_embedding_helper_we_call():
    """The mirror of `test_the_sdk_still_has_the_surface_we_call`, for Laya.

    `FakeEngine.embed` proves the miner *uses* vectors, never that the real
    package can produce them. If `embed_fn_from_agent` were renamed,
    `LayaEngine.embed` would raise, and the miner would quietly stop clustering
    locally and start paying Gemini to group the commands instead — with CI
    green and only a `log.warning` behind it.

    Skipped where `laya` is not installed (it pulls torch, which the rest of the
    suite deliberately keeps off the import path); CI installs both, so this is
    the one place the assertion actually runs. Importing and inspecting the
    helper downloads no weights — only calling it needs a loaded agent.
    """
    laya = pytest.importorskip("laya")

    helper = getattr(laya, "embed_fn_from_agent", None)
    assert helper is not None, "laya.embed_fn_from_agent is gone — see LayaEngine.embed"

    # One positional `agent`, and the rest defaulted: `LayaEngine.embed` passes
    # nothing else, so a newly required argument has to fail here, not live.
    parameters = list(inspect.signature(helper).parameters.values())
    assert parameters[0].name == "agent"
    assert all(p.default is not inspect.Parameter.empty for p in parameters[1:])
