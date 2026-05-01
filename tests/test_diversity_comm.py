"""Tests for diversity-preserving communication system."""
import pytest
import math
from ouroboros.agents.diversity_comm import (
    behavioral_fingerprint, jaccard_similarity, behavioral_diversity,
    herding_index, build_diverse_population, DiverseHint,
    AgentCommState, DiversityPreservingHub,
)
from ouroboros.synthesis.expr_node import ExprNode, NodeType


def make_const(v):
    return ExprNode(NodeType.CONST, value=v)


class TestBehavioralFingerprint:
    def test_const_expr_fingerprint(self):
        expr = make_const(5)
        fp = behavioral_fingerprint(expr, [5]*20)
        assert len(fp) == 20
        assert all(v == 5 for v in fp)

    def test_none_expr_returns_zeros(self):
        fp = behavioral_fingerprint(None, [1]*10)
        assert all(v == 0 for v in fp)

    def test_different_exprs_different_fps(self):
        e1 = make_const(3)
        e2 = make_const(7)
        fp1 = behavioral_fingerprint(e1, [3]*20)
        fp2 = behavioral_fingerprint(e2, [3]*20)
        assert fp1 != fp2

    def test_same_behavior_same_fp(self):
        e1 = make_const(5)
        e2 = make_const(5)
        fp1 = behavioral_fingerprint(e1, [5]*20)
        fp2 = behavioral_fingerprint(e2, [5]*20)
        assert fp1 == fp2


class TestJaccardSimilarity:
    def test_identical_tuples(self):
        fp = (1, 2, 3, 4, 5)
        assert jaccard_similarity(fp, fp) == pytest.approx(1.0)

    def test_disjoint_tuples(self):
        fp1 = (1, 2, 3)
        fp2 = (4, 5, 6)
        # Disjoint when treated as sets of (index, value) pairs
        sim = jaccard_similarity(fp1, fp2)
        assert sim < 1.0

    def test_empty_tuples(self):
        assert jaccard_similarity((), ()) == pytest.approx(1.0) or \
               jaccard_similarity((), ()) == pytest.approx(0.0)

    def test_partial_overlap(self):
        fp1 = (1, 2, 3, 4)
        fp2 = (1, 2, 5, 6)
        sim = jaccard_similarity(fp1, fp2)
        assert 0.0 < sim < 1.0


class TestDiversity:
    def test_all_identical_zero_diversity(self):
        fps = [(1, 2, 3)] * 5
        assert behavioral_diversity(fps) == pytest.approx(0.0)

    def test_all_different_high_diversity(self):
        fps = [(i, i*2, i*3) for i in range(5)]
        d = behavioral_diversity(fps)
        assert d > 0.0

    def test_herding_all_same(self):
        fps = [(1, 2, 3)] * 5
        assert herding_index(fps) == pytest.approx(1.0)

    def test_herding_all_different(self):
        fps = [(i, i+1, i+2) for i in range(5)]
        h = herding_index(fps)
        assert h == pytest.approx(0.0)

    def test_herding_half_same(self):
        fps = [(1, 2)] * 3 + [(3, 4)] * 3
        h = herding_index(fps)
        assert 0.0 < h < 1.0


class TestBuildDiversePopulation:
    def _exprs_and_costs(self, n):
        return [(make_const(float(i)), float(i * 10)) for i in range(1, n+1)]

    def test_returns_diverse_hint(self):
        pairs = self._exprs_and_costs(10)
        obs = [5] * 20
        hint = build_diverse_population(pairs, obs, max_size=5)
        assert isinstance(hint, DiverseHint)
        assert hint.n_expressions <= 5

    def test_respects_max_size(self):
        pairs = self._exprs_and_costs(20)
        obs = [5] * 20
        hint = build_diverse_population(pairs, obs, max_size=3)
        assert hint.n_expressions <= 3

    def test_best_cost_first(self):
        pairs = self._exprs_and_costs(10)
        obs = [5] * 20
        hint = build_diverse_population(pairs, obs, max_size=5)
        if hint.n_expressions >= 2:
            assert hint.mdl_costs[0] <= hint.mdl_costs[1]

    def test_empty_input(self):
        hint = build_diverse_population([], [5]*20, max_size=5)
        assert hint.n_expressions == 0


class TestAgentCommState:
    def test_initial_threshold(self):
        state = AgentCommState("A0", base_threshold=0.10)
        assert state.adoption_threshold == pytest.approx(0.10)

    def test_threshold_decreases_when_stuck(self):
        state = AgentCommState("A0", base_threshold=0.10, stuck_threshold=3)
        state.rounds_since_improvement = 10  # very stuck
        assert state.adoption_threshold < 0.10

    def test_should_adopt_large_improvement(self):
        state = AgentCommState("A0")
        state.current_best_cost = 100.0
        # 20% improvement — above 10% threshold
        assert state.should_adopt(80.0)

    def test_should_not_adopt_small_improvement(self):
        state = AgentCommState("A0")
        state.current_best_cost = 100.0
        # 5% improvement — below 10% threshold
        assert not state.should_adopt(95.0)

    def test_always_adopt_if_no_current(self):
        state = AgentCommState("A0")
        assert state.should_adopt(50.0)

    def test_update_resets_stuck_counter(self):
        state = AgentCommState("A0")
        state.current_best_cost = 100.0
        state.rounds_since_improvement = 5
        state.update(80.0)  # big improvement
        assert state.rounds_since_improvement == 0

    def test_update_increments_stuck(self):
        state = AgentCommState("A0")
        state.current_best_cost = 100.0
        state.update(99.5)  # tiny improvement
        assert state.rounds_since_improvement > 0


class TestDiversityPreservingHub:
    def test_initialization(self):
        hub = DiversityPreservingHub(n_agents=6)
        assert hub.n_agents == 6
        assert hub.mean_herding_index == 0.0

    def test_end_round_updates_metrics(self):
        hub = DiversityPreservingHub(n_agents=4)
        fps = [(1, 2), (3, 4), (1, 2), (5, 6)]
        hub.end_round(fps)
        assert hub.current_herding_index > 0.0

    def test_receive_hints_returns_none_initially(self):
        hub = DiversityPreservingHub(n_agents=4)
        result = hub.receive_hints("A0", [5]*20)
        assert result is None

    def test_summary_is_string(self):
        hub = DiversityPreservingHub(n_agents=4)
        fps = [(1, 2), (3, 4)]
        hub.end_round(fps)
        s = hub.summary()
        assert isinstance(s, str) and "herding" in s.lower()

    def test_herding_history_grows(self):
        hub = DiversityPreservingHub(n_agents=4)
        for _ in range(5):
            hub.end_round([(1, 2), (3, 4), (1, 2), (5, 6)])
        assert len(hub._herding_history) == 5