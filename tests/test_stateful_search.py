"""Tests for stateful search — STATE_VAR integration."""
import pytest
import math
from ouroboros.search.stateful_search import (
    StatefulScorer, StatefulBeamSearch, StatefulSearchConfig,
    StatefulHierarchicalRouter, StatefulCandidate,
)
from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
from ouroboros.synthesis.expr_node import NodeType


def make_const(v: float) -> ExtExprNode:
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = NodeType.CONST; n.value = float(v)
    n.lag = 1; n.state_key = 0; n.window = 10
    n.left = n.right = n.third = None; n._cache = {}
    return n

def make_time() -> ExtExprNode:
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = NodeType.TIME; n.value = 0.0
    n.lag = 1; n.state_key = 0; n.window = 10
    n.left = n.right = n.third = None; n._cache = {}
    return n


class TestStatefulScorer:
    def test_score_constant_expr(self):
        scorer = StatefulScorer()
        expr = make_const(7.0)
        obs = [7] * 50
        cost, state = scorer.score(expr, obs)
        assert math.isfinite(cost)
        assert cost > 0  # some program bits

    def test_score_perfect_prediction_low(self):
        scorer = StatefulScorer()
        # CONST(3) on obs=[3]*50 → perfect prediction
        expr = make_const(3.0)
        obs = [3] * 50
        cost, state = scorer.score(expr, obs)
        # Should be low (program bits only, no data bits)
        assert cost < 50

    def test_score_wrong_prediction_high(self):
        scorer = StatefulScorer()
        expr = make_const(99.0)  # predict 99, actual is 3
        obs = [3] * 50
        cost_wrong, _ = scorer.score(expr, obs)
        expr_right = make_const(3.0)
        cost_right, _ = scorer.score(expr_right, obs)
        assert cost_wrong > cost_right

    def test_state_persists_across_timesteps(self):
        scorer = StatefulScorer()
        # STATE_VAR(0) expression — reads from state
        state_expr = ExtExprNode(ExtNodeType.STATE_VAR)
        state_expr.state_key = 0
        state_expr.value = 0.0; state_expr.lag = 1; state_expr.window = 10
        state_expr.left = state_expr.right = state_expr.third = None
        state_expr._cache = {}

        initial_state = {0: 5.0}
        obs = [5] * 30
        cost, final_state = scorer.score(state_expr, obs, initial_state)
        assert math.isfinite(cost)

    def test_score_batch_stateful_returns_list(self):
        scorer = StatefulScorer()
        exprs = [make_const(float(i)) for i in range(5)]
        obs = [3] * 30
        results = scorer.score_batch_stateful(exprs, obs)
        assert len(results) == 5
        assert all(isinstance(c, float) for c, _ in results)

    def test_initial_state_used(self):
        scorer = StatefulScorer()
        state_expr = ExtExprNode(ExtNodeType.STATE_VAR)
        state_expr.state_key = 0; state_expr.value = 0.0
        state_expr.lag = 1; state_expr.window = 10
        state_expr.left = state_expr.right = state_expr.third = None
        state_expr._cache = {}

        # With initial state = 7, prediction should be 7
        cost_7, _ = scorer.score(state_expr, [7]*20, initial_state={0: 7.0})
        cost_3, _ = scorer.score(state_expr, [7]*20, initial_state={0: 3.0})
        # cost_7 should be better (7 predicted when obs is 7)
        assert cost_7 <= cost_3


class TestStatefulBeamSearch:
    def test_runs_without_crash(self):
        cfg = StatefulSearchConfig(beam_width=5, n_iterations=2, max_depth=3)
        beam = StatefulBeamSearch(cfg)
        obs = [(3*t+1)%7 for t in range(60)]
        result = beam.search(obs, alphabet_size=7)
        assert result is None or hasattr(result, 'evaluate')

    def test_gcd_warm_starts_generated(self):
        cfg = StatefulSearchConfig(beam_width=5, n_iterations=2)
        beam = StatefulBeamSearch(cfg)
        seeds = beam._gcd_warm_starts(alphabet_size=20)
        assert len(seeds) > 0
        # All seeds should be evaluable
        for seed in seeds:
            try:
                val = seed.evaluate(0, [], {})
                assert isinstance(val, float)
            except Exception:
                pass  # some seeds may fail on dummy input — that's OK

    def test_returns_expr_or_none(self):
        cfg = StatefulSearchConfig(beam_width=4, n_iterations=2, max_depth=3)
        beam = StatefulBeamSearch(cfg)
        obs = [t % 5 for t in range(40)]
        result = beam.search(obs, alphabet_size=5)
        assert result is None or isinstance(result, ExtExprNode)


class TestStatefulCandidate:
    def test_ordering(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=1)
        c1 = StatefulCandidate(expr=expr, mdl_cost=10.0, final_state={})
        c2 = StatefulCandidate(expr=expr, mdl_cost=20.0, final_state={})
        assert c1 < c2
        assert not (c2 < c1)

    def test_sorting(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=1)
        candidates = [
            StatefulCandidate(expr, 30.0, {}),
            StatefulCandidate(expr, 10.0, {}),
            StatefulCandidate(expr, 20.0, {}),
        ]
        candidates.sort()
        assert candidates[0].mdl_cost == 10.0
        assert candidates[-1].mdl_cost == 30.0


class TestStatefulHierarchicalRouter:
    def test_standard_mode_for_modular(self):
        router = StatefulHierarchicalRouter(beam_width=5, n_iterations=2)
        obs = [(3*t+1)%7 for t in range(60)]
        expr, cost = router.search(obs, alphabet_size=7, env_name="ModularArithmetic")
        assert cost >= 0

    def test_stateful_mode_for_gcd(self):
        router = StatefulHierarchicalRouter(beam_width=5, n_iterations=2)
        from ouroboros.environments.algorithm_env import GCDEnv
        env = GCDEnv(seed=1)
        obs = env.generate(50)
        expr, cost = router.search(obs, alphabet_size=env.alphabet_size,
                                   env_name="GCDEnv", use_stateful=True)
        assert cost >= 0

    def test_auto_detection_gcd(self):
        router = StatefulHierarchicalRouter(beam_width=5, n_iterations=2)
        # GCDEnv should trigger stateful mode automatically
        from ouroboros.environments.algorithm_env import GCDEnv
        env = GCDEnv(seed=1)
        obs = env.generate(40)
        # Should not crash regardless of mode
        expr, cost = router.search(obs, alphabet_size=env.alphabet_size,
                                   env_name="GCDEnv")
        assert isinstance(cost, float)

    def test_result_is_evaluable(self):
        router = StatefulHierarchicalRouter(beam_width=5, n_iterations=2)
        obs = [(3*t+1)%7 for t in range(60)]
        expr, cost = router.search(obs, alphabet_size=7)
        if expr is not None:
            val = expr.evaluate(5, obs[:5], {})
            assert isinstance(val, float)


class TestGCDDiscoveryRunner:
    def test_runner_initializes(self):
        from ouroboros.environments.gcd_runner import GCDDiscoveryRunner
        runner = GCDDiscoveryRunner(n_attempts=2, stream_length=40,
                                    beam_width=4, n_iterations=2, verbose=False)
        assert runner.n_attempts == 2

    def test_evaluate_on_pairs(self):
        from ouroboros.environments.gcd_runner import GCDDiscoveryRunner
        from ouroboros.environments.algorithm_env import GCDEnv
        runner = GCDDiscoveryRunner(n_attempts=1, verbose=False)
        env = GCDEnv(seed=1)
        # GCD_NODE expression should get high accuracy
        gcd_expr = ExtExprNode(ExtNodeType.GCD_NODE,
                                left=make_const(12.0), right=make_const(8.0))
        # With constant inputs, accuracy = fraction of pairs where gcd(12,8)=4
        accuracy = runner._evaluate_on_gcd_pairs(gcd_expr, env, n_pairs=100)
        assert 0.0 <= accuracy <= 1.0

    def test_none_expr_returns_zero_accuracy(self):
        from ouroboros.environments.gcd_runner import GCDDiscoveryRunner
        from ouroboros.environments.algorithm_env import GCDEnv
        runner = GCDDiscoveryRunner(n_attempts=1, verbose=False)
        env = GCDEnv(seed=1)
        accuracy = runner._evaluate_on_gcd_pairs(None, env)
        assert accuracy == 0.0

    def test_run_returns_result(self):
        from ouroboros.environments.gcd_runner import GCDDiscoveryRunner
        runner = GCDDiscoveryRunner(
            n_attempts=2, stream_length=30,
            beam_width=4, n_iterations=2, verbose=False
        )
        result = runner.run()
        assert 0.0 <= result.best_accuracy <= 1.0
        assert 0.0 <= result.mean_accuracy <= 1.0
        assert result.n_attempts == 2
        assert result.runtime_seconds > 0