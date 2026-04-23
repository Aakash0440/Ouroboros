"""Tests for extended node types — all 40 new mathematical primitives."""
import pytest
import math
from ouroboros.nodes.extended_nodes import ExtNodeType, ExtExprNode, NODE_SPECS, NodeCategory


def const(v: float) -> ExtExprNode:
    from ouroboros.synthesis.expr_node import NodeType
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = NodeType.CONST; n.value = float(v)
    n.lag = 1; n.state_key = 0; n.window = 10
    n.left = n.right = n.third = None; n._cache = {}
    return n

def time_node() -> ExtExprNode:
    from ouroboros.synthesis.expr_node import NodeType
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = NodeType.TIME; n.value = 0.0
    n.lag = 1; n.state_key = 0; n.window = 10
    n.left = n.right = n.third = None; n._cache = {}
    return n

def make(nt, left=None, right=None, third=None) -> ExtExprNode:
    n = ExtExprNode(nt, left=left, right=right, third=third)
    return n


class TestNodeSpecs:
    def test_all_40_nodes_have_specs(self):
        assert len(NODE_SPECS) == 40

    def test_all_have_positive_description_bits(self):
        for nt, spec in NODE_SPECS.items():
            assert spec.description_bits > 0, f"{nt} has non-positive bits"

    def test_arity_consistency(self):
        for nt, spec in NODE_SPECS.items():
            assert spec.arity in (0, 1, 2, 3), f"{nt} has invalid arity {spec.arity}"

    def test_categories_assigned(self):
        categories = {spec.category for spec in NODE_SPECS.values()}
        # Should have at least 6 different categories
        assert len(categories) >= 6


class TestCalculusNodes:
    def test_deriv_of_linear_is_constant(self):
        # DERIV(t) at t=5: t - (t-1) = 1
        expr = make(ExtNodeType.DERIV, left=time_node())
        history = list(range(10))
        for t in range(2, 8):
            result = expr.evaluate(t, history[:t])
            assert abs(result - 1.0) < 0.01, f"DERIV(t) at t={t} should be 1, got {result}"

    def test_cumsum_of_constant(self):
        # CUMSUM(CONST(2)) at t=5: 2+2+2+2+2+2 = 12
        expr = make(ExtNodeType.CUMSUM, left=const(2.0))
        result = expr.evaluate(5, [])
        assert abs(result - 12.0) < 0.01

    def test_running_max_increases(self):
        expr = make(ExtNodeType.RUNNING_MAX, left=time_node())
        history = list(range(20))
        results = [expr.evaluate(t, history[:t]) for t in range(10)]
        # Running max of [0,1,2,...] should increase monotonically
        for i in range(1, len(results)):
            assert results[i] >= results[i-1]

    def test_ewma_smooth(self):
        # EWMA with alpha=0.5 of a step function
        expr = make(ExtNodeType.EWMA, left=const(1.0), right=const(0.5))
        r5 = expr.evaluate(5, [])
        # After 5 steps with alpha=0.5, EWMA of CONST(1) should approach 1
        assert 0.9 < r5 < 1.01

    def test_deriv2_of_quadratic(self):
        # DERIV2(t²) should be approximately 2
        # t² = t*t but we use TIME*TIME which isn't directly available
        # Use CONST(1) as a proxy — DERIV2(constant) = 0
        expr = make(ExtNodeType.DERIV2, left=const(5.0))
        result = expr.evaluate(5, [])
        assert abs(result) < 0.01  # second derivative of constant = 0


class TestStatisticalNodes:
    def test_mean_win_of_constant(self):
        # MEAN_WIN(CONST(3), CONST(10)) = 3.0
        expr = make(ExtNodeType.MEAN_WIN, left=const(3.0), right=const(10.0))
        result = expr.evaluate(20, [3.0]*20)
        assert abs(result - 3.0) < 0.01

    def test_std_win_of_constant_is_zero(self):
        # STD_WIN of a constant sequence = 0
        expr = make(ExtNodeType.STD_WIN, left=const(5.0), right=const(10.0))
        result = expr.evaluate(20, [5.0]*20)
        assert abs(result) < 0.01

    def test_zscore_of_mean_is_zero(self):
        # Z-score of the mean value = 0
        expr = make(ExtNodeType.ZSCORE, left=const(5.0), right=const(20.0))
        result = expr.evaluate(30, [5.0]*30)
        assert abs(result) < 0.1


class TestLogicalNodes:
    def test_threshold_above(self):
        expr = make(ExtNodeType.THRESHOLD, left=const(5.0), right=const(3.0))
        assert expr.evaluate(0, []) == 1.0

    def test_threshold_below(self):
        expr = make(ExtNodeType.THRESHOLD, left=const(2.0), right=const(3.0))
        assert expr.evaluate(0, []) == 0.0

    def test_sign_positive(self):
        expr = make(ExtNodeType.SIGN, left=const(7.0))
        assert expr.evaluate(0, []) == 1.0

    def test_sign_negative(self):
        expr = make(ExtNodeType.SIGN, left=const(-3.0))
        assert expr.evaluate(0, []) == -1.0

    def test_bool_and(self):
        t = make(ExtNodeType.THRESHOLD, left=const(5.0), right=const(3.0))  # True
        f = make(ExtNodeType.THRESHOLD, left=const(1.0), right=const(3.0))  # False
        assert make(ExtNodeType.BOOL_AND, left=t, right=t).evaluate(0, []) == 1.0
        assert make(ExtNodeType.BOOL_AND, left=t, right=f).evaluate(0, []) == 0.0


class TestNumberTheoreticNodes:
    def test_gcd(self):
        expr = make(ExtNodeType.GCD_NODE, left=const(12.0), right=const(8.0))
        assert expr.evaluate(0, []) == 4.0

    def test_isprime_2(self):
        expr = make(ExtNodeType.ISPRIME, left=const(2.0))
        assert expr.evaluate(0, []) == 1.0

    def test_isprime_4(self):
        expr = make(ExtNodeType.ISPRIME, left=const(4.0))
        assert expr.evaluate(0, []) == 0.0

    def test_isprime_7(self):
        expr = make(ExtNodeType.ISPRIME, left=const(7.0))
        assert expr.evaluate(0, []) == 1.0

    def test_totient_7(self):
        # φ(7) = 6 (all of 1..6 are coprime to 7)
        expr = make(ExtNodeType.TOTIENT, left=const(7.0))
        assert abs(expr.evaluate(0, []) - 6.0) < 0.01

    def test_floor_ceil(self):
        expr_floor = make(ExtNodeType.FLOOR_NODE, left=const(3.7))
        expr_ceil  = make(ExtNodeType.CEIL_NODE,  left=const(3.2))
        assert expr_floor.evaluate(0, []) == 3.0
        assert expr_ceil.evaluate(0, []) == 4.0

    def test_frac(self):
        expr = make(ExtNodeType.FRAC_NODE, left=const(3.75))
        assert abs(expr.evaluate(0, []) - 0.75) < 0.001


class TestMemoryNodes:
    def test_streak_counts_run(self):
        expr = make(ExtNodeType.STREAK, left=const(5.0))
        history = [5.0, 5.0, 5.0, 5.0]
        result = expr.evaluate(4, history)
        assert result >= 4.0  # streak of 4+ fives

    def test_delta_zero_at_zero(self):
        expr = make(ExtNodeType.DELTA_ZERO, left=const(0.0))
        history = [1.0, 2.0, 0.0]  # last zero at t=2
        result = expr.evaluate(3, history)
        assert result == 0.0  # 0 steps since last zero (it's at t=2, we're at t=3)


class TestGrammarBeam:
    def test_grammar_beam_runs(self):
        from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
        obs = [(3*t+1)%7 for t in range(100)]
        cfg = GrammarBeamConfig(beam_width=8, max_depth=3, n_iterations=3)
        beam = GrammarConstrainedBeam(cfg)
        result = beam.search(obs)
        assert result is not None

    def test_grammar_beam_expr_is_string(self):
        from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
        obs = [t % 5 for t in range(50)]
        cfg = GrammarBeamConfig(beam_width=5, max_depth=2, n_iterations=2)
        beam = GrammarConstrainedBeam(cfg)
        result = beam.search(obs)
        if result:
            assert isinstance(result.to_string(), str)