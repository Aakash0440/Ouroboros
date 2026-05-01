"""Tests for INTEGRAL_NODE and fundamental theorem of calculus."""
import pytest
import math
from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType, NODE_SPECS, NodeCategory
from ouroboros.environments.calculus_env import (
    FundamentalTheoremEnv, AntiderivativeEnv, CalculusDiscoveryRunner,
)
from ouroboros.synthesis.expr_node import NodeType


def make_const_ext(v: float) -> ExtExprNode:
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = NodeType.CONST; n.value = float(v)
    n.lag = 1; n.state_key = 0; n.window = 10
    n.left = n.right = n.third = None; n._cache = {}
    return n

def make_time_ext() -> ExtExprNode:
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = NodeType.TIME; n.value = 0.0
    n.lag = 1; n.state_key = 0; n.window = 10
    n.left = n.right = n.third = None; n._cache = {}
    return n


class TestINTEGRALNode:
    def test_integral_of_constant_1(self):
        """INTEGRAL(CONST(1))[t] = t+1 (sum of 1 from 0 to t)."""
        expr = ExtExprNode(ExtNodeType.INTEGRAL, left=make_const_ext(1.0))
        for t in range(10):
            result = expr.evaluate(t, [], {})
            assert abs(result - (t + 1)) < 0.01, f"INTEGRAL(1)[{t}]={result}, expected {t+1}"

    def test_integral_of_identity(self):
        """INTEGRAL(TIME)[t] = 0+1+2+...+t = t(t+1)/2."""
        expr = ExtExprNode(ExtNodeType.INTEGRAL, left=make_time_ext())
        for t in range(8):
            result = expr.evaluate(t, [], {})
            expected = t * (t + 1) / 2
            assert abs(result - expected) < 0.01, f"INTEGRAL(t)[{t}]={result}, expected {expected}"

    def test_integral_monotone(self):
        """INTEGRAL of positive f should be non-decreasing."""
        expr = ExtExprNode(ExtNodeType.INTEGRAL, left=make_const_ext(2.0))
        vals = [expr.evaluate(t, [], {}) for t in range(10)]
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i-1]

    def test_integral_node_in_specs(self):
        assert ExtNodeType.INTEGRAL in NODE_SPECS
        spec = NODE_SPECS[ExtNodeType.INTEGRAL]
        assert spec.category == NodeCategory.CALCULUS
        assert spec.arity == 1

    def test_deriv_of_integral_is_original(self):
        """FTC: DERIV(INTEGRAL(CONST(3)))[t] ≈ 3 for t>0."""
        inner = make_const_ext(3.0)
        integral_expr = ExtExprNode(ExtNodeType.INTEGRAL, left=inner)
        integral_at_deriv = ExtExprNode(ExtNodeType.DERIV, left=integral_expr)
        for t in range(1, 10):
            result = integral_at_deriv.evaluate(t, [], {})
            assert abs(result - 3.0) < 0.01, f"DERIV(INTEGRAL(3))[{t}]={result}"

    def test_integral_win_node_in_specs(self):
        assert ExtNodeType.INTEGRAL_WIN in NODE_SPECS

    def test_integral_win_sums_window(self):
        """INTEGRAL_WIN(CONST(2), CONST(3))[t] = 2+2+2 = 6 for t>=2."""
        inner = make_const_ext(2.0)
        window = make_const_ext(3.0)
        expr = ExtExprNode(ExtNodeType.INTEGRAL_WIN, left=inner, right=window)
        result = expr.evaluate(10, [], {})
        assert abs(result - 6.0) < 0.01, f"INTEGRAL_WIN(2,3)={result}"

    def test_integral_of_zero(self):
        """INTEGRAL(CONST(0))[t] = 0."""
        expr = ExtExprNode(ExtNodeType.INTEGRAL, left=make_const_ext(0.0))
        for t in range(5):
            assert abs(expr.evaluate(t, [], {})) < 0.01


class TestFundamentalTheoremEnv:
    def test_identity_values(self):
        env = FundamentalTheoremEnv(f_name="identity")
        vals = env.generate(6)
        # F(0)=0, F(1)=0+1=1, F(2)=0+1+2=3, F(3)=0+1+2+3=6
        assert abs(vals[0] - 0.0) < 0.01
        assert abs(vals[1] - 1.0) < 0.01
        assert abs(vals[2] - 3.0) < 0.01
        assert abs(vals[3] - 6.0) < 0.01

    def test_constant_values(self):
        env = FundamentalTheoremEnv(f_name="constant")
        vals = env.generate(5)
        # F(t) = t+1 (sum of 1s from 0 to t)
        for t in range(5):
            assert abs(vals[t] - (t + 1)) < 0.01

    def test_sine_generates_float(self):
        env = FundamentalTheoremEnv(f_name="sine")
        vals = env.generate(50)
        assert all(isinstance(v, float) for v in vals)

    def test_verify_ftc_holds_for_integral(self):
        """DERIV(INTEGRAL(f)) ≈ f — this should give ~100% accuracy."""
        env = FundamentalTheoremEnv(f_name="constant")
        vals = env.generate(50)
        ftc = env.verify_ftc_holds(vals, tolerance=0.5)
        assert ftc > 0.9, f"FTC accuracy {ftc} too low"

    def test_ground_truth_expression_str(self):
        env = FundamentalTheoremEnv(f_name="identity")
        s = env.ground_truth_expression()
        assert "INTEGRAL" in s


class TestCalculusDiscoveryRunner:
    def test_ftc_test_runs(self):
        runner = CalculusDiscoveryRunner()
        results = runner.run_ftc_test(verbose=False)
        assert "identity" in results
        assert "constant" in results

    def test_identity_ftc_high_accuracy(self):
        runner = CalculusDiscoveryRunner()
        results = runner.run_ftc_test(verbose=False)
        # INTEGRAL(t) should perfectly match F(t) = t(t+1)/2
        assert results["identity"]["match_accuracy"] > 0.9

    def test_constant_ftc_high_accuracy(self):
        runner = CalculusDiscoveryRunner()
        results = runner.run_ftc_test(verbose=False)
        assert results["constant"]["match_accuracy"] > 0.9

    def test_result_has_predictions(self):
        runner = CalculusDiscoveryRunner()
        results = runner.run_ftc_test(verbose=False)
        for key in results:
            assert "predictions_0_5" in results[key]
            assert len(results[key]["predictions_0_5"]) == 5


class TestAntiderivativeEnv:
    def test_identity_f_values(self):
        env = AntiderivativeEnv(f_name="identity")
        vals = env.generate(5)
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(1.0)
        assert vals[2] == pytest.approx(2.0)

    def test_constant_f_values(self):
        env = AntiderivativeEnv(f_name="constant")
        vals = env.generate(5)
        assert all(abs(v - 3.0) < 0.01 for v in vals)