"""Tests for ContinuousExprNode."""
import math
import pytest
from ouroboros.continuous.expr_nodes import (
    ContinuousExprNode as E, ContinuousNodeType,
    build_sine_expr, build_damped_sine_expr, build_polynomial_expr,
    PENALTY,
)


class TestTerminals:
    def test_const(self):
        node = E.const(3.14)
        assert node.evaluate(0, []) == pytest.approx(3.14)
        assert node.evaluate(99, []) == pytest.approx(3.14)

    def test_time(self):
        node = E.time()
        for t in range(20):
            assert node.evaluate(t, []) == pytest.approx(float(t))

    def test_prev_with_history(self):
        node = E.prev(lag=1)
        history = [10.0, 20.0, 30.0]
        assert node.evaluate(1, history) == pytest.approx(10.0)
        assert node.evaluate(2, history) == pytest.approx(20.0)

    def test_prev_before_start(self):
        node = E.prev(lag=1)
        assert node.evaluate(0, []) == pytest.approx(0.0)  # zero-padded


class TestUnaryOps:
    def test_sin(self):
        node = E.sin(E.const(math.pi / 2))
        result = node.evaluate(0, [])
        assert abs(result - 1.0) < 1e-10

    def test_cos(self):
        node = E.cos(E.const(0.0))
        result = node.evaluate(0, [])
        assert abs(result - 1.0) < 1e-10

    def test_exp(self):
        node = E.exp(E.const(0.0))
        assert node.evaluate(0, []) == pytest.approx(1.0)

    def test_exp_clips_overflow(self):
        node = E.exp(E.const(999.0))
        result = node.evaluate(0, [])
        # Clipped at exp(100)
        assert result == pytest.approx(math.exp(100.0))

    def test_log_protected(self):
        node = E(ContinuousNodeType.LOG, left=E.const(0.0))
        result = node.evaluate(0, [])
        assert math.isfinite(result)   # log(EPS) not log(0)

    def test_neg(self):
        node = E(ContinuousNodeType.NEG, left=E.const(5.0))
        assert node.evaluate(0, []) == pytest.approx(-5.0)


class TestBinaryOps:
    def test_add(self):
        node = E.add(E.const(3.0), E.const(4.0))
        assert node.evaluate(0, []) == pytest.approx(7.0)

    def test_mul(self):
        node = E.mul(E.const(3.0), E.const(4.0))
        assert node.evaluate(0, []) == pytest.approx(12.0)

    def test_sub(self):
        node = E.sub(E.const(10.0), E.const(3.0))
        assert node.evaluate(0, []) == pytest.approx(7.0)

    def test_div_protected(self):
        node = E.div(E.const(10.0), E.const(0.0))
        result = node.evaluate(0, [])
        assert math.isfinite(result)   # protected against div/0

    def test_div_normal(self):
        node = E.div(E.const(10.0), E.const(2.0))
        assert node.evaluate(0, []) == pytest.approx(5.0)


class TestNodeMetrics:
    def test_node_count_terminal(self):
        assert E.const(1.0).node_count() == 1
        assert E.time().node_count() == 1

    def test_node_count_unary(self):
        node = E.sin(E.time())
        assert node.node_count() == 2

    def test_node_count_binary(self):
        node = E.add(E.const(1.0), E.time())
        assert node.node_count() == 3

    def test_constant_count(self):
        node = E.add(E.const(1.0), E.mul(E.const(0.5), E.time()))
        assert node.constant_count() == 2

    def test_depth(self):
        node = E.sin(E.mul(E.const(0.5), E.time()))
        assert node.depth() == 3


class TestToString:
    def test_const_str(self):
        assert E.const(3.14).to_string() == "3.1400"

    def test_time_str(self):
        assert E.time().to_string() == "t"

    def test_sin_str(self):
        node = E.sin(E.time())
        assert "sin" in node.to_string() and "t" in node.to_string()

    def test_complex_str_is_string(self):
        node = E.add(E.sin(E.mul(E.const(0.5), E.time())), E.const(1.0))
        s = node.to_string()
        assert isinstance(s, str) and len(s) > 0


class TestPrebuiltExpressions:
    def test_sine_expr_accuracy(self):
        expr = build_sine_expr(frequency=1/7)
        actuals = [math.sin(t / 7.0) for t in range(100)]
        preds = [expr.evaluate(t, []) for t in range(100)]
        errors = [abs(p - a) for p, a in zip(preds, actuals)]
        assert max(errors) < 1e-8

    def test_damped_sine_expr(self):
        expr = build_damped_sine_expr(amplitude=2.0, decay=0.1, omega=0.8)
        import math
        expected = 2.0 * math.exp(-0.1 * 5) * math.sin(0.8 * 5)
        result = expr.evaluate(5, [])
        assert abs(result - expected) < 1e-8

    def test_polynomial_expr_degree_2(self):
        # 1.0 + (-2.0)*t + 0.5*t² — we use simplified builder
        expr = build_polynomial_expr([1.0, -2.0, 0.5])
        # At t=0: 1.0
        assert abs(expr.evaluate(0, []) - 1.0) < 1e-3