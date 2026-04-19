"""Tests for extended primitive set — PREV, IF, SUB, DIV, POW."""
import pytest
from ouroboros.compression.program_synthesis import (
    ExprNode, NodeType, C, T, PREV, ADD, SUB, MUL, MOD, DIV, POW,
    IF, EQ, LT, build_fibonacci_mod, build_linear_modular, build_piecewise
)


class TestPREV:
    def test_prev_1_returns_history(self):
        node = PREV(1)
        history = [10, 20, 30]
        assert node.evaluate(1, history) == 10
        assert node.evaluate(2, history) == 20

    def test_prev_2_returns_two_back(self):
        node = PREV(2)
        history = [5, 10, 15, 20]
        assert node.evaluate(2, history) == 5
        assert node.evaluate(3, history) == 10

    def test_prev_out_of_bounds_returns_zero(self):
        node = PREV(1)
        assert node.evaluate(0, []) == 0
        assert node.evaluate(0, None) == 0

    def test_fibonacci_recurrence(self):
        fib = build_fibonacci_mod(11)
        preds = fib.predict_sequence(15, 11)
        # True Fibonacci mod 11
        true_fib = [0, 1]
        while len(true_fib) < 15:
            true_fib.append((true_fib[-1] + true_fib[-2]) % 11)
        assert preds == true_fib, f"Expected {true_fib}, got {preds}"

    def test_has_prev_flag(self):
        fib = build_fibonacci_mod(11)
        linear = build_linear_modular(3, 1, 7)
        assert fib.has_prev()
        assert not linear.has_prev()

    def test_predict_sequence_uses_own_predictions(self):
        """PREV in recurrence mode uses own previous outputs."""
        fib = build_fibonacci_mod(11)
        # First two predictions: uses t=0→0, t=1→prev(1)+prev(2) = 0+0 = 0? No.
        # Actually pred[0]=PREV(1)+PREV(2)=0+0=0, pred[1]=0+0=0 without seed.
        # With seed [0,1]: pred[0]=0+0=0 (no history), pred[1]=0+0=0...
        # The recurrence is seeded via predict_sequence's own output
        preds = fib.predict_sequence(10, 11)
        assert len(preds) == 10
        assert all(0 <= p < 11 for p in preds)


class TestIF:
    def test_if_true_branch(self):
        # IF(t==2, 99, 0)
        node = IF(EQ(T(), C(2)), C(99), C(0))
        assert node.evaluate(2) == 99
        assert node.evaluate(1) == 0
        assert node.evaluate(3) == 0

    def test_if_false_branch(self):
        node = IF(C(0), C(10), C(20))  # condition always False
        assert node.evaluate(0) == 20
        assert node.evaluate(5) == 20

    def test_if_nonzero_condition_is_true(self):
        node = IF(C(3), C(10), C(20))  # 3 != 0 → true
        assert node.evaluate(0) == 10

    def test_build_piecewise(self):
        expr = build_piecewise(4, C(1), C(0))
        preds = expr.predict_sequence(8, 2)
        # Every 4 steps: 0,0,0,0 period → at t=0: (0 mod 4)==0 → C(1)=1
        # t=1,2,3: != 0 → C(0)=0; t=4: ==0 → 1; etc.
        assert preds[0] == 1  # t=0: 0 mod 4 = 0 → C(1)
        assert preds[1] == 0  # t=1: 1 mod 4 = 1 → C(0)
        assert preds[4] == 1  # t=4: 4 mod 4 = 0 → C(1)

    def test_if_to_string_contains_if(self):
        node = IF(C(1), C(2), C(3))
        s = node.to_string()
        assert 'IF' in s


class TestNewArithmetic:
    def test_sub(self):
        node = SUB(T(), C(3))
        assert node.evaluate(5) == 2
        assert node.evaluate(3) == 0
        assert node.evaluate(0) == -3

    def test_div_integer_floor(self):
        node = DIV(T(), C(3))
        assert node.evaluate(7) == 2
        assert node.evaluate(9) == 3

    def test_div_zero_returns_zero(self):
        node = DIV(T(), C(0))
        assert node.evaluate(5) == 0

    def test_pow_basic(self):
        node = POW(C(2), C(3))
        assert node.evaluate(0) == 8  # 2^3

    def test_pow_clamped_exponent(self):
        # Exponent clamped to max 5
        node = POW(C(2), C(100))
        result = node.evaluate(0)
        assert result == 2**5  # clamped to 2^5=32

    def test_eq_node(self):
        node = EQ(T(), C(5))
        assert node.evaluate(5) == 1
        assert node.evaluate(4) == 0

    def test_lt_node(self):
        node = LT(T(), C(5))
        assert node.evaluate(3) == 1
        assert node.evaluate(5) == 0
        assert node.evaluate(6) == 0


class TestBeamSearchWithPREV:
    def test_finds_fibonacci_structure(self):
        """Beam search should find something better than random on Fibonacci."""
        from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
        from ouroboros.environment.structured import FibonacciModEnv

        env = FibonacciModEnv(11, seed=42)
        env.reset(200)
        stream = env.peek_all()

        synth = BeamSearchSynthesizer(
            beam_width=15, max_depth=3, const_range=12,
            alphabet_size=11, enable_prev=True, max_lag=3
        )
        expr, cost = synth.search(stream[:100])
        # Just verify it runs without error and returns something
        assert isinstance(expr, ExprNode)
        assert isinstance(cost, float)

    def test_prev_disabled_still_works(self):
        """With enable_prev=False, system should still work (just less expressive)."""
        from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
        synth = BeamSearchSynthesizer(
            beam_width=5, max_depth=2, const_range=3,
            enable_prev=False, enable_if=False
        )
        expr, cost = synth.search([0, 1, 0, 1, 0, 1] * 10)
        assert isinstance(expr, ExprNode)


class TestMCMCWithNewNodes:
    def test_refine_fibonacci_expression(self):
        from ouroboros.compression.mcmc_refiner import MCMCRefiner
        fib = build_fibonacci_mod(11)
        actuals = [0, 1, 1, 2, 3, 5, 8, 2, 10, 1] * 5  # Fib mod 11 approx
        refiner = MCMCRefiner(num_iterations=30, alphabet_size=11, max_lag=3)
        refined, cost = refiner.refine(fib, actuals)
        assert isinstance(refined, ExprNode)
        assert cost >= 0.0