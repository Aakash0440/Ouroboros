# tests/unit/test_synthesis.py

"""Unit tests for symbolic program synthesis components."""

import pytest
from ouroboros.compression.program_synthesis import (
    ExprNode, NodeType, C, T, ADD, MUL, MOD, build_linear_modular
)
from ouroboros.compression.beam_search import BeamSearchSynthesizer
from ouroboros.compression.mcmc_refiner import MCMCRefiner


class TestExprNode:

    def test_const_always_returns_value(self):
        node = C(5)
        for t in range(10):
            assert node.evaluate(t) == 5

    def test_time_returns_t(self):
        node = T()
        for t in range(10):
            assert node.evaluate(t) == t

    def test_add_node(self):
        node = ADD(T(), C(3))
        for t in range(10):
            assert node.evaluate(t) == t + 3

    def test_mul_node(self):
        node = MUL(C(3), T())
        for t in range(10):
            assert node.evaluate(t) == 3 * t

    def test_mod_node(self):
        node = MOD(T(), C(7))
        for t in range(20):
            assert node.evaluate(t) == t % 7

    def test_mod_by_zero_returns_zero(self):
        node = MOD(T(), C(0))
        assert node.evaluate(5) == 0

    def test_build_linear_modular_correct(self):
        expr = build_linear_modular(slope=3, intercept=1, modulus=7)
        for t in range(50):
            expected = (3 * t + 1) % 7
            assert expr.evaluate(t) == expected, f"t={t}"

    def test_to_string_readable(self):
        expr = build_linear_modular(3, 1, 7)
        s = expr.to_string()
        assert 't' in s
        assert '3' in s
        assert '7' in s

    def test_to_bytes_is_bytes(self):
        expr = build_linear_modular(3, 1, 7)
        b = expr.to_bytes()
        assert isinstance(b, bytes)
        assert len(b) > 0

    def test_depth_correct(self):
        leaf = C(5)
        assert leaf.depth() == 0
        d1 = ADD(C(1), T())
        assert d1.depth() == 1
        d2 = MOD(ADD(C(1), T()), C(7))
        assert d2.depth() == 2

    def test_contains_time(self):
        assert T().contains_time() is True
        assert C(5).contains_time() is False
        assert ADD(C(1), T()).contains_time() is True
        assert ADD(C(1), C(2)).contains_time() is False

    def test_predict_sequence_clamped(self):
        expr = T()  # evaluates to t
        preds = expr.predict_sequence(10, alphabet_size=4)
        assert all(0 <= p < 4 for p in preds)

    def test_equality(self):
        e1 = build_linear_modular(3, 1, 7)
        e2 = build_linear_modular(3, 1, 7)
        assert e1 == e2

    def test_inequality_different_params(self):
        e1 = build_linear_modular(3, 1, 7)
        e2 = build_linear_modular(3, 2, 7)
        assert e1 != e2


class TestBeamSearchSynthesizer:

    def setup_method(self):
        self.synth = BeamSearchSynthesizer(
            beam_width=15, max_depth=2, const_range=8, alphabet_size=4
        )

    def test_returns_expression_and_cost(self):
        actuals = [0] * 50
        expr, cost = self.synth.search(actuals)
        assert isinstance(expr, ExprNode)
        assert isinstance(cost, float)
        assert cost >= 0

    def test_finds_constant_for_constant_stream(self):
        actuals = [2] * 100  # All 2s
        synth = BeamSearchSynthesizer(
            beam_width=10, max_depth=1, const_range=5, alphabet_size=4
        )
        expr, cost = synth.search(actuals)
        # Best expression should predict 2 for all positions
        preds = expr.predict_sequence(20, 4)
        correct = sum(p == 2 for p in preds)
        assert correct >= 15  # At least 75% correct

    def test_finds_modular_structure(self):
        """Key test: beam search should find good compression on modular stream."""
        synth = BeamSearchSynthesizer(
            beam_width=25, max_depth=3, const_range=10, alphabet_size=7
        )
        actuals = [(3*t+1) % 7 for t in range(200)]
        expr, cost = synth.search(actuals)

        from ouroboros.compression.mdl import naive_bits
        nb = naive_bits(actuals, 7)
        ratio = cost / nb

        # Should find much better than random (even if not exact rule)
        assert ratio < 0.60, f"Expected ratio < 0.60, got {ratio:.3f}"

    def test_empty_input(self):
        expr, cost = self.synth.search([])
        assert cost == float('inf')

    def test_beats_ngram_on_structured(self):
        """Symbolic should beat n-gram on modular stream."""
        synth = BeamSearchSynthesizer(
            beam_width=25, max_depth=3, const_range=14, alphabet_size=7
        )
        actuals = [(3*t+1) % 7 for t in range(300)]
        expr, cost = synth.search(actuals)
        assert synth.beats_ngram(expr, cost, actuals)


class TestMCMCRefiner:

    def test_returns_expression_and_cost(self):
        expr = C(0)
        actuals = [0] * 50
        refiner = MCMCRefiner(num_iterations=10, alphabet_size=4, seed=0)
        refined, cost = refiner.refine(expr, actuals)
        assert isinstance(refined, ExprNode)
        assert cost >= 0

    def test_does_not_make_things_much_worse(self):
        """MCMC should not significantly degrade a good expression."""
        from ouroboros.compression.mdl import MDLCost
        expr = build_linear_modular(3, 1, 7)
        actuals = [(3*t+1) % 7 for t in range(100)]

        mdl = MDLCost()
        preds = expr.predict_sequence(100, 7)
        initial_cost = mdl.total_cost(expr.to_bytes(), preds, actuals, 7)

        refiner = MCMCRefiner(num_iterations=50, alphabet_size=7, seed=42)
        _, refined_cost = refiner.refine(expr, actuals)

        # Refined cost should not be more than 50% worse
        assert refined_cost <= initial_cost * 1.5

    def test_mcmc_accepts_improvement(self):
        """If we start from a bad expression, MCMC should improve."""
        # Start from CONST(0) on an alternating stream
        bad_expr = C(0)
        actuals = [t % 2 for t in range(200)]

        refiner = MCMCRefiner(num_iterations=200, alphabet_size=2, seed=42)

        from ouroboros.compression.mdl import MDLCost
        mdl = MDLCost()
        preds = bad_expr.predict_sequence(200, 2)
        initial_cost = mdl.total_cost(bad_expr.to_bytes(), preds, actuals, 2)

        _, final_cost = refiner.refine(bad_expr, actuals)

        # Should improve at least a little
        assert final_cost <= initial_cost