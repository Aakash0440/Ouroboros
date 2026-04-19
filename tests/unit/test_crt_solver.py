"""Tests for CRTSolver and MultiStartSynthesizer."""
import pytest
from ouroboros.emergence.crt_solver import CRTSolver
from ouroboros.compression.multi_start_synthesis import MultiStartSynthesizer


class TestCRTSolver:
    def setup_method(self):
        self.solver = CRTSolver(7, 3, 1, 11, 5, 2)

    def test_exact_expression_correct(self):
        expr = self.solver.exact_expression()
        acc_all, acc_m1, acc_m2 = self.solver.verify_expression(expr, 200)
        assert acc_m1 > 0.99
        assert acc_m2 > 0.99

    def test_joint_value_consistency(self):
        for t in range(50):
            jv = self.solver.joint_value(t)
            assert jv % 7 == self.solver.obs1(t)
            assert jv % 11 == self.solver.obs2(t)
            assert 0 <= jv < 77

    def test_generate_joint_stream(self):
        stream = self.solver.generate_joint_stream(100)
        assert len(stream) == 100
        assert all(0 <= v < 77 for v in stream)

    def test_non_coprime_raises(self):
        with pytest.raises(ValueError):
            CRTSolver(6, 1, 0, 9, 1, 0)

    def test_find_exact_expression_returns_ints(self):
        slope, intercept = self.solver.find_exact_expression()
        assert isinstance(slope, int)
        assert isinstance(intercept, int)
        assert 0 <= slope < 77
        assert 0 <= intercept < 77

    def test_compression_ratio_below_threshold(self):
        cr = self.solver.compression_ratio_exact(200)
        # Exact expression should compress very well
        assert cr < 0.30

    def test_report_is_string(self):
        r = self.solver.report()
        assert isinstance(r, str)
        assert 'CRTSolver' in r
        assert 'mod1' in r


class TestMultiStartSynthesizer:
    def test_returns_expr_and_cost(self):
        ms = MultiStartSynthesizer(
            num_starts=2, beam_width=5, max_depth=2,
            const_range=10, mcmc_iterations=10, alphabet_size=7
        )
        seq = [(3*t+1) % 7 for t in range(50)]
        expr, cost = ms.search(seq)
        assert expr is not None
        assert cost < float('inf')

    def test_beats_single_start_on_structured(self):
        """Multi-start should generally do at least as well as single start."""
        from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
        seq = [(3*t+1) % 7 for t in range(100)]

        single = BeamSearchSynthesizer(
            beam_width=10, max_depth=2, const_range=14, alphabet_size=7
        )
        _, single_cost = single.search(seq)

        ms = MultiStartSynthesizer(
            num_starts=3, beam_width=10, max_depth=2,
            const_range=14, mcmc_iterations=50, alphabet_size=7
        )
        _, multi_cost = ms.search(seq)

        # Multi-start should be at least as good
        assert multi_cost <= single_cost * 1.1  # Allow 10% margin