"""
Tests for Lean4 bridge — translator and runner.
These tests run WITHOUT Lean4 installed (they test the translation only).
The runner tests use the fallback path.
"""
import pytest
from ouroboros.proof_market.lean4_bridge import (
    Lean4Translator, Lean4Runner, VerificationResult, VerificationReport,
    FormalProofMarket
)
from ouroboros.compression.program_synthesis import (
    build_linear_modular, C, T, ADD, MUL, MOD, IF, EQ, PREV
)


class TestLean4Translator:
    def setup_method(self):
        self.tr = Lean4Translator()

    def test_const_translates(self):
        node = C(5)
        result = self.tr.expr_to_lean4(node)
        assert result == '5'

    def test_time_translates(self):
        node = T()
        result = self.tr.expr_to_lean4(node)
        assert result == 't'

    def test_add_translates(self):
        node = ADD(T(), C(1))
        result = self.tr.expr_to_lean4(node)
        assert '+' in result
        assert 't' in result
        assert '1' in result

    def test_mul_translates(self):
        node = MUL(C(3), T())
        result = self.tr.expr_to_lean4(node)
        assert '*' in result
        assert '3' in result

    def test_mod_translates(self):
        node = MOD(T(), C(7))
        result = self.tr.expr_to_lean4(node)
        assert '%' in result
        assert '7' in result

    def test_linear_modular_translates(self):
        expr = build_linear_modular(3, 1, 7)
        result = self.tr.expr_to_lean4(expr)
        # Should contain 3, 1, 7, t, and arithmetic ops
        assert '3' in result
        assert '1' in result
        assert '7' in result
        assert 't' in result

    def test_if_node_translates(self):
        node = IF(EQ(T(), C(5)), C(1), C(0))
        result = self.tr.expr_to_lean4(node)
        assert 'if' in result.lower()

    def test_prev_translates(self):
        node = PREV(1)
        result = self.tr.expr_to_lean4(node)
        assert 'if' in result.lower()  # PREV uses conditional

    def test_build_verification_script_has_theorem(self):
        expr = build_linear_modular(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(10)]
        script = self.tr.build_verification_script(expr, stream, 7)
        assert 'theorem' in script
        assert 'decide' in script or 'native_decide' in script
        assert 'exprMatchesStream' in script

    def test_script_contains_stream(self):
        expr = C(0)
        stream = [0, 1, 2, 3]
        script = self.tr.build_verification_script(expr, stream, 4)
        for v in stream:
            assert str(v) in script

    def test_long_stream_uses_native_decide(self):
        expr = C(0)
        stream = [0] * 100  # > 50 items
        script = self.tr.build_verification_script(expr, stream, 2)
        assert 'native_decide' in script

    def test_short_stream_uses_decide(self):
        expr = C(0)
        stream = [0] * 30  # <= 50 items
        script = self.tr.build_verification_script(expr, stream, 2)
        assert 'decide' in script

    def test_counterexample_script_has_eval(self):
        prop = C(3)
        curr = build_linear_modular(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(20)]
        script = self.tr.build_counterexample_search_script(prop, curr, stream, 7)
        assert '#eval' in script


class TestLean4Runner:
    def setup_method(self):
        self.runner = Lean4Runner(lean_executable='lean_NONEXISTENT_FOR_TEST')
        self.runner._lean4_available = False  # Force unavailable

    def test_unavailable_returns_unavailable(self):
        report = self.runner.run_script("theorem foo : 1 = 1 := by rfl")
        assert report.result == VerificationResult.UNAVAILABLE

    def test_is_available_cached(self):
        """Once checked, result is cached."""
        self.runner._lean4_available = True
        assert self.runner.is_available()
        self.runner._lean4_available = False
        assert not self.runner.is_available()

    def test_verification_report_approved_property(self):
        proved = VerificationReport(VerificationResult.PROVED)
        timeout = VerificationReport(VerificationResult.TIMEOUT)
        refuted = VerificationReport(VerificationResult.REFUTED)
        empirical_none = VerificationReport(VerificationResult.EMPIRICAL_NONE)
        empirical_ce = VerificationReport(VerificationResult.EMPIRICAL_CE)

        assert proved.approved
        assert timeout.approved    # Conservative: approve on timeout
        assert not refuted.approved
        assert empirical_none.approved
        assert not empirical_ce.approved

    def test_is_conclusive_property(self):
        assert VerificationReport(VerificationResult.PROVED).is_conclusive
        assert VerificationReport(VerificationResult.REFUTED).is_conclusive
        assert not VerificationReport(VerificationResult.TIMEOUT).is_conclusive
        assert not VerificationReport(VerificationResult.UNAVAILABLE).is_conclusive


class TestFormalProofMarket:
    def test_init(self):
        fpm = FormalProofMarket(num_agents=4)
        assert fpm.lean4_verifications == 0

    def test_verify_formally_fallback_when_unavailable(self):
        fpm = FormalProofMarket(num_agents=4)
        fpm.runner._lean4_available = False

        expr = build_linear_modular(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(50)]
        report = fpm.verify_formally(expr, stream, 7)

        # Should use empirical fallback
        assert report.method == 'empirical'
        assert report.result in (
            VerificationResult.EMPIRICAL_NONE,
            VerificationResult.EMPIRICAL_CE
        )

    def test_perfect_expression_returns_empirical_none(self):
        fpm = FormalProofMarket(num_agents=4)
        fpm.runner._lean4_available = False

        expr = build_linear_modular(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(50)]
        report = fpm.verify_formally(expr, stream, 7)
        assert report.result == VerificationResult.EMPIRICAL_NONE

    def test_wrong_expression_returns_empirical_ce(self):
        fpm = FormalProofMarket(num_agents=4)
        fpm.runner._lean4_available = False

        expr = C(99)  # Wrong expression
        stream = [(3*t+1)%7 for t in range(50)]
        report = fpm.verify_formally(expr, stream, 7)
        assert report.result == VerificationResult.EMPIRICAL_CE

    def test_stats_summary_string(self):
        fpm = FormalProofMarket(num_agents=4)
        s = fpm.stats_summary()
        assert 'FormalProofMarket' in s

    def test_formal_stats_structure(self):
        fpm = FormalProofMarket(num_agents=4)
        stats = fpm.formal_stats()
        assert 'total_verifications' in stats
        assert 'lean4_available' in stats