# tests/unit/test_emergence.py

"""Unit tests for proto-axiom emergence components."""

import pytest
from ouroboros.compression.program_synthesis import (
    build_linear_modular, C, T, MOD, ADD
)
from ouroboros.emergence.fingerprint import (
    behavioral_fingerprint, expressions_equivalent, compression_fingerprint
)
from ouroboros.emergence.proto_axiom_pool import ProtoAxiomPool, ProtoAxiom


class TestBehavioralFingerprint:

    def test_same_expression_same_fingerprint(self):
        expr = build_linear_modular(3, 1, 7)
        fp1 = behavioral_fingerprint(expr, 7, 50)
        fp2 = behavioral_fingerprint(expr, 7, 50)
        assert fp1 == fp2

    def test_different_expressions_different_fingerprints(self):
        e1 = build_linear_modular(3, 1, 7)
        e2 = build_linear_modular(2, 1, 7)  # Different slope
        fp1 = behavioral_fingerprint(e1, 7, 50)
        fp2 = behavioral_fingerprint(e2, 7, 50)
        assert fp1 != fp2

    def test_equivalent_expressions_same_fingerprint(self):
        """(t*3+1) mod 7 and (1+t*3) mod 7 should have same fingerprint."""
        from ouroboros.compression.program_synthesis import MUL
        # (t*3 + 1) mod 7
        e1 = build_linear_modular(3, 1, 7)
        # (1 + t*3) mod 7
        e2 = MOD(ADD(C(1), MUL(T(), C(3))), C(7))
        assert expressions_equivalent(e1, e2, 7, 100)

    def test_fingerprint_length(self):
        expr = build_linear_modular(3, 1, 7)
        fp = behavioral_fingerprint(expr, 7, 50)
        assert len(fp) == 50

    def test_fingerprint_values_in_alphabet(self):
        expr = build_linear_modular(3, 1, 7)
        fp = behavioral_fingerprint(expr, 7, 100)
        assert all(0 <= v < 7 for v in fp)

    def test_compression_fingerprint_perfect(self):
        expr = build_linear_modular(3, 1, 7)
        stream = [(3*t+1) % 7 for t in range(100)]
        err = compression_fingerprint(expr, stream, 7)
        assert err == 0.0

    def test_compression_fingerprint_bad(self):
        expr = C(0)  # Always predicts 0
        stream = [1] * 100  # Always 1
        err = compression_fingerprint(expr, stream, 7)
        assert err == 1.0

    def test_empty_sequence_returns_one(self):
        expr = C(0)
        assert compression_fingerprint(expr, [], 7) == 1.0


class TestProtoAxiomPool:

    def _make_pool(self, num_agents=8, threshold=0.5, alpha=7):
        return ProtoAxiomPool(
            num_agents=num_agents,
            consensus_threshold=threshold,
            alphabet_size=alpha,
            fingerprint_length=50,
        )

    def _make_good_expr(self):
        return build_linear_modular(3, 1, 7)

    def test_submit_adds_to_submissions(self):
        pool = self._make_pool()
        expr = self._make_good_expr()
        pool.submit(agent_id=0, expression=expr, mdl_cost=10.0, step=100)
        assert 0 in pool.submissions

    def test_submit_none_is_ignored(self):
        pool = self._make_pool()
        pool.submit(agent_id=0, expression=None, mdl_cost=0.0, step=100)
        assert 0 not in pool.submissions

    def test_consensus_not_reached_below_threshold(self):
        pool = self._make_pool(num_agents=8, threshold=0.5)
        expr = self._make_good_expr()
        # Only 3/8 agents — below 0.5 threshold
        for i in range(3):
            pool.submit(i, expr, 10.0, 100)
        axioms = pool.detect_consensus(100, 'test', naive_bits=1000.0)
        assert len(axioms) == 0

    def test_consensus_reached_above_threshold(self):
        pool = self._make_pool(num_agents=8, threshold=0.5)
        expr = self._make_good_expr()
        # 4/8 agents — meets 0.5 threshold
        for i in range(4):
            pool.submit(i, expr, 10.0, 100)
        axioms = pool.detect_consensus(100, 'test', naive_bits=1000.0)
        assert len(axioms) == 1

    def test_promoted_axiom_has_correct_id_format(self):
        pool = self._make_pool()
        expr = self._make_good_expr()
        for i in range(4):
            pool.submit(i, expr, 10.0, 100)
        axioms = pool.detect_consensus(100, 'test', naive_bits=1000.0)
        assert axioms[0].axiom_id.startswith('AX_')
        assert len(axioms[0].axiom_id) == 8  # "AX_" + 5 digits

    def test_same_fingerprint_not_promoted_twice(self):
        pool = self._make_pool()
        expr = self._make_good_expr()
        for i in range(4):
            pool.submit(i, expr, 10.0, 100)
        axioms1 = pool.detect_consensus(100, 'test', naive_bits=1000.0)
        pool.clear_submissions()
        for i in range(4):
            pool.submit(i, expr, 10.0, 200)
        axioms2 = pool.detect_consensus(200, 'test', naive_bits=1000.0)
        # Should not be promoted again
        assert len(axioms2) == 0
        assert len(pool.axioms) == 1  # Still only 1 axiom total

    def test_confidence_positive_for_good_expression(self):
        pool = self._make_pool()
        expr = self._make_good_expr()
        for i in range(6):
            pool.submit(i, expr, 50.0, 100)
        axioms = pool.detect_consensus(100, 'test', naive_bits=5000.0)
        assert len(axioms) > 0
        assert axioms[0].confidence > 0

    def test_behavioral_equivalence_counts_as_consensus(self):
        """Two different expression strings with same behavior should count as one."""
        from ouroboros.compression.program_synthesis import MUL
        pool = self._make_pool(num_agents=8, threshold=0.5)
        nb = 5000.0

        e1 = build_linear_modular(3, 1, 7)        # (t*3+1) mod 7
        e2 = MOD(ADD(C(1), MUL(T(), C(3))), C(7)) # (1+t*3) mod 7 — same behavior

        # Agents 0-2 find e1, agents 3-5 find e2
        for i in range(3):
            pool.submit(i, e1, 50.0, 100)
        for i in range(3, 6):
            pool.submit(i, e2, 50.0, 100)

        axioms = pool.detect_consensus(100, 'test', naive_bits=nb)
        # 6/8 agents found equivalent expressions → should promote
        assert len(axioms) == 1

    def test_summary_string(self):
        pool = self._make_pool()
        expr = self._make_good_expr()
        for i in range(4):
            pool.submit(i, expr, 50.0, 100)
        pool.detect_consensus(100, 'test', naive_bits=5000.0)
        summary = pool.summary()
        assert 'AX_' in summary

    def test_len(self):
        pool = self._make_pool()
        assert len(pool) == 0
        expr = self._make_good_expr()
        for i in range(4):
            pool.submit(i, expr, 50.0, 100)
        pool.detect_consensus(100, 'test', naive_bits=5000.0)
        assert len(pool) == 1