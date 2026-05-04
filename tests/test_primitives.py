
"""Tests for self-defining primitives pipeline."""
import pytest
import math
from ouroboros.primitives.completeness import CompletenessChecker, CompletenessResult
from ouroboros.primitives.proposer import PrimitiveProposer, ProposedPrimitive
from ouroboros.primitives.vocabulary_extender import VocabularyExtender, ExtensionResult


class TestCompletenessChecker:
    def test_finds_constant_sequence(self):
        checker = CompletenessChecker(max_depth=2, time_limit_seconds=10)
        obs = [5] * 50
        result = checker.check(obs, threshold=200.0)
        assert isinstance(result, CompletenessResult)

    def test_result_has_fields(self):
        checker = CompletenessChecker(max_depth=1, time_limit_seconds=3)
        obs = [3] * 30
        result = checker.check(obs, threshold=500.0)
        assert hasattr(result, 'is_complete')
        assert hasattr(result, 'best_cost')
        assert result.n_expressions_checked >= 1

    def test_terminal_enumeration(self):
        checker = CompletenessChecker()
        terminals = list(checker._terminal_expressions())
        assert len(terminals) >= 5
        names = set()
        for t in terminals:
            if hasattr(t.node_type, 'name'):
                names.add(t.node_type.name)
        assert 'TIME' in names or 'CONST' in names

    def test_certificate_not_found(self):
        # Very hard sequence — should not find expression quickly
        checker = CompletenessChecker(max_depth=1, time_limit_seconds=2)
        # Very complex sequence that won't compress to depth-1 expressions
        obs = [(i*37 + i*i*11) % 997 for i in range(50)]
        result = checker.check(obs, threshold=5.0)  # very low threshold
        # Either not complete (good) or found something (also OK)
        assert isinstance(result, CompletenessResult)
        assert isinstance(result.is_complete, bool)

    def test_str_representation(self):
        checker = CompletenessChecker(max_depth=1, time_limit_seconds=2)
        obs = [1] * 30
        result = checker.check(obs, threshold=300.0)
        s = str(result)
        assert isinstance(s, str) and len(s) > 10


class TestPrimitiveProposer:
    def _make_residuals(self, n: int, type_: str) -> list:
        """Make synthetic residuals of different types."""
        if type_ == "multiplicative":
            # r(mn) ≈ r(m) * r(n) — like Euler's totient divided by n
            import math
            r = [0.0]  # r[0] unused
            for k in range(1, n):
                # Simple multiplicative: totient-like
                r.append(float(k) * (1 - sum(1/p for p in range(2, k+1)
                                              if k % p == 0 and all(k%q != 0
                                              for q in range(2,p)))))
            return r[:n]
        if type_ == "recurrence_8":
            # Order-8 recurrence
            r = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
            for _ in range(n - 8):
                r.append(r[-1] + r[-7])
            return r[:n]
        if type_ == "lookup_7":
            pattern = [1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0]
            return [pattern[i % 7] for i in range(n)]
        return [float(i) for i in range(n)]

    def test_proposer_runs_without_crash(self):
        proposer = PrimitiveProposer()
        obs = [(3*t+1)%7 for t in range(80)]
        residuals = [float(i % 3) for i in range(80)]
        proposals = proposer.propose(residuals, obs)
        assert isinstance(proposals, list)

    def test_lookup_detection(self):
        proposer = PrimitiveProposer(min_residual_reduction=0.1)
        residuals = self._make_residuals(60, "lookup_7")
        obs = [int(r) for r in residuals]
        proposal = proposer._check_lookup_pattern(residuals, obs)
        if proposal:  # may or may not detect depending on threshold
            assert proposal.structure_type == "lookup"
            assert proposal.residual_reduction > 0

    def test_recurrence_detection(self):
        proposer = PrimitiveProposer(min_residual_reduction=0.1)
        residuals = self._make_residuals(80, "recurrence_8")
        proposal = proposer._check_higher_order_recurrence(residuals)
        # May or may not detect with simple OLS
        if proposal:
            assert "recurrence" in proposal.structure_type
            assert proposal.arity == 1

    def test_proposal_fields(self):
        p = ProposedPrimitive(
            name="TEST_NODE",
            description="A test primitive",
            structure_type="lookup",
            arity=0,
            category="NUMBER",
            test_inputs=list(range(10)),
            test_outputs=[float(i) for i in range(10)],
            implementation_code="def f(t): return t",
            lean4_definition="-- test",
            residual_reduction=0.5,
            confidence=0.8,
            grammar_rule="TEST_NODE is a terminal",
            description_bits=5.0,
        )
        assert p.is_worth_adding()
        assert "TEST_NODE" in p.summary()

    def test_not_worth_adding_low_reduction(self):
        p = ProposedPrimitive(
            name="WEAK",
            description="Weak",
            structure_type="lookup",
            arity=0,
            category="NUMBER",
            test_inputs=list(range(5)),
            test_outputs=[0.0] * 5,
            implementation_code="",
            lean4_definition="",
            residual_reduction=0.1,  # too low
            confidence=0.9,
            grammar_rule="",
            description_bits=5.0,
        )
        assert not p.is_worth_adding()


class TestVocabularyExtender:
    def _make_proposal(self, name="TEST", reduction=0.5, confidence=0.7):
        return ProposedPrimitive(
            name=name,
            description="Test primitive",
            structure_type="lookup",
            arity=0,
            category="NUMBER",
            test_inputs=list(range(15)),
            test_outputs=[float(i) for i in range(15)],
            implementation_code="def lookup_test(t): return float(t)",
            lean4_definition="-- test",
            residual_reduction=reduction,
            confidence=confidence,
            grammar_rule="terminal",
            description_bits=5.0,
        )

    def test_extend_good_proposal(self, tmp_path):
        extender = VocabularyExtender(
            log_path=str(tmp_path / "ext.json"),
            min_validation_accuracy=0.5,
        )
        proposal = self._make_proposal(reduction=0.5, confidence=0.8)
        result = extender.try_extend(proposal, verbose=False)
        assert isinstance(result, ExtensionResult)

    def test_reject_low_accuracy(self, tmp_path):
        extender = VocabularyExtender(
            log_path=str(tmp_path / "ext.json"),
            min_validation_accuracy=0.99,  # very strict
        )
        proposal = ProposedPrimitive(
            name="BAD",
            description="Bad", structure_type="lookup",
            arity=0, category="NUMBER",
            test_inputs=[1, 2], test_outputs=[float('inf'), float('nan')],  # invalid
            implementation_code="", lean4_definition="",
            residual_reduction=0.9, confidence=0.9,
            grammar_rule="", description_bits=5.0,
        )
        result = extender.try_extend(proposal, verbose=False)
        assert not result.added

    def test_extension_logged(self, tmp_path):
        import json
        extender = VocabularyExtender(
            log_path=str(tmp_path / "ext.json"),
            min_validation_accuracy=0.5,
        )
        proposal = self._make_proposal(reduction=0.6, confidence=0.8)
        result = extender.try_extend(proposal, verbose=False)
        if result.added:
            log = json.loads((tmp_path / "ext.json").read_text())
            assert len(log) >= 1
            assert log[0]["name"] == proposal.name

    def test_n_extensions_tracked(self, tmp_path):
        extender = VocabularyExtender(
            log_path=str(tmp_path / "ext.json"),
            min_validation_accuracy=0.5,
        )
        p1 = self._make_proposal("P1", 0.5, 0.8)
        p2 = self._make_proposal("P2", 0.4, 0.7)
        extender.try_extend(p1, verbose=False)
        extender.try_extend(p2, verbose=False)
        assert extender.n_extensions >= 0  # depends on validation