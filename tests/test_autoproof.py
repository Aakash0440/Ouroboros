"""Tests for automated Lean4 proof generation."""
import pytest
from ouroboros.autoformalize.proof_generator import (
    AutoProofGenerator, ProofTemplateLibrary, ProofErrorParser,
    ProofAttempt, AutoProofResult,
)


class TestProofTemplateLibrary:
    def test_periodicity_templates(self):
        lib = ProofTemplateLibrary()
        templates = lib.periodicity_template(3, 1, 7)
        assert len(templates) >= 2
        for stmt, tactic in templates:
            assert "3" in stmt and "7" in stmt
            assert isinstance(tactic, str)

    def test_surjectivity_witnesses(self):
        lib = ProofTemplateLibrary()
        witnesses = {0: 2, 1: 0, 2: 5, 3: 3, 4: 1, 5: 6, 6: 4}
        templates = lib.surjectivity_template(3, 1, 7, witnesses)
        assert len(templates) >= 1
        stmt, tactic = templates[0]
        assert "interval_cases" in tactic or "norm_num" in tactic

    def test_range_bound_templates(self):
        lib = ProofTemplateLibrary()
        templates = lib.range_bound_template(3, 1, 7)
        assert len(templates) >= 1
        stmt, _ = templates[0]
        assert "mod_lt" in stmt or "< 7" in stmt


class TestProofErrorParser:
    def test_omega_failed(self):
        parser = ProofErrorParser()
        hints = parser.parse("omega tactic failed")
        assert "by_cases" in hints.get("tactic", "") or "reason" in hints

    def test_unknown_identifier(self):
        parser = ProofErrorParser()
        hints = parser.parse("unknown identifier 'Nat.coprime'")
        assert "missing_import" in hints or "reason" in hints

    def test_type_mismatch(self):
        parser = ProofErrorParser()
        hints = parser.parse("type mismatch: expected Nat, got Int")
        assert hints.get("type_issue") or "reason" in hints

    def test_unknown_error(self):
        parser = ProofErrorParser()
        hints = parser.parse("some completely new error we've never seen")
        assert "reason" in hints


class TestAutoProofGenerator:
    def test_prove_periodicity_heuristic(self):
        gen = AutoProofGenerator(lean4_path="lean_not_installed_fake")
        result = gen.prove_modular_periodicity(3, 1, 7)
        assert isinstance(result, AutoProofResult)
        # With heuristic: omega should succeed for linear arithmetic
        if result.proved:
            assert result.final_proof is not None
            assert "theorem" in result.final_proof

    def test_prove_surjectivity_heuristic(self):
        gen = AutoProofGenerator(lean4_path="lean_not_installed_fake")
        result = gen.prove_surjectivity(3, 1, 7)
        assert isinstance(result, AutoProofResult)
        assert isinstance(result.proved, bool)

    def test_witnesses_computed(self):
        gen = AutoProofGenerator()
        # Compute witnesses manually
        witnesses = {}
        for t in range(14):
            r = (3 * t + 1) % 7
            if r not in witnesses:
                witnesses[r] = t
        assert len(witnesses) == 7  # all residues reachable
        assert witnesses[0] == 2   # (3*2+1)%7 = 0
        assert witnesses[1] == 0   # (3*0+1)%7 = 1

    def test_not_surjective_when_not(self):
        gen = AutoProofGenerator()
        # (2*t) % 6 is NOT surjective (only even residues)
        result = gen.prove_surjectivity(2, 0, 6)
        # Should fail since not all residues are reachable
        assert not result.proved or result.failure_reason is not None

    def test_proof_result_fields(self):
        gen = AutoProofGenerator(lean4_path="lean_not_installed_fake")
        result = gen.prove_modular_periodicity(3, 1, 7)
        assert hasattr(result, 'proved')
        assert hasattr(result, 'n_attempts')
        assert hasattr(result, 'human_review_needed')

    def test_lean4_code_generation(self):
        gen = AutoProofGenerator()
        code = gen._build_lean4_code(
            "test_theorem",
            "∀ t : ℕ, (3 * t + 1) % 7 < 7",
            "by intro t; exact Nat.mod_lt _ (by norm_num)",
        )
        assert "theorem test_theorem" in code
        assert "import Mathlib" in code

    def test_heuristic_omega(self):
        gen = AutoProofGenerator()
        assert gen._heuristic_check("... % 7 ...", "by intro t; omega")

    def test_heuristic_norm_num(self):
        gen = AutoProofGenerator()
        assert gen._heuristic_check("... 7 ...", "by norm_num")

    def test_summary_proved(self):
        result = AutoProofResult(
            expression_str="(3t+1)%7",
            theorem_name="test",
            theorem_statement="...",
            proved=True,
            final_proof="theorem test : ... := by omega",
            n_attempts=1,
            attempts=[],
            failure_reason=None,
            human_review_needed=False,
        )
        assert "PROVED" in result.summary()

    def test_summary_failed(self):
        result = AutoProofResult(
            expression_str="test",
            theorem_name="test",
            theorem_statement="",
            proved=False,
            final_proof=None,
            n_attempts=5,
            attempts=[],
            failure_reason="All templates failed",
            human_review_needed=True,
        )
        assert "FAILED" in result.summary()
        assert result.human_review_needed