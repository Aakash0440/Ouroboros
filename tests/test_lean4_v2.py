"""Tests for Lean4BridgeV2."""
import pytest
from pathlib import Path
from ouroboros.proof_market.lean4_bridge_v2 import (
    Lean4BridgeV2, ProofStatus, FormalProofResult
)


class TestProofStatus:
    def test_formally_verified_is_fully_verified(self):
        result = FormalProofResult(
            theorem_name="test",
            theorem_statement="test",
            lean4_code="",
            status=ProofStatus.FORMALLY_VERIFIED,
        )
        assert result.is_fully_verified

    def test_sorry_present_is_not_fully_verified(self):
        result = FormalProofResult(
            theorem_name="test",
            theorem_statement="test",
            lean4_code="",
            status=ProofStatus.SORRY_PRESENT,
        )
        assert not result.is_fully_verified

    def test_confidence_multiplier_formally_verified(self):
        result = FormalProofResult(
            theorem_name="test",
            theorem_statement="test",
            lean4_code="",
            status=ProofStatus.FORMALLY_VERIFIED,
        )
        assert result.confidence_multiplier == 2.0

    def test_confidence_multiplier_compile_error(self):
        result = FormalProofResult(
            theorem_name="test",
            theorem_statement="test",
            lean4_code="",
            status=ProofStatus.COMPILE_ERROR,
        )
        assert result.confidence_multiplier == 0.5

    def test_all_statuses_have_multiplier(self):
        for status in ProofStatus:
            result = FormalProofResult("t", "t", "", status)
            assert isinstance(result.confidence_multiplier, float)


class TestLean4BridgeV2Init:
    def test_default_path(self):
        bridge = Lean4BridgeV2()
        assert "OuroborosVerifier" in str(bridge.lean_project_path)

    def test_custom_path(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        assert bridge.lean_project_path == Path("lean4_verification/OuroborosVerifier")

    def test_lean4_available_returns_bool(self):
        bridge = Lean4BridgeV2()
        result = bridge.is_lean4_available()
        assert isinstance(result, bool)


class TestBuildProject:
    def test_build_returns_formal_proof_result(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        result = bridge.build_project()
        assert isinstance(result, FormalProofResult)
        assert isinstance(result.status, ProofStatus)

    def test_build_status_is_known_value(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        result = bridge.build_project()
        assert result.status in ProofStatus

    def test_sorry_count_is_non_negative(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        result = bridge.build_project()
        assert result.sorry_count >= 0


class TestVerifyModularAxiom:
    def test_returns_result(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        result = bridge.verify_modular_axiom(slope=3, intercept=1, modulus=7)
        assert isinstance(result, FormalProofResult)
        assert "3" in result.theorem_name
        assert "7" in result.theorem_name

    def test_lean4_available_gives_verified_or_sorry(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        if bridge.is_lean4_available():
            result = bridge.verify_modular_axiom(3, 1, 7)
            assert result.status in (
                ProofStatus.FORMALLY_VERIFIED,
                ProofStatus.SORRY_PRESENT,
                ProofStatus.COMPILE_ERROR,
            )


class TestGetSummary:
    def test_summary_has_expected_keys(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        summary = bridge.get_verified_theorem_summary()
        assert "project_path" in summary
        assert "build_status" in summary
        assert "theorems" in summary
        assert "total_theorems" in summary
        assert "sorry_count" in summary

    def test_theorems_is_list(self):
        bridge = Lean4BridgeV2(lean_project_path="lean4_verification/OuroborosVerifier")
        summary = bridge.get_verified_theorem_summary()
        assert isinstance(summary["theorems"], list)