"""Tests for FormalAxiomRegistry."""
import pytest
import tempfile
import os
from ouroboros.emergence.formal_axiom_registry import FormalAxiomRegistry, FormalAxiom
from ouroboros.compression.program_synthesis import build_linear_modular
from ouroboros.emergence.proto_axiom_pool import ProtoAxiom


def make_proto_axiom(slope, intercept, modulus):
    from ouroboros.emergence.fingerprint import behavioral_fingerprint
    expr = build_linear_modular(slope, intercept, modulus)
    fp = behavioral_fingerprint(expr, modulus, 50)
    return ProtoAxiom(
        axiom_id=f"AX_test_{slope}_{intercept}_{modulus}",
        expression=expr,
        fingerprint=fp,
        supporting_agents=[0, 1, 2],
        confidence=0.75,
        environment_name="TestEnv",
        compression_ratio=0.4,
        discovery_step=0,
    )


@pytest.fixture
def registry(tmp_path):
    db_path = str(tmp_path / 'formal_test.db')
    r = FormalAxiomRegistry(db_path)
    r.runner._lean4_available = False  # Force empirical
    yield r


class TestFormalAxiomRegistry:
    def test_verify_correct_expr_empirical(self, registry):
        ax = make_proto_axiom(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(50)]
        aid, report = registry.verify_and_register(ax, stream, 7, "TestEnv")
        assert aid > 0
        assert report.method == 'empirical'

    def test_correct_expr_passes(self, registry):
        ax = make_proto_axiom(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(50)]
        _, report = registry.verify_and_register(ax, stream, 7, "TestEnv")
        from ouroboros.proof_market.lean4_bridge import VerificationResult
        assert report.result == VerificationResult.EMPIRICAL_NONE

    def test_wrong_expr_fails(self, registry):
        from ouroboros.compression.program_synthesis import C
        from ouroboros.emergence.proto_axiom_pool import ProtoAxiom
        bad_expr = C(99)
        fp = tuple(99 % 7 for _ in range(100))
        ax = ProtoAxiom(
            axiom_id="AX_test_bad",
            expression=bad_expr,
            fingerprint=fp,
            supporting_agents=[0],
            confidence=0.1,
            environment_name="TestEnv",
            compression_ratio=0.9,
            discovery_step=0,
        )
        stream = [(3*t+1)%7 for t in range(50)]
        _, report = registry.verify_and_register(ax, stream, 7, "TestEnv")
        from ouroboros.proof_market.lean4_bridge import VerificationResult
        assert report.result == VerificationResult.EMPIRICAL_CE

    def test_get_fully_verified(self, registry):
        ax = make_proto_axiom(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(50)]
        registry.verify_and_register(ax, stream, 7, "TestEnv")
        verified = registry.get_fully_verified_axioms()
        assert isinstance(verified, list)

    def test_registry_summary_string(self, registry):
        s = registry.registry_summary()
        assert 'FormalAxiomRegistry' in s
        assert 'Lean4 available' in s

    def test_export_lean4_library(self, registry, tmp_path):
        ax = make_proto_axiom(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(50)]
        registry.verify_and_register(ax, stream, 7, "TestEnv")
        out = str(tmp_path / 'discovered.lean')
        registry.export_lean4_library(out)
        assert os.path.exists(out)
        content = open(out).read()
        assert 'OUROBOROS' in content
        assert 'namespace' in content


class TestFormalAxiom:
    def test_combined_confidence_lean4_verified(self):
        from ouroboros.core.knowledge_base import StoredAxiom
        stored = StoredAxiom(1, "(t*3+1)mod7", "hash", 0.7, 0.004, "E", 7)
        fa = FormalAxiom(stored=stored, lean4_verified=True, empirical_verified=True)
        # 0.7 * 1.0 + 0.3 * 0.7 = 0.91
        assert fa.combined_confidence > 0.85

    def test_combined_confidence_unverified(self):
        from ouroboros.core.knowledge_base import StoredAxiom
        stored = StoredAxiom(1, "expr", "hash", 0.5, 0.1, "E", 7)
        fa = FormalAxiom(stored=stored)
        assert fa.combined_confidence == 0.0

    def test_is_fully_verified(self):
        from ouroboros.core.knowledge_base import StoredAxiom
        stored = StoredAxiom(1, "expr", "hash", 0.5, 0.1, "E", 7)
        fa = FormalAxiom(stored=stored, lean4_verified=True, empirical_verified=True)
        assert fa.is_fully_verified
        fa2 = FormalAxiom(stored=stored, lean4_verified=False, empirical_verified=True)
        assert not fa2.is_fully_verified
