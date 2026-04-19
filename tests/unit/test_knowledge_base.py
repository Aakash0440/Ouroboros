"""Unit tests for KnowledgeBase."""
import pytest
import os
import tempfile
from ouroboros.core.knowledge_base import KnowledgeBase, StoredAxiom
from ouroboros.compression.program_synthesis import build_linear_modular
from ouroboros.emergence.proto_axiom_pool import ProtoAxiom


def make_test_axiom(expr_str_suffix=""):
    """Create a test ProtoAxiom."""
    from ouroboros.compression.program_synthesis import build_linear_modular
    expr = build_linear_modular(3, 1, 7)
    fp = tuple((3*t+1) % 7 for t in range(100))
    return ProtoAxiom(
        axiom_id="AX_00001",
        expression=expr,
        fingerprint=fp,
        supporting_agents=[0, 1, 2],
        confidence=0.72,
        environment_name=f"TestEnv{expr_str_suffix}",
        compression_ratio=0.004,
        discovery_step=500,
    )


@pytest.fixture
def kb(tmp_path):
    db_path = str(tmp_path / 'test_kb.db')
    with KnowledgeBase(db_path) as k:
        yield k


class TestKBCreation:
    def test_creates_db_file(self, tmp_path):
        path = str(tmp_path / 'new.db')
        with KnowledgeBase(path) as kb:
            pass
        assert os.path.exists(path)

    def test_tables_created(self, kb):
        stats = kb.statistics()
        assert 'total_axioms' in stats
        assert stats['total_axioms'] == 0


class TestSaveAxiom:
    def test_save_returns_id(self, kb):
        axiom = make_test_axiom()
        aid = kb.save_axiom(axiom, "TestEnv", 7)
        assert isinstance(aid, int)
        assert aid > 0

    def test_save_increments_count(self, kb):
        axiom = make_test_axiom()
        kb.save_axiom(axiom, "TestEnv", 7)
        assert kb.statistics()['total_axioms'] == 1

    def test_same_fingerprint_increments_confirmed(self, kb):
        axiom = make_test_axiom()
        kb.save_axiom(axiom, "TestEnv", 7)
        kb.save_axiom(axiom, "TestEnv", 7)
        # Still 1 axiom, but times_confirmed=2
        assert kb.statistics()['total_axioms'] == 1
        loaded = kb.load_all_axioms(min_confidence=0.0)
        assert loaded[0].times_confirmed == 2

    def test_different_fingerprints_create_separate(self, kb):
        from ouroboros.compression.program_synthesis import build_linear_modular
        from ouroboros.emergence.proto_axiom_pool import ProtoAxiom

        for slope in [3, 5]:
            expr = build_linear_modular(slope, 1, 7)
            fp = tuple((slope*t+1) % 7 for t in range(100))
            ax = ProtoAxiom(
    axiom_id=f"AX_{slope:05d}",
    expression=expr,
    fingerprint=fp,
    supporting_agents=[0],
    confidence=0.5,
    environment_name="TestEnv",
    compression_ratio=0.05,
    discovery_step=100,
)
            kb.save_axiom(ax, "TestEnv", 7)

        assert kb.statistics()['total_axioms'] == 2

    def test_save_updates_confidence_if_higher(self, kb):
        axiom_low = make_test_axiom()
        axiom_low.confidence = 0.3
        kb.save_axiom(axiom_low, "TestEnv", 7)

        axiom_high = make_test_axiom()
        axiom_high.confidence = 0.9
        kb.save_axiom(axiom_high, "TestEnv", 7)

        loaded = kb.load_all_axioms(min_confidence=0.0)
        assert loaded[0].confidence == 0.9


class TestLoadAxioms:
    def test_load_empty_returns_empty(self, kb):
        result = kb.load_axioms_for_environment("NonExistentEnv")
        assert result == []

    def test_load_by_environment(self, kb):
        ax1 = make_test_axiom()
        kb.save_axiom(ax1, "ModArith(7)", 7)
        loaded = kb.load_axioms_for_environment("ModArith(7)")
        assert len(loaded) == 1
        assert loaded[0].environment_name == "ModArith(7)"

    def test_min_confidence_filter(self, kb):
        axiom = make_test_axiom()
        axiom.confidence = 0.2
        kb.save_axiom(axiom, "TestEnv", 7)
        high = kb.load_all_axioms(min_confidence=0.5)
        assert len(high) == 0
        low = kb.load_all_axioms(min_confidence=0.1)
        assert len(low) == 1

    def test_get_seed_expressions_by_alphabet(self, kb):
        ax = make_test_axiom()
        kb.save_axiom(ax, "TestEnv", 7)
        seeds = kb.get_seed_expressions_for_search(alphabet_size=7)
        assert len(seeds) >= 1
        assert isinstance(seeds[0], str)

    def test_seed_expressions_wrong_alpha_empty(self, kb):
        ax = make_test_axiom()
        kb.save_axiom(ax, "TestEnv", 7)
        seeds = kb.get_seed_expressions_for_search(alphabet_size=11)
        assert len(seeds) == 0


class TestRunHistory:
    def test_save_run(self, kb):
        kb.save_run("run_001", "TestEnv", {
            'best_ratio': 0.004,
            'elapsed_seconds': 30.0,
            'axioms_promoted': [{'axiom_id': 'AX_001'}]
        })
        assert kb.statistics()['total_runs'] == 1


class TestStatistics:
    def test_statistics_structure(self, kb):
        stats = kb.statistics()
        required_keys = ['total_axioms', 'high_confidence_axioms',
                         'multi_confirmed_axioms', 'total_runs',
                         'distinct_environments']
        for k in required_keys:
            assert k in stats

    def test_multi_confirmed_count(self, kb):
        ax = make_test_axiom()
        kb.save_axiom(ax, "TestEnv", 7)  # times_confirmed=1
        kb.save_axiom(ax, "TestEnv", 7)  # times_confirmed=2
        kb.save_axiom(ax, "TestEnv", 7)  # times_confirmed=3
        assert kb.statistics()['multi_confirmed_axioms'] == 1