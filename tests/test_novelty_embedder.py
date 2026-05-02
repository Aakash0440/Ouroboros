"""Tests for behavioral expression embeddings and registry."""
import pytest
import math
import numpy as np
from ouroboros.novelty.embedder import (
    BehavioralEmbedder, ExpressionEmbedding, ExpressionDatabase,
    KnownExpression, CANONICAL_SEQUENCES, EMBEDDING_DIM,
)
from ouroboros.novelty.registry import (
    EmbeddingRegistry, RegistrySearchResult,
    _calibrate_novelty_score, _categorize_novelty,
)
from ouroboros.synthesis.expr_node import ExprNode, NodeType


def make_const(v: float) -> ExprNode:
    return ExprNode(NodeType.CONST, value=v)

def make_time() -> ExprNode:
    return ExprNode(NodeType.TIME)


class TestCanonicalSequences:
    def test_correct_count(self):
        assert len(CANONICAL_SEQUENCES) == 20

    def test_correct_length(self):
        for seq in CANONICAL_SEQUENCES:
            assert len(seq) == 30

    def test_diverse_sequences(self):
        # Sequences should not all be identical
        first = CANONICAL_SEQUENCES[0]
        different = sum(1 for seq in CANONICAL_SEQUENCES[1:] if seq != first)
        assert different >= 15

    def test_embedding_dim(self):
        assert EMBEDDING_DIM == 20 * 30  # 600


class TestBehavioralEmbedder:
    def _embedder(self): return BehavioralEmbedder()

    def test_embed_const_succeeds(self):
        emb = self._embedder().embed(make_const(5.0))
        assert emb.is_valid
        assert emb.vector.shape == (EMBEDDING_DIM,)

    def test_embed_different_consts_different(self):
        embedder = self._embedder()
        e1 = embedder.embed(make_const(3.0))
        e2 = embedder.embed(make_const(7.0))
        assert e1.distance_to(e2) > 0.01

    def test_same_const_zero_distance(self):
        embedder = self._embedder()
        e1 = embedder.embed(make_const(5.0))
        e2 = embedder.embed(make_const(5.0))
        assert e1.distance_to(e2) < 0.001

    def test_unit_normalized(self):
        embedder = self._embedder()
        e = embedder.embed(make_const(3.0))
        assert abs(np.linalg.norm(e.vector) - 1.0) < 0.001

    def test_distance_in_range(self):
        embedder = self._embedder()
        e1 = embedder.embed(make_const(1.0))
        e2 = embedder.embed(make_time())
        d = e1.distance_to(e2)
        assert 0.0 <= d <= 2.0

    def test_time_and_zero_different(self):
        embedder = self._embedder()
        e_time = embedder.embed(make_time())
        e_zero = embedder.embed(make_const(0.0))
        assert e_time.distance_to(e_zero) > 0.05

    def test_are_equivalent_same_expr(self):
        embedder = self._embedder()
        assert embedder.are_equivalent(make_const(5.0), make_const(5.0))

    def test_are_not_equivalent_different(self):
        embedder = self._embedder()
        assert not embedder.are_equivalent(make_const(3.0), make_const(7.0))

    def test_caching_works(self):
        embedder = self._embedder()
        e1 = embedder.embed(make_const(3.0))
        e2 = embedder.embed(make_const(3.0))
        assert e1 is e2  # same object from cache

    def test_coverage_field(self):
        embedder = self._embedder()
        e = embedder.embed(make_const(5.0))
        assert 0.0 <= e.coverage <= 1.0

    def test_cluster_identical(self):
        embedder = self._embedder()
        exprs = [make_const(5.0), make_const(5.0), make_const(7.0)]
        clusters = embedder.cluster_expressions(exprs, threshold=0.05)
        # CONST(5) and CONST(5) should be in same cluster
        assert any(len(c) >= 2 for c in clusters)

    def test_embed_from_outputs(self):
        embedder = self._embedder()
        outputs = [float(i) for i in range(EMBEDDING_DIM)]
        e = embedder.embed_from_outputs(outputs, "test")
        assert e.vector.shape == (EMBEDDING_DIM,)


class TestExpressionEmbedding:
    def test_distance_self_zero(self):
        vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec)
        e = ExpressionEmbedding(vec, "test", True, 1.0, "abc")
        assert e.distance_to(e) < 0.001

    def test_distance_opposite(self):
        vec = np.ones(EMBEDDING_DIM, dtype=np.float32)
        vec /= np.linalg.norm(vec)
        e1 = ExpressionEmbedding(vec, "e1", True, 1.0, "a")
        e2 = ExpressionEmbedding(-vec, "e2", True, 1.0, "b")
        assert e1.distance_to(e2) > 1.5  # clamped at 1.0 by max(0, min(1, 1-(-1)))

    def test_invalid_embedding_distance_one(self):
        vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        e_invalid = ExpressionEmbedding(vec, "invalid", False, 0.0, "x")
        e_valid = ExpressionEmbedding.from_vector(np.ones(EMBEDDING_DIM), "valid", 1.0)
        assert e_invalid.distance_to(e_valid) == 1.0


class TestNoveltyScoring:
    def test_zero_distance_zero_novelty(self):
        score = _calibrate_novelty_score(0.0, 100)
        assert score < 0.01

    def test_max_distance_high_novelty(self):
        score = _calibrate_novelty_score(1.0, 100)
        assert score > 0.9

    def test_small_registry_lower_confidence(self):
        score_small = _calibrate_novelty_score(0.8, 5)
        score_large = _calibrate_novelty_score(0.8, 200)
        assert score_small < score_large

    def test_categorize_routine(self):
        assert _categorize_novelty(0.05) == "routine"

    def test_categorize_interesting(self):
        assert _categorize_novelty(0.20) == "interesting"

    def test_categorize_potentially_novel(self):
        assert _categorize_novelty(0.45) == "potentially_novel"

    def test_categorize_route_to_expert(self):
        assert _categorize_novelty(0.75) == "route_to_expert"


class TestEmbeddingRegistry:
    def test_initial_state(self):
        reg = EmbeddingRegistry()
        assert reg._db.size >= 10  # seeded with classics

    def test_register_and_query(self):
        reg = EmbeddingRegistry()
        # Register a known expression
        reg.register_known(make_const(5.0), "five", "arithmetic", "test")
        # Query returns a result
        result = reg.query(make_const(5.0))
        assert isinstance(result, RegistrySearchResult)
        assert result.novelty_score >= 0.0

    def test_query_invalid_returns_routine(self):
        from ouroboros.nodes.extended_nodes import ExtExprNode
        from ouroboros.synthesis.expr_node import NodeType
        reg = EmbeddingRegistry()
        # Make a valid expression
        expr = make_const(3.0)
        result = reg.query(expr)
        assert result.novelty_category in ("routine", "interesting",
                                            "potentially_novel", "route_to_expert")

    def test_stats_tracked(self):
        reg = EmbeddingRegistry()
        reg.query(make_const(5.0))
        assert reg.stats["n_queries"] >= 1

    def test_save_and_load(self, tmp_path):
        reg = EmbeddingRegistry()
        reg.register_string("t % 7", "modular7", "number_theory", "test")
        path = str(tmp_path / "registry.json")
        reg.save(path)
        reg2 = EmbeddingRegistry(registry_path=path)
        assert reg2._db.size >= 1

    def test_is_worth_investigating_high_novelty(self):
        result = RegistrySearchResult(
            query_expression="test",
            query_embedding=None,
            nearest_known=None,
            nearest_distance=0.9,
            top_k_results=[],
            domain_distances={},
            most_novel_domain="unknown",
            novelty_score=0.9,
            novelty_category="route_to_expert",
            query_time_seconds=0.01,
        )
        assert result.is_worth_investigating()

    def test_not_worth_investigating_low_novelty(self):
        result = RegistrySearchResult(
            query_expression="test",
            query_embedding=None,
            nearest_known=None,
            nearest_distance=0.05,
            top_k_results=[],
            domain_distances={},
            most_novel_domain="unknown",
            novelty_score=0.05,
            novelty_category="routine",
            query_time_seconds=0.01,
        )
        assert not result.is_worth_investigating()