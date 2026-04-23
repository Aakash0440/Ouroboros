"""Tests for EnvironmentClassifier, NeuralNodePrior, and HierarchicalSearchRouter."""
import pytest
from ouroboros.search.env_classifier import (
    EnvironmentClassifier, MathFamily, ClassificationResult,
)
from ouroboros.search.neural_prior import NeuralNodePrior
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.nodes.extended_nodes import ExtNodeType, NodeCategory


class TestEnvironmentClassifier:
    def _clf(self): return EnvironmentClassifier()

    def test_periodic_detects_sine(self):
        import math
        obs = [math.sin(t / 7.0) for t in range(200)]
        result = self._clf().classify(obs)
        assert result.primary_family in (MathFamily.PERIODIC, MathFamily.MIXED)

    def test_random_detects_noise(self):
        import random
        rng = random.Random(42)
        obs = [float(rng.randint(0, 6)) for _ in range(300)]
        result = self._clf().classify(obs)
        assert result.primary_family in (MathFamily.RANDOM, MathFamily.MIXED,
                                          MathFamily.NUMBER_THEOR)

    def test_modular_detects_number_theoretic(self):
        obs = [float((3*t+1) % 7) for t in range(300)]
        result = self._clf().classify(obs)
        # Could be number-theoretic or periodic
        assert result.primary_family in (
            MathFamily.NUMBER_THEOR, MathFamily.PERIODIC, MathFamily.MIXED
        )

    def test_monotone_detects_increasing(self):
        obs = [float(t) for t in range(200)]
        result = self._clf().classify(obs)
        assert result.primary_family in (MathFamily.MONOTONE, MathFamily.MIXED)

    def test_classification_has_all_fields(self):
        obs = [float(t % 7) for t in range(100)]
        result = self._clf().classify(obs)
        assert isinstance(result.entropy, float)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.recommended_categories) > 0

    def test_recommended_categories_are_valid(self):
        obs = [float(t % 7) for t in range(100)]
        result = self._clf().classify(obs)
        for cat in result.recommended_categories:
            assert isinstance(cat, NodeCategory)

    def test_family_scores_sum_positive(self):
        obs = [float(t % 7) for t in range(100)]
        result = self._clf().classify(obs)
        assert sum(result.family_scores.values()) > 0

    def test_short_sequence_returns_mixed(self):
        result = self._clf().classify([1.0, 2.0, 3.0])
        assert result.primary_family == MathFamily.MIXED


class TestNeuralNodePrior:
    def _prior(self): return NeuralNodePrior(seed=42)

    def test_get_weights_returns_all_nodes(self):
        prior = self._prior()
        stats = [0.5, 0.3, 0.1, 0.2, 0.7, 0.4]
        weights = prior.get_weights(stats)
        assert len(weights) == len(ExtNodeType)
        assert all(w > 0 for w in weights.values())

    def test_weights_change_after_update(self):
        from ouroboros.synthesis.expr_node import NodeType
        from ouroboros.nodes.extended_nodes import ExtExprNode
        prior = self._prior()
        stats = [0.5, 0.3, 0.1, 0.2, 0.7, 0.4]
        
        before = prior.get_weights(stats).copy()
        
        # Create a simple expression using ISPRIME
        expr = ExtExprNode(ExtNodeType.ISPRIME)
        expr.left = ExtExprNode.__new__(ExtExprNode)
        expr.left.node_type = NodeType.CONST
        expr.left.value = 7.0
        expr.left.lag = 1; expr.left.state_key = 0; expr.left.window = 10
        expr.left.left = expr.left.right = expr.left.third = None
        expr.left._cache = {}
        expr.right = None; expr.third = None
        
        prior.update(stats, expr, reward=50.0)
        after = prior.get_weights(stats)
        
        # At least ISPRIME weight should have changed
        assert any(before[k] != after[k] for k in before)

    def test_negative_reward_no_update(self):
        from ouroboros.nodes.extended_nodes import ExtExprNode
        prior = self._prior()
        stats = [0.5]*6
        before = prior.get_weights(stats).copy()
        # reward=0 → no update
        prior.update(stats, ExtExprNode(ExtNodeType.SIGN), reward=0.0)
        after = prior.get_weights(stats)
        assert before == after

    def test_top_nodes_returned(self):
        prior = self._prior()
        stats = [0.5]*6
        top = prior.top_nodes_for_stats(stats, top_k=5)
        assert len(top) == 5
        assert all(isinstance(name, str) for name, _ in top)

    def test_stats_track_updates(self):
        from ouroboros.nodes.extended_nodes import ExtExprNode
        prior = self._prior()
        stats = [0.5]*6
        assert prior.stats.n_updates == 0
        prior.update(stats, ExtExprNode(ExtNodeType.SIGN), reward=10.0)
        assert prior.stats.n_updates == 1

    def test_category_weights_returned(self):
        prior = self._prior()
        stats = [0.5]*6
        cat_weights = prior.get_category_weights(stats)
        assert len(cat_weights) == len(NodeCategory)
        assert all(w > 0 for w in cat_weights.values())

    def test_save_load_roundtrip(self, tmp_path):
        prior = self._prior()
        path = str(tmp_path / "prior.json")
        prior.save(path)
        prior2 = NeuralNodePrior()
        prior2.load(path)
        stats = [0.5]*6
        w1 = prior.get_weights(stats)
        w2 = prior2.get_weights(stats)
        for k in w1:
            assert abs(w1[k] - w2.get(k, 0)) < 0.001


class TestHierarchicalSearchRouter:
    def test_search_returns_result(self):
        router = HierarchicalSearchRouter(RouterConfig(
            beam_width=8, max_depth=3, n_iterations=3
        ))
        obs = [(3*t+1)%7 for t in range(100)]
        result = router.search(obs, alphabet_size=7)
        assert result.expr is not None or result.math_family is not None

    def test_result_has_family(self):
        router = HierarchicalSearchRouter(RouterConfig(beam_width=5, n_iterations=2))
        obs = [t % 5 for t in range(80)]
        result = router.search(obs, alphabet_size=5)
        assert isinstance(result.math_family, MathFamily)

    def test_time_recorded(self):
        router = HierarchicalSearchRouter(RouterConfig(beam_width=5, n_iterations=2))
        obs = [t % 5 for t in range(50)]
        result = router.search(obs, alphabet_size=5)
        assert result.time_seconds > 0

    def test_prior_update_runs(self):
        from ouroboros.nodes.extended_nodes import ExtExprNode
        router = HierarchicalSearchRouter(RouterConfig(beam_width=5, n_iterations=2))
        obs = [(3*t+1)%7 for t in range(100)]
        result = router.search(obs, alphabet_size=7)
        if result.expr:
            router.update_prior(obs, result.expr, reward_bits=30.0)
        assert router.prior_stats.n_queries >= 1

    def test_prior_save_runs(self, tmp_path):
        router = HierarchicalSearchRouter(RouterConfig(beam_width=5, n_iterations=2))
        path = str(tmp_path / "prior.json")
        router.save_prior(path)
        import os
        assert os.path.exists(path)
        