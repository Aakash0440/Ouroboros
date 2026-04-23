"""
HierarchicalSearchRouter — Combines all three search complexity solutions.

The three solutions working together:
  1. Grammar constraints    → prune invalid node combinations (10^16x smaller space)
  2. Hierarchical classify  → restrict to relevant node category (further 8x smaller)
  3. Neural prior           → bias sampling toward successful nodes (faster convergence)

The router is the unified entry point for all expression search in the
extended 60-node OUROBOROS system. It replaces both BeamSearchSynthesizer
and SparseBeamSearch as the main search engine.

Usage:
    router = HierarchicalSearchRouter()
    expr = router.search(observations, alphabet_size=7, verbose=True)
    
    # After successful discovery, update the neural prior:
    router.update_prior(observations, expr, reward_bits=45.0)
    
    # Save learned weights for next session:
    router.save_prior("results/node_prior.json")
"""

from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.nodes.extended_nodes import ExtNodeType, NodeCategory
from ouroboros.grammar.math_grammar import MathGrammar, DEFAULT_GRAMMAR
from ouroboros.search.env_classifier import EnvironmentClassifier, MathFamily
from ouroboros.search.neural_prior import NeuralNodePrior
from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
from ouroboros.nodes.extended_nodes import ExtExprNode


@dataclass
class RouterConfig:
    """Configuration for the hierarchical search router."""
    beam_width: int = 25
    max_depth: int = 5
    const_range: int = 50
    max_lag: int = 10
    n_iterations: int = 15
    time_budget_seconds: float = 10.0
    random_seed: int = 42
    
    # Prior learning
    prior_learning_rate: float = 0.05
    prior_path: Optional[str] = None   # load/save learned weights


@dataclass
class RouterResult:
    """Result from the hierarchical search router."""
    expr: Optional[ExtExprNode]
    mdl_cost: float
    math_family: MathFamily
    classification_confidence: float
    categories_searched: List[NodeCategory]
    time_seconds: float
    prior_updated: bool = False

    def description(self) -> str:
        return (
            f"RouterResult: family={self.math_family.name}, "
            f"confidence={self.classification_confidence:.2f}\n"
            f"  MDL cost: {self.mdl_cost:.2f} bits\n"
            f"  Categories: {[c.name for c in self.categories_searched]}\n"
            f"  Time: {self.time_seconds:.2f}s\n"
            f"  Expression: {self.expr.to_string()[:80] if self.expr else 'None'}"
        )


class HierarchicalSearchRouter:
    """
    The unified search entry point for the 60-node OUROBOROS system.
    
    Search protocol:
    1. Classify the observation sequence → MathFamily
    2. Build category-restricted GrammarBeamConfig using classification
    3. Apply neural prior weights to category_weights in config
    4. Run GrammarConstrainedBeam with this config
    5. If result is good, update neural prior
    6. If result is poor and confidence was low, try MIXED (all categories)
    """

    def __init__(self, config: RouterConfig = None):
        self.cfg = config or RouterConfig()
        self._classifier = EnvironmentClassifier()
        self._prior = NeuralNodePrior(
            learning_rate=self.cfg.prior_learning_rate,
            seed=self.cfg.random_seed,
        )
        self._grammar = DEFAULT_GRAMMAR

        # Load prior if path provided
        if self.cfg.prior_path:
            try:
                self._prior.load(self.cfg.prior_path)
            except Exception:
                pass  # Start fresh if load fails

    def search(
        self,
        observations: List[int],
        alphabet_size: int = 13,
        verbose: bool = False,
    ) -> RouterResult:
        """
        Full hierarchical search: classify → restrict → guide → search.
        """
        start = time.time()

        # Step 1: Classify
        float_obs = [float(v) for v in observations]
        classification = self._classifier.classify(float_obs, verbose=verbose)

        if verbose:
            print(f"\n[Router] Family: {classification.primary_family.name} "
                  f"(confidence={classification.classification_confidence:.2f})")
            print(f"[Router] Recommended: {[c.name for c in classification.recommended_categories]}")

        # Step 2: Build beam config with classification-restricted categories
        stats_vector = [
            classification.entropy,
            classification.autocorr_lag1,
            classification.autocorr_lag7,
            classification.deriv_variance,
            classification.monotonicity,
            classification.unique_ratio,
        ]

        # Step 3: Get neural prior weights
        prior_weights = self._prior.get_weights(stats_vector)
        prior_cat_weights = self._prior.get_category_weights(stats_vector)

        # Build category weights for beam config:
        # Start with classification recommendations, scale by neural prior
        category_weights: Dict[NodeCategory, float] = {}
        all_cats = list(NodeCategory)
        for cat in all_cats:
            base = 0.1  # minimal exploration of all categories
            if cat in classification.recommended_categories:
                base = 1.0  # strongly recommended
            prior_scale = prior_cat_weights.get(cat.name, 1.0)
            category_weights[cat] = base * prior_scale

        # Always keep TERMINAL at high weight
        category_weights[NodeCategory.TERMINAL] = 3.0

        # Step 4: Build grammar beam config
        beam_cfg = GrammarBeamConfig(
            beam_width=self.cfg.beam_width,
            max_depth=self.cfg.max_depth,
            const_range=max(self.cfg.const_range, alphabet_size * 2),
            max_lag=self.cfg.max_lag,
            n_iterations=self.cfg.n_iterations,
            random_seed=self.cfg.random_seed,
            grammar=self._grammar,
            category_weights=category_weights,
        )

        # Step 5: Run grammar-constrained beam search
        beam = GrammarConstrainedBeam(beam_cfg)
        expr = beam.search(observations, verbose=verbose)

        elapsed = time.time() - start
        cost = float('inf')

        if expr is not None:
            # Score the result
            from ouroboros.compression.mdl_engine import MDLEngine
            mdl = MDLEngine()
            try:
                preds = [
                    int(round(expr.evaluate(t, observations[:t])))
                    for t in range(len(observations))
                ]
                r = mdl.compute(preds, observations, expr.node_count(), expr.constant_count())
                cost = r.total_mdl_cost
            except Exception:
                cost = float('inf')

        result = RouterResult(
            expr=expr,
            mdl_cost=cost,
            math_family=classification.primary_family,
            classification_confidence=classification.classification_confidence,
            categories_searched=classification.recommended_categories,
            time_seconds=elapsed,
        )

        # Step 6: If poor result and low confidence, try MIXED (fallback)
        if (cost > 500 and classification.classification_confidence < 0.4
                and elapsed < self.cfg.time_budget_seconds * 0.7):
            if verbose:
                print(f"[Router] Poor result + low confidence, trying MIXED fallback")
            fallback_cfg = copy.copy(beam_cfg)
            # Equal weights for all categories
            fallback_cfg.category_weights = {cat: 1.0 for cat in NodeCategory}
            fallback_cfg.category_weights[NodeCategory.TERMINAL] = 3.0
            fallback_beam = GrammarConstrainedBeam(fallback_cfg)
            fallback_expr = fallback_beam.search(observations, verbose=False)
            if fallback_expr is not None:
                try:
                    preds = [int(round(fallback_expr.evaluate(t, observations[:t])))
                             for t in range(len(observations))]
                    r = mdl.compute(preds, observations,
                                   fallback_expr.node_count(), fallback_expr.constant_count())
                    if r.total_mdl_cost < cost:
                        result.expr = fallback_expr
                        result.mdl_cost = r.total_mdl_cost
                        result.categories_searched = list(NodeCategory)
                except Exception:
                    pass

        if verbose:
            print(f"\n[Router] Final: {result.description()}")

        return result

    def update_prior(
        self,
        observations: List[int],
        expr: ExtExprNode,
        reward_bits: float,
    ) -> None:
        """Update the neural prior based on a successful search."""
        float_obs = [float(v) for v in observations]
        classification = self._classifier.classify(float_obs)
        stats = [
            classification.entropy, classification.autocorr_lag1,
            classification.autocorr_lag7, classification.deriv_variance,
            classification.monotonicity, classification.unique_ratio,
        ]
        self._prior.update(stats, expr, reward_bits)

    def save_prior(self, path: str) -> None:
        """Save learned neural prior to disk."""
        self._prior.save(path)

    def load_prior(self, path: str) -> None:
        """Load learned neural prior from disk."""
        self._prior.load(path)

    @property
    def prior_stats(self) -> Any:
        return self._prior.stats