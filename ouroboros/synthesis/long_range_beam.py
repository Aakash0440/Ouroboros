"""
LongRangeBeamSearch — Beam search with PREV nodes up to lag=20.

Extends BeamSearchSynthesizer (Day 2) by:
  1. Allowing PREV(k) for k in 1..max_lag (previously capped at 3)
  2. Prioritizing PREV nodes in the warm-start seeds
  3. Using RecurrenceDetector as a preprocessing step:
     - If BM finds a recurrence → use it as the starting point
     - If BM fails → fall back to standard beam search

The key insight: for recurrence sequences (Tribonacci, Lucas, etc.),
BM finds the answer in O(n) time. Beam search is only needed when
BM fails (nonlinear structure, or composite modulus).
"""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional

from ouroboros.synthesis.expr_node import ExprNode, NodeType
from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
from ouroboros.synthesis.mcmc_refiner import MCMCRefiner, MCMCConfig
from ouroboros.compression.mdl_engine import MDLEngine, MDLResult
from ouroboros.emergence.recurrence_detector import RecurrenceDetector, RecurrenceAxiom


@dataclass
class LongRangeBeamConfig:
    """Configuration for long-range beam search."""
    beam_width: int = 25
    max_lag: int = 20           # the key extension: up to lag 20
    const_range: int = 30
    max_depth: int = 5
    mcmc_iterations: int = 150
    use_bm_warmstart: bool = True  # use BM before beam search
    bm_accuracy_threshold: float = 0.95
    random_seed: int = 42


@dataclass
class LongRangeResult:
    """Result of long-range beam search."""
    best_expr: Optional[ExprNode]
    best_mdl_cost: float
    discovery_method: str   # "BerlekampMassey", "BeamSearch", or "Hybrid"
    recurrence_axiom: Optional[RecurrenceAxiom] = None


def recurrence_to_expr(axiom: RecurrenceAxiom) -> Optional[ExprNode]:
    """
    Convert a RecurrenceAxiom to an ExprNode expression tree.
    
    (c1*PREV(1) + c2*PREV(2) + ... + ck*PREV(k)) % modulus
    """
    nonzero = [(i+1, c) for i, c in enumerate(axiom.coefficients) if c != 0]
    if not nonzero:
        return ExprNode(NodeType.CONST, value=0)

    # Build sum of terms
    def make_term(lag: int, coeff: int) -> ExprNode:
        prev_node = ExprNode(NodeType.PREV, lag=lag)
        if coeff == 1:
            return prev_node
        return ExprNode(
            NodeType.MUL,
            left=ExprNode(NodeType.CONST, value=coeff),
            right=prev_node,
        )

    terms = [make_term(lag, c) for lag, c in nonzero]
    
    # Fold terms with ADD
    acc = terms[0]
    for term in terms[1:]:
        acc = ExprNode(NodeType.ADD, left=acc, right=term)

    # Wrap with MOD
    return ExprNode(
        NodeType.MOD,
        left=acc,
        right=ExprNode(NodeType.CONST, value=axiom.modulus),
    )


class LongRangeBeamSearch:
    """
    Beam search supporting PREV(k) for k up to max_lag.
    
    Algorithm:
    1. Try Berlekamp-Massey first (O(n), finds linear recurrences)
    2. If BM succeeds with accuracy > threshold → return BM result
    3. Otherwise: run standard beam search with extended PREV nodes
    """

    def __init__(self, config: LongRangeBeamConfig = None):
        self.cfg = config or LongRangeBeamConfig()
        self._bm = RecurrenceDetector(
            max_order=self.cfg.max_lag,
            accuracy_threshold=self.cfg.bm_accuracy_threshold,
        )
        self._mdl = MDLEngine()

    def search(
        self,
        observations: List[int],
        modulus: int,
        environment_name: str = "unknown",
        verbose: bool = False,
    ) -> LongRangeResult:
        """
        Search for the best expression explaining the observations.
        """
        # ── Phase 1: Berlekamp-Massey ─────────────────────────────────────
        if self.cfg.use_bm_warmstart and len(observations) >= 50:
            axiom = self._bm.detect(observations, modulus, environment_name)
            if axiom is not None:
                expr = recurrence_to_expr(axiom)
                if expr is not None:
                    cost = self._score(expr, observations)
                    if verbose:
                        print(f"  BM found recurrence: {axiom.expression_str}")
                        print(f"  MDL cost: {cost:.2f}, fit_error: {axiom.fit_error:.6f}")
                    return LongRangeResult(
                        best_expr=expr,
                        best_mdl_cost=cost,
                        discovery_method="BerlekampMassey",
                        recurrence_axiom=axiom,
                    )

        # ── Phase 2: Beam search with extended PREV ───────────────────────
        if verbose:
            print(f"  BM failed, running beam search with max_lag={self.cfg.max_lag}")

        # Build warm-start seeds: PREV(1) + PREV(2), PREV(1) + PREV(2) + PREV(3), etc.
        seeds = self._make_recurrence_seeds(modulus)

        beam_cfg = BeamConfig(
            beam_width=self.cfg.beam_width,
            const_range=self.cfg.const_range,
            max_depth=self.cfg.max_depth,
            max_lag=self.cfg.max_lag,
            mcmc_iterations=self.cfg.mcmc_iterations,
            random_seed=self.cfg.random_seed,
            seed_expressions=seeds,
        )
        synthesizer = BeamSearchSynthesizer(beam_cfg)
        best_expr = synthesizer.search(observations)

        if best_expr is None:
            return LongRangeResult(
                best_expr=None,
                best_mdl_cost=float('inf'),
                discovery_method="BeamSearch",
            )

        cost = self._score(best_expr, observations)
        return LongRangeResult(
            best_expr=best_expr,
            best_mdl_cost=cost,
            discovery_method="BeamSearch",
        )

    def _score(self, expr: ExprNode, observations: List[int]) -> float:
        """Score an expression under MDL."""
        predictions = [
            expr.evaluate(t, observations[:t])
            for t in range(len(observations))
        ]
        result = self._mdl.compute(
            predictions, observations,
            expr.node_count(), expr.constant_count()
        )
        return result.total_mdl_cost

    def _make_recurrence_seeds(self, modulus: int) -> List[ExprNode]:
        """
        Build warm-start seeds for common recurrence patterns.
        """
        seeds = []
        mod_node = ExprNode(NodeType.CONST, value=modulus)

        # Fibonacci-type: PREV(1) + PREV(2) (mod N)
        fib = ExprNode(NodeType.MOD,
            left=ExprNode(NodeType.ADD,
                left=ExprNode(NodeType.PREV, lag=1),
                right=ExprNode(NodeType.PREV, lag=2),
            ),
            right=mod_node,
        )
        seeds.append(fib)

        # Tribonacci-type: PREV(1) + PREV(2) + PREV(3) (mod N)
        trib = ExprNode(NodeType.MOD,
            left=ExprNode(NodeType.ADD,
                left=ExprNode(NodeType.ADD,
                    left=ExprNode(NodeType.PREV, lag=1),
                    right=ExprNode(NodeType.PREV, lag=2),
                ),
                right=ExprNode(NodeType.PREV, lag=3),
            ),
            right=copy.deepcopy(mod_node),
        )
        seeds.append(trib)

        # Tetranacci-type: sum of last 4
        for order in range(4, min(self.cfg.max_lag + 1, 8)):
            acc = ExprNode(NodeType.PREV, lag=1)
            for lag in range(2, order + 1):
                acc = ExprNode(NodeType.ADD,
                    left=acc,
                    right=ExprNode(NodeType.PREV, lag=lag),
                )
            seeds.append(ExprNode(NodeType.MOD, left=acc, right=copy.deepcopy(mod_node)))

        return seeds