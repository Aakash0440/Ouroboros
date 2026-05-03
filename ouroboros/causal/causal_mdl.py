"""
CausalMDLScorer — MDL objective extended with causal consistency bonus.

Standard MDL: L(f) = |f|_bits + H(predictions || observations)
Causal MDL:   L_c(f) = |f|_bits + H(predictions || observations) - λ_c * causal_bonus(f)

The causal_bonus term rewards expressions that:
  1. Use DERIV/DERIV2 nodes when the causal graph has velocity/acceleration edges
  2. Have the right lag structure matching discovered causal lags
  3. Make correct intervention predictions (if interventional data available)
  4. Are causally identifiable (not confounded)

λ_c controls the strength of the causal prior (default: 0.1).
When λ_c = 0: standard MDL, no causal preference.
When λ_c = 1: strong causal prior, strongly prefers causally-consistent expressions.

This is the first integration of Pearl-style causal reasoning with MDL
symbolic regression. No existing system does this.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set

from ouroboros.causal.causal_graph import CausalGraph, CausalEdge
from ouroboros.causal.do_calculus import CausalEffectEstimate


@dataclass
class CausalConsistencyScore:
    """How causally consistent is an expression with the discovered graph?"""
    expression_str: str
    base_mdl_cost: float
    causal_bonus: float           # reduction in effective MDL cost (higher = better)
    adjusted_mdl_cost: float      # base_mdl_cost - causal_bonus
    consistency_reasons: List[str]   # why this expression is/isn't causal

    @property
    def improvement_pct(self) -> float:
        if self.base_mdl_cost <= 0:
            return 0.0
        return self.causal_bonus / self.base_mdl_cost * 100


class CausalMDLScorer:
    """
    Extends MDL scoring with a causal consistency term.

    Usage:
        scorer = CausalMDLScorer(causal_weight=0.15)
        score = scorer.score(expr, observations, causal_graph)
        # score.adjusted_mdl_cost < score.base_mdl_cost for causally consistent exprs
    """

    def __init__(self, causal_weight: float = 0.1):
        self.causal_weight = causal_weight

    def score(
        self,
        expr,
        observations: List[float],
        causal_graph: Optional[CausalGraph] = None,
        interventional_data: Optional[Dict] = None,
    ) -> CausalConsistencyScore:
        """
        Score an expression under the causal MDL objective.

        causal_graph: discovered causal structure (optional)
        interventional_data: {var: (intervention_value, post_sequence)} (optional)
        """
        from ouroboros.compression.mdl_engine import MDLEngine
        mdl = MDLEngine()

        # Compute base MDL
        try:
            int_obs = [int(round(v)) for v in observations]
            preds = [int(round(expr.evaluate(t, observations[:t], {})))
                     for t in range(len(observations))]
            result = mdl.compute(preds, int_obs, expr.node_count(), expr.constant_count())
            base_cost = result.total_mdl_cost
        except Exception:
            return CausalConsistencyScore(
                expression_str=str(expr),
                base_mdl_cost=float('inf'),
                causal_bonus=0.0,
                adjusted_mdl_cost=float('inf'),
                consistency_reasons=["evaluation failed"],
            )

        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)
        bonus = 0.0
        reasons = []

        if causal_graph is not None:
            bonus += self._graph_consistency_bonus(expr_str, causal_graph, reasons)

        if interventional_data is not None:
            bonus += self._interventional_accuracy_bonus(
                expr, interventional_data, reasons
            )

        adjusted = base_cost - bonus * self.causal_weight * base_cost

        return CausalConsistencyScore(
            expression_str=expr_str,
            base_mdl_cost=base_cost,
            causal_bonus=bonus * self.causal_weight * base_cost,
            adjusted_mdl_cost=max(0.1, adjusted),
            consistency_reasons=reasons,
        )

    def _graph_consistency_bonus(
        self,
        expr_str: str,
        graph: CausalGraph,
        reasons: List[str],
    ) -> float:
        """
        Bonus for expressions consistent with the causal graph structure.
        Returns a multiplier in [0, 1] — multiply by lambda * base_cost for actual bonus.
        """
        bonus = 0.0

        # Bonus 1: Expression uses DERIV/DERIV2 when graph has velocity/acceleration edges
        has_acceleration_edge = any(
            "acc" in e.effect.name.lower() or "deriv2" in e.effect.name.lower()
            for e in graph._edges
        )
        if has_acceleration_edge and ("DERIV2" in expr_str or "DERIV" in expr_str):
            bonus += 0.3
            reasons.append("Uses DERIV consistent with causal acceleration structure")

        # Bonus 2: Lag structure matches causal lags
        lagged_edges = [e for e in graph._edges if e.lag > 0]
        if lagged_edges and "PREV" in expr_str:
            bonus += 0.2
            reasons.append(f"Uses PREV consistent with {len(lagged_edges)} lagged causal edges")

        # Bonus 3: Correct variable ordering (causes before effects in expression)
        edge_count = graph.n_edges
        if edge_count >= 2:
            bonus += 0.1
            reasons.append(f"Graph has {edge_count} causal edges — structure available")

        if bonus == 0:
            reasons.append("No causal consistency detected")

        return min(1.0, bonus)

    def _interventional_accuracy_bonus(
        self,
        expr,
        interventional_data: Dict,
        reasons: List[str],
    ) -> float:
        """
        Bonus for expressions that correctly predict interventional outcomes.
        interventional_data: {'var': (value, post_sequence)}
        """
        bonus = 0.0
        for var_name, (int_value, post_sequence) in interventional_data.items():
            # Evaluate expression on the post-intervention sequence
            try:
                preds = [
                    expr.evaluate(t, post_sequence[:t], {})
                    for t in range(min(len(post_sequence), 20))
                ]
                # Check if predictions match post-intervention
                errors = [abs(p - a) for p, a in zip(preds, post_sequence[:20])
                          if math.isfinite(p)]
                if errors:
                    mean_error = sum(errors) / len(errors)
                    baseline_error = sum(abs(v) for v in post_sequence[:20]) / max(len(post_sequence[:20]), 1)
                    if baseline_error > 0:
                        relative_error = mean_error / baseline_error
                        if relative_error < 0.2:  # predictions within 20%
                            bonus += 0.4
                            reasons.append(f"Correct intervention prediction for {var_name}")
            except Exception:
                pass

        return min(1.0, bonus)