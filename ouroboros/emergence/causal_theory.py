"""
CausalTheory — a collection of scale-tagged axioms forming a theory.

In Phase 1, agents had a single best expression.
In Phase 3, agents maintain a THEORY: one axiom per scale,
with cross-scale consistency scores.

Structure:
    theory = {
        scale_1:  Axiom("(t * 3 + 1) mod 7", confidence=0.92),
        scale_4:  Axiom("(k * 11 + 4) mod 7", confidence=0.71),
        scale_16: Axiom("(k * 2 + 1) mod 4", confidence=0.58),
    }

A theory is "complete" when:
    - All configured scales have a non-null axiom
    - Mean confidence > threshold
    - Cross-scale consistency > threshold

A theory is "richer" than another if:
    - Mean compression ratio is lower across all scales
    - Cross-scale consistency is higher

The recursive ascent in Phase 3 measures THEORY richness, not just
single-expression compression.

The landmark experiment tests whether two independent theories
(derived from ModArith(7) and ModArith(11)) can be COMBINED into
a joint theory that captures the Chinese Remainder Theorem.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.emergence.proto_axiom_pool import ProtoAxiom


@dataclass
class ScaleAxiom:
    """
    One axiom within a CausalTheory, at a specific scale.

    Fields:
        expression: Symbolic expression for this scale
        scale: Temporal scale (window size)
        confidence: How certain we are this axiom is correct [0,1]
        compression_ratio: MDL ratio at this scale
        discovery_step: When this was found
        times_survived_market: How many proof market rounds without rejection
    """
    expression: ExprNode
    scale: int
    confidence: float
    compression_ratio: float
    discovery_step: int = 0
    times_survived_market: int = 0
    times_challenged: int = 0

    def challenge_survival_rate(self) -> float:
        if self.times_challenged == 0:
            return 1.0
        return self.times_survived_market / self.times_challenged

    def __repr__(self) -> str:
        return (f"ScaleAxiom(scale={self.scale}, "
                f"expr={self.expression.to_string()!r}, "
                f"conf={self.confidence:.3f}, "
                f"ratio={self.compression_ratio:.4f}, "
                f"survived={self.times_survived_market}/{self.times_challenged})")


class CausalTheory:
    """
    A multi-scale mathematical theory held by an agent.

    Manages a dictionary of scale → ScaleAxiom.
    Tracks cross-scale consistency.
    Provides theory-level metrics: completeness, richness, consistency.

    Args:
        scales: Temporal scales this theory covers
        alphabet_size: Symbol alphabet size
        consistency_test_length: Symbols used for cross-scale checks
    """

    def __init__(
        self,
        scales: List[int],
        alphabet_size: int,
        consistency_test_length: int = 100
    ):
        self.scales = scales
        self.alphabet_size = alphabet_size
        self.consistency_test_length = consistency_test_length
        self.axioms: Dict[int, Optional[ScaleAxiom]] = {s: None for s in scales}
        self._consistency_score: float = 0.0

    def update_scale(
        self,
        scale: int,
        expression: ExprNode,
        compression_ratio: float,
        confidence: float,
        step: int = 0
    ) -> None:
        """Update or add an axiom at a given scale."""
        if scale not in self.scales:
            return
        existing = self.axioms.get(scale)
        if existing and existing.compression_ratio <= compression_ratio:
            return  # Don't downgrade
        self.axioms[scale] = ScaleAxiom(
            expression=expression,
            scale=scale,
            confidence=confidence,
            compression_ratio=compression_ratio,
            discovery_step=step
        )

    def record_market_result(self, scale: int, survived: bool) -> None:
        """Record a proof market challenge result for a scale."""
        ax = self.axioms.get(scale)
        if ax is None:
            return
        ax.times_challenged += 1
        if survived:
            ax.times_survived_market += 1
            ax.confidence = min(1.0, ax.confidence + 0.05)
        else:
            ax.confidence = max(0.0, ax.confidence - 0.15)

    def is_complete(self, min_confidence: float = 0.50) -> bool:
        """All scales have axioms with confidence above threshold."""
        return all(
            ax is not None and ax.confidence >= min_confidence
            for ax in self.axioms.values()
        )

    def mean_compression_ratio(self) -> float:
        """Mean compression ratio across all active scales."""
        ratios = [ax.compression_ratio for ax in self.axioms.values()
                  if ax is not None]
        return float(np.mean(ratios)) if ratios else 1.0

    def mean_confidence(self) -> float:
        """Mean confidence across all active scales."""
        confs = [ax.confidence for ax in self.axioms.values()
                 if ax is not None]
        return float(np.mean(confs)) if confs else 0.0

    def compute_cross_scale_consistency(self) -> float:
        """
        Cross-scale consistency: do axioms at different scales agree?

        Method: generate predictions from scale-1 axiom, aggregate them,
        compare to scale-k axiom predictions on aggregated data.

        Returns score in [0, 1]. Higher = more consistent.
        Updates self._consistency_score.
        """
        scale1_axiom = self.axioms.get(1)
        if scale1_axiom is None:
            return 0.0

        scores = []
        n = self.consistency_test_length
        scale1_preds = scale1_axiom.expression.predict_sequence(n, self.alphabet_size)

        for scale, axiom in self.axioms.items():
            if scale == 1 or axiom is None:
                continue
            from ouroboros.compression.hierarchical_mdl import aggregate_sequence
            agg_scale1 = aggregate_sequence(scale1_preds, scale, self.alphabet_size)
            agg_len = len(agg_scale1)
            if agg_len == 0:
                continue
            scale_preds = axiom.expression.predict_sequence(agg_len, self.alphabet_size)
            matches = sum(p == a for p, a in zip(scale_preds, agg_scale1))
            scores.append(matches / agg_len)

        self._consistency_score = float(np.mean(scores)) if scores else 0.0
        return self._consistency_score

    @property
    def consistency_score(self) -> float:
        return self._consistency_score

    def richness_score(self) -> float:
        """
        Theory richness = (1 - mean_ratio) * mean_confidence * consistency
        Range [0, 1]. Higher = richer, more verified, more consistent theory.
        """
        ratio_score = max(0.0, 1.0 - self.mean_compression_ratio())
        conf_score = self.mean_confidence()
        cons_score = self._consistency_score
        return ratio_score * conf_score * (0.5 + 0.5 * cons_score)

    def active_axioms(self) -> List[ScaleAxiom]:
        return [ax for ax in self.axioms.values() if ax is not None]

    def best_expression(self) -> Optional[ExprNode]:
        """Return the expression from the lowest-ratio scale."""
        active = self.active_axioms()
        if not active:
            return None
        return min(active, key=lambda ax: ax.compression_ratio).expression

    def to_dict(self) -> dict:
        return {
            'scales': self.scales,
            'axioms': {
                str(s): {
                    'expression': ax.expression.to_string(),
                    'confidence': round(ax.confidence, 4),
                    'compression_ratio': round(ax.compression_ratio, 6),
                    'times_survived': ax.times_survived_market,
                    'times_challenged': ax.times_challenged,
                }
                for s, ax in self.axioms.items()
                if ax is not None
            },
            'mean_ratio': round(self.mean_compression_ratio(), 4),
            'mean_confidence': round(self.mean_confidence(), 4),
            'consistency': round(self._consistency_score, 4),
            'richness': round(self.richness_score(), 4),
        }

    def summary(self) -> str:
        lines = [f"CausalTheory (scales={self.scales}):"]
        for s in self.scales:
            ax = self.axioms.get(s)
            if ax:
                lines.append(f"  Scale {s:3d}: {ax.expression.to_string()!r}  "
                              f"conf={ax.confidence:.3f}  ratio={ax.compression_ratio:.4f}  "
                              f"survived={ax.times_survived_market}/{ax.times_challenged}")
            else:
                lines.append(f"  Scale {s:3d}: [no axiom]")
        lines.append(f"  Richness={self.richness_score():.4f}  "
                     f"Consistency={self._consistency_score:.4f}  "
                     f"Complete={self.is_complete()}")
        return '\n'.join(lines)