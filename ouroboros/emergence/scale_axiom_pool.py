"""
ScaleAxiomPool — ProtoAxiomPool extended with scale tagging.

Each axiom is tagged with its emergence scale:
    scale=1   → axiom describes symbol-level structure
    scale=16  → axiom describes window-level structure

Axioms at different scales can COEXIST without conflict.
An environment can have one axiom at scale 1 AND another at scale 16.
These are complementary mathematical descriptions, not competing ones.

This is the multi-scale analog of the Phase 1 ProtoAxiomPool.
In Phase 2, the proof market will test axioms at EACH scale separately.

Multi-scale consistency bonus:
    If the same behavioral fingerprint appears at multiple scales,
    the axiom gets a consistency bonus: confidence multiplied by
    (1 + 0.2 * num_scales_confirmed).
    An axiom confirmed at 3 scales gets 60% confidence bonus.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ouroboros.emergence.proto_axiom_pool import ProtoAxiomPool, ProtoAxiom
from ouroboros.compression.program_synthesis import ExprNode


@dataclass
class ScaleTaggedAxiom:
    """
    A proto-axiom with scale information.

    Extends ProtoAxiom with:
        emergence_scale: Which temporal scale this was found at
        confirmed_at_scales: All scales where this fingerprint appears
        consistency_bonus: Confidence boost from multi-scale confirmation
    """
    base_axiom: ProtoAxiom
    emergence_scale: int
    confirmed_at_scales: List[int] = field(default_factory=list)
    consistency_bonus: float = 0.0

    @property
    def adjusted_confidence(self) -> float:
        """Confidence with multi-scale consistency bonus applied."""
        bonus = 1.0 + 0.2 * (len(self.confirmed_at_scales) - 1)
        return min(1.0, self.base_axiom.confidence * bonus)

    @property
    def axiom_id(self) -> str:
        return f"{self.base_axiom.axiom_id}_s{self.emergence_scale}"

    def __repr__(self) -> str:
        return (f"ScaleTaggedAxiom({self.axiom_id}: "
                f"{self.base_axiom.expression.to_string()!r} "
                f"scale={self.emergence_scale} "
                f"adj_conf={self.adjusted_confidence:.3f})")


class ScaleAxiomPool:
    """
    Manages proto-axioms across all temporal scales.

    Maintains one ProtoAxiomPool per scale.
    After detection, cross-references fingerprints across scales
    to award consistency bonuses.

    Usage:
        pool = ScaleAxiomPool(scales=[1, 4, 16], num_agents=6, alphabet_size=4)

        # Submit per-scale expressions from HierarchicalAgents
        for agent in agents:
            for scale, expr in agent.get_all_scale_expressions().items():
                pool.submit(scale, agent.agent_id, expr, cost, step)

        # Detect consensus at each scale
        new_axioms = pool.detect_all_scales(step, env_name, naive_bits_per_scale)
    """

    def __init__(
        self,
        scales: List[int],
        num_agents: int,
        consensus_threshold: float = 0.5,
        alphabet_size: int = 4,
        fingerprint_length: int = 80
    ):
        self.scales = scales
        self.num_agents = num_agents
        self.alphabet_size = alphabet_size

        # One pool per scale
        self.scale_pools: Dict[int, ProtoAxiomPool] = {
            scale: ProtoAxiomPool(
                num_agents=num_agents,
                consensus_threshold=consensus_threshold,
                alphabet_size=alphabet_size,
                fingerprint_length=fingerprint_length
            )
            for scale in scales
        }

        # All promoted scale-tagged axioms
        self.scale_axioms: List[ScaleTaggedAxiom] = []

    def submit(
        self,
        scale: int,
        agent_id: int,
        expression: Optional[ExprNode],
        mdl_cost: float,
        step: int = 0
    ) -> None:
        """Submit an expression for a specific scale."""
        if scale not in self.scale_pools:
            return
        self.scale_pools[scale].submit(agent_id, expression, mdl_cost, step)

    def clear_all(self) -> None:
        """Clear submissions at all scales."""
        for pool in self.scale_pools.values():
            pool.clear_submissions()

    def detect_all_scales(
        self,
        step: int,
        environment_name: str,
        naive_bits_per_scale: Dict[int, float]
    ) -> List[ScaleTaggedAxiom]:
        """
        Run consensus detection at every scale.
        Awards consistency bonuses for axioms confirmed at multiple scales.

        Returns: list of newly promoted ScaleTaggedAxioms.
        """
        newly_promoted: List[ScaleTaggedAxiom] = []

        # Detect at each scale
        for scale in self.scales:
            pool = self.scale_pools[scale]
            nb = naive_bits_per_scale.get(scale, 1000.0)
            new_axioms = pool.detect_consensus(
                step=step,
                environment_name=f"{environment_name}@scale{scale}",
                total_naive_bits=nb
            )
            for ax in new_axioms:
                tagged = ScaleTaggedAxiom(
                    base_axiom=ax,
                    emergence_scale=scale,
                    confirmed_at_scales=[scale]
                )
                self.scale_axioms.append(tagged)
                newly_promoted.append(tagged)

        # Cross-scale consistency bonus
        self._update_consistency_bonuses()

        return newly_promoted

    def _update_consistency_bonuses(self) -> None:
        """
        Check if any axioms have the same fingerprint across scales.
        Award consistency bonuses.
        """
        from collections import defaultdict
        fp_to_axioms: Dict[tuple, List[ScaleTaggedAxiom]] = defaultdict(list)

        for tagged_ax in self.scale_axioms:
            fp = tagged_ax.base_axiom.fingerprint
            fp_to_axioms[fp].append(tagged_ax)

        for fp, axiom_list in fp_to_axioms.items():
            scales_confirmed = sorted(set(ax.emergence_scale for ax in axiom_list))
            for ax in axiom_list:
                ax.confirmed_at_scales = scales_confirmed

    def best_axiom(self) -> Optional[ScaleTaggedAxiom]:
        """Return highest adjusted-confidence axiom."""
        if not self.scale_axioms:
            return None
        return max(self.scale_axioms, key=lambda ax: ax.adjusted_confidence)

    def axioms_by_scale(self, scale: int) -> List[ScaleTaggedAxiom]:
        """Return all axioms that emerged at a given scale."""
        return [ax for ax in self.scale_axioms if ax.emergence_scale == scale]

    def summary(self) -> str:
        lines = [f"ScaleAxiomPool: {len(self.scale_axioms)} axiom(s) across {len(self.scales)} scales"]
        for ax in sorted(self.scale_axioms, key=lambda x: -x.adjusted_confidence):
            lines.append(
                f"  [{ax.axiom_id}] {ax.base_axiom.expression.to_string()!r}  "
                f"scale={ax.emergence_scale}  "
                f"adj_conf={ax.adjusted_confidence:.3f}  "
                f"confirmed_at={ax.confirmed_at_scales}"
            )
        return '\n'.join(lines)