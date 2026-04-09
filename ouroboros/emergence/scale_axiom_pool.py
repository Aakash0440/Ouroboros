from dataclasses import dataclass, field
from typing import List, Dict, Optional
from ouroboros.emergence.proto_axiom_pool import ProtoAxiomPool, ProtoAxiom


@dataclass
class ScaleTaggedAxiom:
    base_axiom: ProtoAxiom
    emergence_scale: int
    confirmed_at_scales: List[int] = field(default_factory=list)

    @property
    def adjusted_confidence(self) -> float:
        bonus = 1.0 + 0.2 * (len(self.confirmed_at_scales) - 1)
        return min(1.0, self.base_axiom.confidence * bonus)

    @property
    def axiom_id(self) -> str:
        return f"{self.base_axiom.axiom_id}_s{self.emergence_scale}"


class ScaleAxiomPool:
    def __init__(
        self,
        scales: List[int],
        num_agents: int,
        consensus_threshold: float = 0.5,
        alphabet_size: int = 4,
        fingerprint_length: int = 80
    ):
        self.scales = scales

        self.scale_pools: Dict[int, ProtoAxiomPool] = {
            scale: ProtoAxiomPool(
                num_agents=num_agents,
                consensus_threshold=consensus_threshold,
                alphabet_size=alphabet_size,
                fingerprint_length=fingerprint_length
            )
            for scale in scales
        }

        self.scale_axioms: List[ScaleTaggedAxiom] = []

    def submit(self, scale, agent_id, expression, mdl_cost, step=0):
        if scale in self.scale_pools:
            self.scale_pools[scale].submit(agent_id, expression, mdl_cost, step)

    def clear_all(self):
        for pool in self.scale_pools.values():
            pool.clear_submissions()

    def detect_all_scales(self, step, environment_name, naive_bits_per_scale):
        new_axioms = []

        for scale in self.scales:
            pool = self.scale_pools[scale]

            detected = pool.detect_consensus(
                step=step,
                environment_name=f"{environment_name}@scale{scale}",
                total_naive_bits=naive_bits_per_scale.get(scale, 1000.0)
            )

            for ax in detected:
                tagged = ScaleTaggedAxiom(
                    base_axiom=ax,
                    emergence_scale=scale,
                    confirmed_at_scales=[scale]
                )
                self.scale_axioms.append(tagged)
                new_axioms.append(tagged)

        self._update_consistency()

        return new_axioms

    def _update_consistency(self):
        from collections import defaultdict

        fp_map = defaultdict(list)

        for ax in self.scale_axioms:
            fp_map[ax.base_axiom.fingerprint].append(ax)

        for group in fp_map.values():
            scales = sorted({ax.emergence_scale for ax in group})
            for ax in group:
                ax.confirmed_at_scales = scales

    def summary(self):
        if not self.scale_axioms:
            return "ScaleAxiomPool: 0 axiom(s)"

        lines = [f"ScaleAxiomPool: {len(self.scale_axioms)} axiom(s)"]

        for ax in self.scale_axioms:
            lines.append(
                f"{ax.axiom_id} scale={ax.emergence_scale} conf={ax.adjusted_confidence:.2f}"
            )

        return "\n".join(lines)