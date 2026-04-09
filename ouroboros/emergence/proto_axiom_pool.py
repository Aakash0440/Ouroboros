# ouroboros/emergence/proto_axiom_pool.py

"""
ProtoAxiomPool — the first repository of emergent mathematical structure.

This is the bridge between Phase 1 (compression) and Phase 2 (proof market).

HOW IT WORKS:
1. Each agent submits its best expression to the pool via submit()
2. After all submissions, detect_consensus() groups by behavioral fingerprint
3. Groups with >= consensus_threshold * num_agents agents → promoted to axiom
4. Each axiom gets a confidence score:
       confidence = (supporting_agents / total_agents) * compression_improvement
5. Axioms are stored in self.axioms with full metadata

WHY CONSENSUS MATTERS:
An expression found by ONE agent could be a fluke — a program that
happens to compress one prefix but fails on new data.
An expression independently discovered by 5/8 agents is much more
likely to reflect real structure. The independence is key — each agent
ran a different search path (different random seeds, different order).

WHY BEHAVIORAL FINGERPRINTING (not string equality):
Agents find "(1 + t * 3) mod 7" and "(t * 3 + 1) mod 7" — same rule,
different syntax. Fingerprinting groups them as ONE axiom.

THE LANDMARK OUTPUT:
When AX_00001 is promoted with expression "(t * 3 + 1) mod 7",
confidence=0.73, supported by 6/8 agents — that is the system's
first piece of discovered mathematics.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import naive_bits, MDLCost
from ouroboros.emergence.fingerprint import behavioral_fingerprint, compression_fingerprint


@dataclass
class ProtoAxiom:
    """
    A candidate mathematical axiom discovered by agent consensus.

    Fields:
        axiom_id:         Unique ID (e.g. "AX_00001")
        expression:       The symbolic expression
        fingerprint:      Behavioral fingerprint tuple (for quick comparison)
        supporting_agents: IDs of agents that found equivalent expressions
        confidence:       Score 0..1 (higher = more trustworthy)
        environment_name: Which environment this came from
        compression_ratio: MDL ratio achieved by this expression
        discovery_step:   Observation count when first promoted
        promotion_time:   Wall-clock time of promotion
    """
    axiom_id: str
    expression: ExprNode
    fingerprint: Tuple[int, ...]
    supporting_agents: List[int]
    confidence: float
    environment_name: str
    compression_ratio: float
    discovery_step: int
    promotion_time: float = field(default_factory=time.time)

    def predicts_sequence_correctly(
        self,
        sequence: List[int],
        alphabet_size: int,
        tolerance: float = 0.05
    ) -> bool:
        """
        Check if this axiom correctly predicts a sequence.

        Returns True if error rate < tolerance.
        Used for OOD testing in Phase 3.
        """
        err = compression_fingerprint(self.expression, sequence, alphabet_size)
        return err < tolerance

    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            'axiom_id': self.axiom_id,
            'expression': self.expression.to_string(),
            'num_supporting_agents': len(self.supporting_agents),
            'supporting_agents': self.supporting_agents,
            'confidence': round(self.confidence, 4),
            'environment': self.environment_name,
            'compression_ratio': round(self.compression_ratio, 4),
            'discovery_step': self.discovery_step,
        }

    def __repr__(self) -> str:
        return (
            f"ProtoAxiom({self.axiom_id}: "
            f"{self.expression.to_string()!r} "
            f"conf={self.confidence:.3f} "
            f"support={len(self.supporting_agents)})"
        )


class ProtoAxiomPool:
    """
    Pool of candidate proto-axioms discovered by agent consensus.

    Args:
        num_agents: Total number of agents in the society
        consensus_threshold: Fraction of agents needed for promotion (default 0.5)
        alphabet_size: Symbol count (for fingerprinting)
        fingerprint_length: Number of timesteps for fingerprinting (default 200)
    """

    def __init__(
        self,
        num_agents: int,
        consensus_threshold: float = 0.5,
        alphabet_size: int = 10,
        fingerprint_length: int = 200,
    ):
        self.num_agents = num_agents
        self.consensus_threshold = consensus_threshold
        self.alphabet_size = alphabet_size
        self.fingerprint_length = fingerprint_length

        # Current round submissions: agent_id → (expression, mdl_cost)
        self.submissions: Dict[int, Tuple[ExprNode, float]] = {}

        # All promoted axioms (never removed, only added)
        self.axioms: List[ProtoAxiom] = []

        # Counter for unique axiom IDs
        self._axiom_counter: int = 0

        # Submission history for analysis
        self.submission_history: List[dict] = []

    def submit(
        self,
        agent_id: int,
        expression: Optional[ExprNode],
        mdl_cost: float,
        step: int,
    ) -> None:
        """
        Agent submits its best expression.

        Called after each agent's search_and_update().
        Expression may be None if agent didn't find a symbolic program.

        Args:
            agent_id: ID of the submitting agent
            expression: Best symbolic expression (or None)
            mdl_cost: MDL cost of this expression
            step: Current observation step
        """
        if expression is None:
            return

        self.submissions[agent_id] = (expression, mdl_cost)
        self.submission_history.append({
            'agent_id': agent_id,
            'expression': expression.to_string(),
            'mdl_cost': round(mdl_cost, 2),
            'step': step,
        })

    def _next_axiom_id(self) -> str:
        self._axiom_counter += 1
        return f"AX_{self._axiom_counter:05d}"

    def detect_consensus(
        self,
        step: int,
        environment_name: str,
        stream_naive_bits: float,
    ) -> List[ProtoAxiom]:
        """
        Find expressions agreed upon by >= threshold of agents.
        Promotes them to proto-axioms.

        Args:
            step: Current observation count
            environment_name: Label for the environment
            stream_naive_bits: Naive description length of the stream
                               (used to compute compression improvement)

        Returns:
            List of newly promoted ProtoAxioms (may be empty)
        """
        if len(self.submissions) < 2:
            return []

        # Group submissions by behavioral fingerprint
        fingerprint_to_group: Dict[Tuple, List[Tuple[int, ExprNode, float]]] = {}

        for agent_id, (expr, cost) in self.submissions.items():
            fp = behavioral_fingerprint(expr, self.alphabet_size, self.fingerprint_length)
            if fp not in fingerprint_to_group:
                fingerprint_to_group[fp] = []
            fingerprint_to_group[fp].append((agent_id, expr, cost))

        min_support = max(2, int(self.consensus_threshold * self.num_agents))
        new_axioms: List[ProtoAxiom] = []

        for fp, group in fingerprint_to_group.items():
            if len(group) < min_support:
                continue

            # Check not already promoted
            already_promoted = any(ax.fingerprint == fp for ax in self.axioms)
            if already_promoted:
                continue

            # Build the axiom
            agent_ids = [g[0] for g in group]
            best_entry = min(group, key=lambda g: g[2])  # lowest cost
            best_cost = best_entry[2]
            best_expr = best_entry[1]

            # Compression improvement vs. naive
            if stream_naive_bits > 0:
                compression_improvement = max(0.0, 1.0 - best_cost / stream_naive_bits)
            else:
                compression_improvement = 0.0

            compression_r = best_cost / stream_naive_bits if stream_naive_bits > 0 else 1.0

            # Confidence: fraction of agents * how much improvement they achieve
            agent_fraction = len(group) / self.num_agents
            confidence = agent_fraction * compression_improvement

            axiom = ProtoAxiom(
                axiom_id=self._next_axiom_id(),
                expression=best_expr,
                fingerprint=fp,
                supporting_agents=agent_ids,
                confidence=confidence,
                environment_name=environment_name,
                compression_ratio=compression_r,
                discovery_step=step,
            )

            self.axioms.append(axiom)
            new_axioms.append(axiom)

        return new_axioms

    def clear_submissions(self) -> None:
        """Clear current round submissions (call between episodes)."""
        self.submissions.clear()

    def get_axioms_sorted(self) -> List[ProtoAxiom]:
        """Return axioms sorted by confidence (highest first)."""
        return sorted(self.axioms, key=lambda ax: ax.confidence, reverse=True)

    def has_axiom_for_expression(self, expr: ExprNode) -> bool:
        """Check if an equivalent expression is already an axiom."""
        fp = behavioral_fingerprint(expr, self.alphabet_size, self.fingerprint_length)
        return any(ax.fingerprint == fp for ax in self.axioms)

    def summary(self) -> str:
        """Return human-readable summary."""
        if not self.axioms:
            return "ProtoAxiomPool: 0 axioms"
        lines = [f"ProtoAxiomPool: {len(self.axioms)} axiom(s)"]
        for ax in self.get_axioms_sorted():
            lines.append(
                f"  {ax.axiom_id}: {ax.expression.to_string()!r}"
                f"  conf={ax.confidence:.3f}"
                f"  support={len(ax.supporting_agents)}/{self.num_agents}"
                f"  ratio={ax.compression_ratio:.4f}"
            )
        return '\n'.join(lines)

    def __len__(self) -> int:
        return len(self.axioms)

    def __repr__(self) -> str:
        return f"ProtoAxiomPool(axioms={len(self.axioms)}, agents={self.num_agents})"