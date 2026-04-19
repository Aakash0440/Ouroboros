"""
SelfModifyingAgent — agent that can propose modifications to its own program.

This is the Gödel Machine component of OUROBOROS.

How self-modification works:
    1. Agent has a current best expression (from Phase 1 synthesis)
    2. Agent runs beam search + MCMC on new data → finds candidate improvement
    3. If candidate has lower MDL cost → propose to proof market
    4. If market approves AND OOD test passes → agent permanently upgrades
    5. Agent's program is now richer → reveals more structure → better search
    6. GOTO 1 (recursive ascent)

What makes this a Gödel Machine improvement:
    - Original GM: agent verifies its own modification (circular)
    - OUROBOROS: external society verifies modification (non-circular)
    - Original GM: single agent, no external consistency check
    - OUROBOROS: OOD pressure ensures generalization, not overfitting

The recursive improvement is measurable:
    Each approved round → compression ratio decreases
    The compression curve over rounds is the "recursive ascent" signal

Args:
    agent_id: Unique identifier
    alphabet_size: Symbol alphabet size
    modification_threshold: MDL improvement needed to propose (bits)
    All other args forwarded to SynthesisAgent
"""

from typing import List, Optional, Tuple
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import MDLCost, naive_bits
from ouroboros.utils.logger import get_logger


class ModificationProposal:
    """
    A proposed self-modification ready for the proof market.

    Created by SelfModifyingAgent.generate_proposal().
    Passed to ProofMarket.propose().
    """

    def __init__(
        self,
        agent_id: int,
        current_expr: ExprNode,
        proposed_expr: ExprNode,
        current_cost: float,
        proposed_cost: float,
        test_sequence: List[int],
        alphabet_size: int
    ):
        self.agent_id = agent_id
        self.current_expr = current_expr
        self.proposed_expr = proposed_expr
        self.improvement_bits = current_cost - proposed_cost
        self.test_sequence = test_sequence
        self.alphabet_size = alphabet_size

    def is_improvement(self) -> bool:
        return self.improvement_bits > 0

    def __repr__(self) -> str:
        return (f"ModProposal(agent={self.agent_id}, "
                f"Δ={self.improvement_bits:.1f}bits, "
                f"{self.current_expr.to_string()!r} → "
                f"{self.proposed_expr.to_string()!r})")


class SelfModifyingAgent(SynthesisAgent):
    """
    Agent with self-modification capability for the proof market.

    Extends SynthesisAgent with:
    1. generate_proposal(): produce a modification proposal for the market
    2. apply_approved_modification(): update own program after market approval
    3. track_modification_history(): log all modifications for analysis

    Args:
        agent_id: Unique identifier
        alphabet_size: Symbol alphabet size
        modification_threshold: Minimum MDL improvement (bits) to propose
        max_proposals_per_round: Throttle proposal rate
        (all other args forwarded to SynthesisAgent)
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        modification_threshold: float = 10.0,
        max_proposals_per_round: int = 1,
        **kwargs
    ):
        super().__init__(agent_id, alphabet_size, **kwargs)
        self.modification_threshold = modification_threshold
        self.max_proposals_per_round = max_proposals_per_round
        self.logger = get_logger(f'SelfModAgent_{agent_id}')

        # Modification history: list of (step, old_expr, new_expr, improvement, outcome)
        self.modification_history: List[Tuple] = []
        self.approved_modifications: int = 0
        self.rejected_modifications: int = 0
        self.revoked_modifications: int = 0

    def generate_proposal(
        self,
        new_data: List[int],
        min_improvement_bits: Optional[float] = None
    ) -> Optional[ModificationProposal]:
        """
        Search for an improvement on new_data and generate a proposal.

        Process:
        1. Run beam + MCMC search on new_data
        2. Compare to current program under MDL
        3. If improvement > threshold, create proposal
        4. Return None if no significant improvement found

        Args:
            new_data: New observations to search on
            min_improvement_bits: Override threshold (optional)

        Returns:
            ModificationProposal if improvement found, else None
        """
        if not self.best_expression:
            return None

        threshold = min_improvement_bits or self.modification_threshold

        # Run synthesis on new data
        candidate_expr, candidate_cost = self.synthesizer.search(
            new_data[:min(500, len(new_data))]
        )
        if self.refiner is not None:
            refined_expr, refined_cost = self.refiner.refine(candidate_expr, new_data)
            if refined_cost < candidate_cost:
                candidate_expr = refined_expr
                candidate_cost = refined_cost

        # Compute current program's cost on new data
        mdl = MDLCost()
        n = len(new_data)
        current_preds = self.best_expression.predict_sequence(n, self.alphabet_size)
        current_cost = mdl.total_cost(
            self.best_expression.to_bytes(),
            current_preds, new_data, self.alphabet_size
        )

        improvement_bits = current_cost - candidate_cost

        if improvement_bits < threshold:
            return None  # Not worth proposing

        # Check it's not already the same expression
        if (candidate_expr.to_string() == self.best_expression.to_string()):
            return None

        proposal = ModificationProposal(
            agent_id=self.agent_id,
            current_expr=self.best_expression,
            proposed_expr=candidate_expr,
            current_cost=current_cost,
            proposed_cost=candidate_cost,
            test_sequence=new_data[:200],
            alphabet_size=self.alphabet_size
        )

        self.logger.debug(
            f"Agent {self.agent_id} generated proposal: "
            f"{proposal.improvement_bits:.1f} bit improvement"
        )

        return proposal

    def apply_approved_modification(
        self,
        proposal: ModificationProposal,
        step: int
    ) -> None:
        """
        Permanently update agent's program after market + OOD approval.

        This is the irreversible self-modification step.
        After this call, the agent's best_expression is updated forever.
        The old program is logged in modification_history.

        Args:
            proposal: The approved proposal
            step: Current observation step (for logging)
        """
        old_expr = self.best_expression
        self.best_expression = proposal.proposed_expr
        self._using_symbolic = True
        self.approved_modifications += 1

        self.modification_history.append((
            step,
            old_expr.to_string() if old_expr else 'None',
            proposal.proposed_expr.to_string(),
            proposal.improvement_bits,
            'APPROVED'
        ))

        self.logger.info(
            f"Agent {self.agent_id}: self-modification APPLIED at step {step}. "
            f"Program: {old_expr.to_string() if old_expr else 'None'!r} → "
            f"{proposal.proposed_expr.to_string()!r}"
        )

    def record_rejection(
        self,
        proposal: ModificationProposal,
        step: int,
        reason: str = 'market_rejected'
    ) -> None:
        """Log a rejected modification."""
        self.modification_history.append((
            step,
            proposal.current_expr.to_string(),
            proposal.proposed_expr.to_string(),
            proposal.improvement_bits,
            f'REJECTED({reason})'
        ))
        if reason == 'ood_failed':
            self.revoked_modifications += 1
        else:
            self.rejected_modifications += 1

    def recursive_ascent_score(self) -> float:
        """
        Measure recursive self-improvement progress.

        Score = total approved improvement bits / initial naive bits
        Higher = more recursive improvement achieved.
        """
        approved = [(step, imp) for step, old, new, imp, outcome
                    in self.modification_history
                    if 'APPROVED' in outcome]
        if not approved:
            return 0.0
        total_improvement = sum(imp for _, imp in approved)
        # Normalize by naive bits of a typical 200-symbol stream
        return total_improvement / max(
            naive_bits(list(range(self.alphabet_size)) * 30, self.alphabet_size),
            1.0
        )

    def modification_summary(self) -> str:
        lines = [f"Agent {self.agent_id} modification history:"]
        for step, old, new, imp, outcome in self.modification_history:
            lines.append(
                f"  step={step}: {old!r} → {new!r}  "
                f"Δ={imp:.1f}bits  [{outcome}]"
            )
        lines.append(
            f"  Approved={self.approved_modifications}  "
            f"Rejected={self.rejected_modifications}  "
            f"Revoked={self.revoked_modifications}"
        )
        return '\n'.join(lines)