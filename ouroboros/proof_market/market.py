"""
The OUROBOROS Proof Market.

Central mechanism of Phase 2. Coordinates adversarial verification
of proposed agent self-modifications using the commit-reveal protocol.

Full round lifecycle:
    1. PROPOSE:    Agent A stakes bounty, proposes new expression
    2. COMMIT:     All other agents search for CEs, submit hashes
    3. REVEAL:     Commit window closes, all agents reveal CEs + salts
    4. VERIFY:     Hashes checked, valid reveals identified
    5. ADJUDICATE: No valid CEs → APPROVED + OOD test
                   Valid CEs → REJECTED + bounty distributed
    6. OOD_TEST:   Approved proposals tested on never-seen environment
                   Fails OOD → REVOKED

Bounty economics:
    Proposer stakes bounty_amount (e.g., 10 credits).
    If rejected: bounty distributed equally among valid CE finders.
    If approved: proposer gets bounty back + 20% bonus.
    Invalid reveals (hash mismatch): penalty = 5 credits.

This resolves the Gödel Machine flaw:
    Original GM: self-referential verification (agent verifies own verifier)
    OUROBOROS: external adversarial verification with cryptographic fairness
               + OOD pressure as external consistency check

The ProofMarket does NOT know about agent internals.
It only sees: proposals (bytes), commitments (hashes), reveals (CE bytes).
"""

import uuid
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from ouroboros.proof_market.commit_reveal import (
    Commitment, RoundState, make_commitment, make_null_commitment,
    verify_reveal, is_null_commitment
)
from ouroboros.proof_market.counterexample import CounterexampleResult
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import MDLCost
from ouroboros.utils.logger import get_logger


@dataclass
class MarketAgent:
    """
    An agent's economic state in the proof market.

    Separate from Phase 1 agent — tracks credits and market behavior.
    """
    agent_id: int
    credit: float = 100.0
    proposals_made: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    counterexamples_found: int = 0
    counterexamples_attempted: int = 0
    invalid_reveals: int = 0
    total_bounty_earned: float = 0.0
    total_bounty_staked: float = 0.0

    # Specialization scores (updated as market runs)
    adversary_score: float = 0.0    # Good at finding CEs
    proposer_score: float = 0.0     # Good at proposing valid modifications
    role: str = 'generalist'        # generalist | adversary | proposer

    def update_role(self) -> None:
        """Update dominant role from score history."""
        if self.adversary_score > self.proposer_score * 1.5:
            self.role = 'adversary'
        elif self.proposer_score > self.adversary_score * 1.5:
            self.role = 'proposer'
        else:
            self.role = 'generalist'


@dataclass
class ProposalRecord:
    """Complete record of a proposal and its outcome."""
    proposal_id: str
    proposer_id: int
    current_expression: ExprNode
    proposed_expression: ExprNode
    test_sequence: List[int]
    alphabet_size: int
    bounty_amount: float
    claimed_improvement_bits: float

    # Populated after adjudication
    outcome: str = 'PENDING'   # PENDING | APPROVED | REJECTED | REVOKED
    counterexample_finders: List[int] = field(default_factory=list)
    ood_passed: Optional[bool] = None
    round_state: Optional[RoundState] = None

    def __repr__(self) -> str:
        return (f"Proposal(id={self.proposal_id[:8]}, "
                f"proposer={self.proposer_id}, "
                f"outcome={self.outcome})")


class ProofMarket:
    """
    Manages proof market rounds with cryptographic commit-reveal.

    Args:
        num_agents: Total agents participating
        starting_credit: Initial credit per agent
        invalid_reveal_penalty: Credits deducted for hash mismatch
        approval_bonus_fraction: Proposer earns this fraction extra on approval
    """

    def __init__(
        self,
        num_agents: int = 8,
        starting_credit: float = 100.0,
        invalid_reveal_penalty: float = 5.0,
        approval_bonus_fraction: float = 0.20
    ):
        self.num_agents = num_agents
        self.invalid_reveal_penalty = invalid_reveal_penalty
        self.approval_bonus_fraction = approval_bonus_fraction
        self.logger = get_logger('ProofMarket')

        # Agent states
        self.agents: Dict[int, MarketAgent] = {
            i: MarketAgent(agent_id=i, credit=starting_credit)
            for i in range(num_agents)
        }

        # History
        self.proposals: List[ProposalRecord] = []
        self.current_proposal: Optional[ProposalRecord] = None
        self.current_round: Optional[RoundState] = None

        # Approved modifications — survivors of the market
        self.approved_proposals: List[ProposalRecord] = []

        self._mdl = MDLCost()

    # ─── Phase 1: PROPOSE ────────────────────────────────────────────────

    def propose(
        self,
        proposer_id: int,
        current_expr: ExprNode,
        proposed_expr: ExprNode,
        test_sequence: List[int],
        alphabet_size: int,
        bounty: float = 10.0
    ) -> str:
        """
        Agent proposes a self-modification.

        The proposal claims proposed_expr is better than current_expr
        on test_sequence under MDL.

        Args:
            proposer_id: Agent making the proposal
            current_expr: Current best expression
            proposed_expr: Proposed replacement
            test_sequence: Test data to evaluate on
            alphabet_size: Symbol alphabet size
            bounty: Credits staked (lost if rejected, earned+bonus if approved)

        Returns:
            proposal_id (str)

        Raises:
            ValueError: If proposer has insufficient credit
        """
        if self.agents[proposer_id].credit < bounty:
            raise ValueError(
                f"Agent {proposer_id} has {self.agents[proposer_id].credit:.1f} "
                f"credits but needs {bounty}"
            )
        if self.current_round is not None:
            raise RuntimeError("A round is already in progress. Resolve it first.")

        # Deduct bounty
        self.agents[proposer_id].credit -= bounty
        self.agents[proposer_id].total_bounty_staked += bounty
        self.agents[proposer_id].proposals_made += 1

        # Compute claimed improvement
        n = len(test_sequence)
        curr_preds = current_expr.predict_sequence(n, alphabet_size)
        prop_preds = proposed_expr.predict_sequence(n, alphabet_size)
        curr_cost = self._mdl.total_cost(
            current_expr.to_bytes(), curr_preds, test_sequence, alphabet_size
        )
        prop_cost = self._mdl.total_cost(
            proposed_expr.to_bytes(), prop_preds, test_sequence, alphabet_size
        )
        claimed_bits = curr_cost - prop_cost

        proposal = ProposalRecord(
            proposal_id=str(uuid.uuid4()),
            proposer_id=proposer_id,
            current_expression=current_expr,
            proposed_expression=proposed_expr,
            test_sequence=test_sequence,
            alphabet_size=alphabet_size,
            bounty_amount=bounty,
            claimed_improvement_bits=claimed_bits
        )

        round_state = RoundState(
            round_id=proposal.proposal_id[:8],
            proposer_id=proposer_id,
            proposal_bytes=proposed_expr.to_bytes(),
            proposal_description=(
                f"{current_expr.to_string()!r} → {proposed_expr.to_string()!r}"
            )
        )
        round_state.advance_to_commit()

        self.current_proposal = proposal
        self.current_round = round_state
        proposal.round_state = round_state

        self.logger.info(
            f"Proposal {proposal.proposal_id[:8]}: "
            f"Agent {proposer_id} claims {claimed_bits:.1f} bit improvement"
        )

        return proposal.proposal_id

    # ─── Phase 2: COMMIT ─────────────────────────────────────────────────

    def commit(
        self,
        agent_id: int,
        ce_result: CounterexampleResult
    ) -> None:
        """
        Agent commits to their counterexample attempt.

        Called during COMMIT phase. The agent has already run their search
        and found (or not found) a counterexample. They commit to it now
        without revealing it.

        Args:
            agent_id: Committing agent
            ce_result: Their counterexample search result
        """
        if self.current_round is None or self.current_round.phase != 'COMMIT':
            self.logger.warning(f"Agent {agent_id}: commit outside COMMIT phase")
            return
        if agent_id == self.current_proposal.proposer_id:
            return  # Proposer doesn't vote on own proposal

        # Serialize counterexample result deterministically
        ce_bytes = ce_result.to_bytes()

        commitment = make_commitment(
            agent_id=agent_id,
            counterexample=ce_bytes,
            round_id=self.current_round.round_id
        )
        self.current_round.commitments[agent_id] = commitment
        self.agents[agent_id].counterexamples_attempted += 1

    def commit_null(self, agent_id: int) -> None:
        """Agent found no counterexample — still must commit."""
        if self.current_round is None or self.current_round.phase != 'COMMIT':
            return
        if agent_id == self.current_proposal.proposer_id:
            return
        commitment = make_null_commitment(agent_id, self.current_round.round_id)
        self.current_round.commitments[agent_id] = commitment

    def close_commit_phase(self) -> None:
        """Close commit window. No more commitments accepted."""
        if self.current_round:
            self.current_round.advance_to_reveal()
            self.logger.info(
                f"Round {self.current_round.round_id}: "
                f"COMMIT closed ({len(self.current_round.commitments)} commitments)"
            )

    # ─── Phase 3: REVEAL ─────────────────────────────────────────────────

    def reveal(self, agent_id: int) -> bool:
        """
        Agent reveals their committed counterexample.

        The commitment's counterexample and salt are now public.
        Anyone can verify the hash.

        Returns True if reveal was valid (hash matched).
        """
        if self.current_round is None or self.current_round.phase != 'REVEAL':
            return False

        commitment = self.current_round.commitments.get(agent_id)
        if commitment is None:
            return False

        commitment.revealed = True
        is_valid = verify_reveal(commitment)

        if not is_valid:
            # Penalize tampered reveals
            self.agents[agent_id].credit -= self.invalid_reveal_penalty
            self.agents[agent_id].invalid_reveals += 1
            self.logger.warning(
                f"Agent {agent_id}: INVALID REVEAL — hash mismatch! "
                f"Penalty: -{self.invalid_reveal_penalty} credits"
            )

        return is_valid

    def close_reveal_phase(self) -> None:
        """Close reveal window."""
        if self.current_round:
            self.current_round.advance_to_verify()

    # ─── Phase 4: ADJUDICATE ─────────────────────────────────────────────

    def adjudicate(self) -> bool:
        """
        Resolve the current round.

        1. Collect all valid, non-null reveals
        2. Check if any are genuine counterexamples
        3. Distribute bounty or approve the proposal

        Returns: True if modification approved, False if rejected.
        """
        if self.current_round is None:
            raise RuntimeError("No active round to adjudicate")

        self.current_round.phase = 'VERIFY'
        proposal = self.current_proposal
        valid_ce_finders = []

        for agent_id, commitment in self.current_round.commitments.items():
            # Skip unrevealed or invalid commitments
            if not commitment.revealed or not commitment.valid:
                continue
            if is_null_commitment(commitment):
                continue

            # Parse the CounterexampleResult from the bytes
            import json
            try:
                data = json.loads(commitment.counterexample.decode('utf-8'))
                ce_cost = data.get('ce_cost', float('inf'))
                prop_cost = data.get('proposal_cost', float('inf'))
                is_valid_ce = data.get('is_valid', False)
            except Exception:
                continue

            if is_valid_ce and ce_cost < prop_cost:
                valid_ce_finders.append(agent_id)

        self.current_round.counterexamples_verified = len(valid_ce_finders)
        proposal.counterexample_finders = valid_ce_finders

        if valid_ce_finders:
            # REJECTED — distribute bounty
            proposal.outcome = 'REJECTED'
            self.current_round.resolve(False, f"Rejected by {len(valid_ce_finders)} CE(s)")
            self.agents[proposal.proposer_id].proposals_rejected += 1

            share = proposal.bounty_amount / len(valid_ce_finders)
            for aid in valid_ce_finders:
                self.agents[aid].credit += share
                self.agents[aid].total_bounty_earned += share
                self.agents[aid].counterexamples_found += 1
                self.current_round.bounty_distributions[aid] = share

            # Update adversary scores
            for aid in valid_ce_finders:
                self.agents[aid].adversary_score += 0.1

            self.logger.info(
                f"Round {self.current_round.round_id}: REJECTED "
                f"({len(valid_ce_finders)} CEs). "
                f"Bounty {proposal.bounty_amount:.1f} distributed."
            )
        else:
            # APPROVED (pending OOD test)
            proposal.outcome = 'APPROVED'
            self.current_round.resolve(True, "No valid counterexamples found")
            self.agents[proposal.proposer_id].proposals_accepted += 1

            bonus = proposal.bounty_amount * (1 + self.approval_bonus_fraction)
            self.agents[proposal.proposer_id].credit += bonus
            self.agents[proposal.proposer_id].total_bounty_earned += bonus * self.approval_bonus_fraction
            self.agents[proposal.proposer_id].proposer_score += 0.1

            self.approved_proposals.append(proposal)

            self.logger.info(
                f"Round {self.current_round.round_id}: APPROVED "
                f"(no valid CEs). Bonus: +{bonus:.1f} credits."
            )

        # Update roles
        for agent in self.agents.values():
            agent.update_role()

        self.proposals.append(proposal)
        result = self.current_round.modification_approved
        self.current_round = None
        self.current_proposal = None

        return result

    # ─── Utilities ───────────────────────────────────────────────────────

    def run_full_round(
        self,
        proposer_id: int,
        current_expr: ExprNode,
        proposed_expr: ExprNode,
        test_sequence: List[int],
        alphabet_size: int,
        adversarial_agents: List[int],
        ce_results: Dict[int, CounterexampleResult],
        bounty: float = 10.0
    ) -> bool:
        """
        Run a complete round in one call (for testing and demos).

        Args:
            proposer_id: The proposing agent
            current_expr: Current expression
            proposed_expr: Proposed expression
            test_sequence: Test data
            alphabet_size: Symbol alphabet
            adversarial_agents: List of agent IDs that will commit
            ce_results: Pre-computed CE results per agent
            bounty: Stake amount

        Returns: True if approved
        """
        # Propose
        self.propose(proposer_id, current_expr, proposed_expr,
                     test_sequence, alphabet_size, bounty)

        # Commit phase
        for aid in adversarial_agents:
            if aid == proposer_id:
                continue
            result = ce_results.get(aid)
            if result and result.is_valid_counterexample:
                self.commit(aid, result)
            else:
                self.commit_null(aid)

        self.close_commit_phase()

        # Reveal phase
        for aid in adversarial_agents:
            if aid == proposer_id:
                continue
            self.reveal(aid)

        self.close_reveal_phase()

        # Adjudicate
        return self.adjudicate()

    def credit_summary(self) -> Dict[int, float]:
        return {aid: ag.credit for aid, ag in self.agents.items()}

    def market_summary(self) -> str:
        lines = [
            f"ProofMarket: {len(self.proposals)} rounds",
            f"  Approved: {sum(1 for p in self.proposals if p.outcome == 'APPROVED')}",
            f"  Rejected: {sum(1 for p in self.proposals if p.outcome == 'REJECTED')}",
            "",
            "Agent credits and roles:"
        ]
        for aid, ag in sorted(self.agents.items()):
            lines.append(
                f"  Agent {aid}: {ag.credit:.1f} cr  "
                f"role={ag.role}  "
                f"props={ag.proposals_made}  "
                f"CEs={ag.counterexamples_found}  "
                f"invalid={ag.invalid_reveals}"
            )
        return '\n'.join(lines)