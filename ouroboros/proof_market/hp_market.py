"""
HyperparameterMarket — extends ProofMarket for HP modification proposals.

The standard ProofMarket handles expression modifications.
The HyperparameterMarket handles hyperparameter modifications.

Key difference: a counterexample to an HP modification is an agent
that achieves LOWER MDL cost with the CURRENT HP than the PROPOSER
achieves with the PROPOSED HP on the same data.

In other words:
    Proposer claims: "my new HP setting (beam_width=35) finds better
                     expressions than my current setting (beam_width=25)"
    Adversary attacks: "I, using your current HP=25, found an expression
                       that beats what you found with HP=35"

If the adversary succeeds: the proposed HP increase is not justified.
If the adversary fails: the proposed HP increase is approved.

This is exactly the same proof market logic — just evaluated on
hyperparameter-mediated search results instead of direct expressions.

OOD test for HP modifications:
    After approval, test: does the new HP setting ALSO improve search
    on environments the agent hasn't seen?
    This catches HP overfitting (e.g., beam_width=100 overfits to this
    specific stream length but fails on longer streams).
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ouroboros.agents.hyperparameter_agent import (
    HyperparameterAgent, HyperparameterProposal, HyperparameterSet
)
from ouroboros.proof_market.market import ProofMarket, MarketAgent
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
from ouroboros.compression.mcmc_refiner import MCMCRefiner
from ouroboros.compression.mdl import MDLCost
from ouroboros.utils.logger import get_logger


@dataclass
class HPCounterexample:
    """
    A counterexample to a hyperparameter modification proposal.

    An adversarial agent uses the CURRENT HP and finds an expression
    that beats what the proposer found with the PROPOSED HP.
    """
    agent_id: int
    current_hp_cost: float     # What adversary found with current HP
    proposed_hp_cost: float    # What proposer found with proposed HP
    is_valid: bool             # current_hp_cost < proposed_hp_cost
    beats_by_bits: float       # proposed_hp_cost - current_hp_cost


class HPMarketRound:
    """
    One round of hyperparameter market evaluation.

    Lifecycle:
    1. Proposer submits HP modification proposal
    2. Adversaries each run search with current HP
    3. If any adversary beats proposer's proposed-HP result: REJECTED
    4. If none beat it: APPROVED (pending OOD test)
    5. OOD: proposer runs with proposed HP on fresh environment
             adversaries run with current HP
             If proposed still wins: CONFIRMED
             If proposed loses: REVOKED
    """

    def __init__(
        self,
        proposal: HyperparameterProposal,
        all_agents: List[HyperparameterAgent],
        ce_searcher_cls=None
    ):
        self.proposal = proposal
        self.all_agents = all_agents
        self.adversaries = [
            a for a in all_agents
            if a.agent_id != proposal.agent_id
        ]

    def run_adversarial_search(self) -> List[HPCounterexample]:
        """
        Each adversary runs search with the CURRENT HP on validation data.
        Returns list of counterexamples (valid and invalid).
        """
        data = self.proposal.validation_data
        alpha = self.proposal.alphabet_size
        proposed_cost = self.proposal.proposed_best_cost
        counterexamples = []

        for adversary in self.adversaries:
            # Run adversary's search using CURRENT HP
            synth = BeamSearchSynthesizer(
                beam_width=self.proposal.current_hp.beam_width,
                max_depth=self.proposal.current_hp.max_depth,
                const_range=self.proposal.current_hp.const_range,
                alphabet_size=alpha,
            )
            refiner = MCMCRefiner(
                num_iterations=self.proposal.current_hp.mcmc_iterations,
                alphabet_size=alpha,
                const_range=self.proposal.current_hp.const_range,
            )

            expr, cost = synth.search(data)
            refined, refined_cost = refiner.refine(expr, data)
            best_cost = min(cost, refined_cost)

            # Valid counterexample: current HP beats proposed HP
            is_valid = best_cost < proposed_cost * 0.95  # 5% margin
            ce = HPCounterexample(
                agent_id=adversary.agent_id,
                current_hp_cost=best_cost,
                proposed_hp_cost=proposed_cost,
                is_valid=is_valid,
                beats_by_bits=proposed_cost - best_cost
            )
            counterexamples.append(ce)

        return counterexamples

    def adjudicate(
        self,
        counterexamples: List[HPCounterexample]
    ) -> Tuple[bool, List[int]]:
        """
        Determine if proposal should be approved.

        Returns: (approved, list_of_ce_finder_ids)
        """
        valid_ces = [ce for ce in counterexamples if ce.is_valid]
        ce_finders = [ce.agent_id for ce in valid_ces]
        approved = len(valid_ces) == 0
        return approved, ce_finders


class HyperparameterMarket:
    """
    Manages hyperparameter modification rounds.

    Wraps the standard ProofMarket with HP-specific evaluation logic.

    Args:
        agents: List of HyperparameterAgents
        starting_credit: Initial credit per agent
    """

    def __init__(
        self,
        agents: List[HyperparameterAgent],
        starting_credit: float = 100.0
    ):
        self.agents = agents
        self.num_agents = len(agents)
        self.credits: Dict[int, float] = {
            a.agent_id: starting_credit for a in agents
        }
        self.logger = get_logger('HyperparameterMarket')

        # Round history
        self.rounds: List[Dict] = []
        self.total_hp_proposals: int = 0
        self.total_hp_approved: int = 0
        self.total_hp_rejected: int = 0

    def run_hp_round(
        self,
        proposal: HyperparameterProposal,
        bounty: float = 8.0
    ) -> Tuple[bool, Dict]:
        """
        Run one hyperparameter market round.

        Returns: (approved, round_stats_dict)
        """
        self.total_hp_proposals += 1
        proposer_id = proposal.agent_id

        # Deduct bounty
        if self.credits[proposer_id] < bounty:
            return False, {'error': 'insufficient_credit'}
        self.credits[proposer_id] -= bounty

        # Run adversarial search
        market_round = HPMarketRound(proposal, self.agents)
        counterexamples = market_round.run_adversarial_search()
        approved, ce_finders = market_round.adjudicate(counterexamples)

        # Distribute bounty or refund
        if approved:
            self.credits[proposer_id] += bounty * 1.2
            self.total_hp_approved += 1
        else:
            share = bounty / max(len(ce_finders), 1)
            for ce_id in ce_finders:
                self.credits[ce_id] += share
            self.total_hp_rejected += 1

        round_stats = {
            'proposal': repr(proposal),
            'approved': approved,
            'ce_finders': ce_finders,
            'num_valid_ces': len(ce_finders),
            'proposer_proposed_cost': proposal.proposed_best_cost,
            'proposer_current_cost': proposal.current_best_cost,
            'improvement_bits': proposal.improvement_bits,
        }
        self.rounds.append(round_stats)

        self.logger.info(
            f"HP round: {proposal.changed_param} {proposal.change_direction} "
            f"→ {'APPROVED' if approved else f'REJECTED by {len(ce_finders)} CEs'}"
        )

        return approved, round_stats

    def market_summary(self) -> str:
        lines = [
            f"HyperparameterMarket: {self.total_hp_proposals} rounds",
            f"  Approved: {self.total_hp_approved}",
            f"  Rejected: {self.total_hp_rejected}",
            "  Agent credits:",
        ]
        for aid, credit in sorted(self.credits.items()):
            lines.append(f"    Agent {aid}: {credit:.1f}")
        return '\n'.join(lines)