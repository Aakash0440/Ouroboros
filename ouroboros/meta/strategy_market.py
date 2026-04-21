"""
StrategyProofMarket — Layer 3 adversarial evaluation of search strategies.

Protocol:
1. An agent (Layer3Agent) proposes switching from strategy A to strategy B.
   Evidence: B finds lower MDL cost than A on training environment.

2. Adversaries try to find a validation environment where B performs WORSE:
   - They run both A and B on the validation environment
   - If B's MDL cost > A's MDL cost + tolerance → reject
   - Also check: is B so slow it's not worth the extra bits saved?

3. Time-aware evaluation: a strategy that's 10× slower must save
   proportionately more bits to be approved. The approval criterion is:
     bits_saved / extra_seconds > efficiency_threshold

4. If no adversary can find a validation environment where B fails → approve.
   The proposing agent permanently switches to strategy B.
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ouroboros.meta.search_strategy import (
    SearchStrategy, SearchConfig, SearchResult,
)
from ouroboros.meta.strategy_library import (
    StrategyProposal, StrategyEvaluationResult, STRATEGY_LIBRARY,
)
from ouroboros.environments.base import Environment


@dataclass
class StrategyMarketConfig:
    """Configuration for the strategy proof market."""
    n_adversaries: int = 3
    min_cost_improvement_bits: float = 3.0
    efficiency_threshold_bits_per_second: float = 2.0  # must save 2 bits/sec overhead
    validation_stream_length: int = 400
    search_time_budget: float = 3.0      # seconds per evaluation
    tolerance_bits: float = 5.0          # validation can be 5 bits worse — still OK
    random_seed: int = 42


class StrategyProofMarket:
    """
    Adversarially evaluates search strategy proposals.

    The adversarial question: "Is this strategy better in general,
    or just better on this training environment?"
    """

    def __init__(
        self,
        config: StrategyMarketConfig = None,
        validation_environments: List[Environment] = None,
    ):
        self.cfg = config or StrategyMarketConfig()
        self.validation_envs = validation_environments or []
        self._rng = random.Random(self.cfg.random_seed)
        self._approved_strategies: List[str] = []
        self._evaluation_history: List[StrategyEvaluationResult] = []

    def evaluate_proposal(
        self,
        proposal: StrategyProposal,
        current_strategy: SearchStrategy,
        proposed_strategy: SearchStrategy,
        training_env: Environment,
    ) -> StrategyEvaluationResult:
        """
        Evaluate a strategy proposal.
        
        Returns approval/rejection with full justification.
        """
        # ── Check 1: Minimum improvement ──────────────────────────────────
        if proposal.cost_improvement < self.cfg.min_cost_improvement_bits:
            result = StrategyEvaluationResult(
                proposal=proposal,
                approved=False,
                validation_env_name="threshold_check",
                validation_current_cost=proposal.current_best_cost,
                validation_proposed_cost=proposal.proposed_best_cost,
                rejection_reason=(
                    f"Cost improvement {proposal.cost_improvement:.2f} bits "
                    f"< threshold {self.cfg.min_cost_improvement_bits:.2f} bits"
                ),
            )
            self._evaluation_history.append(result)
            return result

        # ── Check 2: Time efficiency ───────────────────────────────────────
        if proposal.time_overhead_fraction > 1.0:  # more than 2× slower
            efficiency = proposal.cost_efficiency
            if efficiency < self.cfg.efficiency_threshold_bits_per_second:
                result = StrategyEvaluationResult(
                    proposal=proposal,
                    approved=False,
                    validation_env_name="efficiency_check",
                    validation_current_cost=proposal.current_best_cost,
                    validation_proposed_cost=proposal.proposed_best_cost,
                    rejection_reason=(
                        f"Strategy is {proposal.time_overhead_fraction*100:.0f}% slower "
                        f"but only saves {efficiency:.2f} bits/sec "
                        f"(threshold: {self.cfg.efficiency_threshold_bits_per_second:.2f})"
                    ),
                )
                self._evaluation_history.append(result)
                return result

        # ── Check 3: OOD validation ───────────────────────────────────────
        if self.validation_envs:
            val_env = self._rng.choice(self.validation_envs)
            val_obs = val_env.generate(self.cfg.validation_stream_length)

            search_config = SearchConfig(
                time_budget_seconds=self.cfg.search_time_budget,
                beam_width=20,
                const_range=20,
                random_seed=self.cfg.random_seed,
            )

            current_result = current_strategy.search(val_obs, search_config)
            proposed_result = proposed_strategy.search(val_obs, search_config)

            current_cost = current_result.best_mdl_cost
            proposed_cost = proposed_result.best_mdl_cost

            # Adversarial check: does proposed perform worse OOD?
            if proposed_cost > current_cost + self.cfg.tolerance_bits:
                result = StrategyEvaluationResult(
                    proposal=proposal,
                    approved=False,
                    validation_env_name=val_env.name,
                    validation_current_cost=current_cost,
                    validation_proposed_cost=proposed_cost,
                    rejection_reason=(
                        f"Proposed strategy performs worse on {val_env.name}: "
                        f"+{proposed_cost - current_cost:.2f} bits"
                    ),
                )
                self._rejected_proposal_count = getattr(self, '_rejected_proposal_count', 0) + 1
                self._evaluation_history.append(result)
                return result

            val_env_name = val_env.name
            val_current = current_cost
            val_proposed = proposed_cost
        else:
            val_env_name = "no_validation"
            val_current = proposal.current_best_cost
            val_proposed = proposal.proposed_best_cost

        # ── Approved! ─────────────────────────────────────────────────────
        result = StrategyEvaluationResult(
            proposal=proposal,
            approved=True,
            validation_env_name=val_env_name,
            validation_current_cost=val_current,
            validation_proposed_cost=val_proposed,
        )
        self._approved_strategies.append(proposal.proposed_strategy_name)
        self._evaluation_history.append(result)
        return result

    @property
    def approval_rate(self) -> float:
        n = len(self._evaluation_history)
        if n == 0:
            return 0.0
        return sum(1 for r in self._evaluation_history if r.approved) / n

    @property
    def approved_strategies(self) -> List[str]:
        return list(self._approved_strategies)