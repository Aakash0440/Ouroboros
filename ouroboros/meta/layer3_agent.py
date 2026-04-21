"""
Layer3Agent — Full recursive self-improvement: expression + HP + objective + strategy.

The four layers:
  Layer 0: Modify expressions (Days 1–12)
  Layer 1: Modify search hyperparameters (Day 20)
  Layer 2: Modify MDL objective (Day 23)
  Layer 3: Modify search algorithm (today)

Layer 3 is the most abstract: the agent proposes switching from one
search algorithm to another. The proposal is evaluated by the
StrategyProofMarket, which tests the new strategy on held-out environments.

The agent maintains a current_strategy. When Layer 3 fires:
  1. Enumerate alternative strategies from STRATEGY_LIBRARY
  2. Run each alternative on training data (time-boxed)
  3. If any alternative finds a better MDL cost → propose it
  4. StrategyProofMarket evaluates the proposal OOD
  5. If approved → agent permanently switches strategies

This closes the recursive loop at the algorithm level.
"""

from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.meta.search_strategy import (
    SearchStrategy, SearchConfig, SearchResult, BeamSearchStrategy,
)
from ouroboros.meta.strategy_library import (
    StrategyProposal, StrategyEvaluationResult, STRATEGY_LIBRARY,
)
from ouroboros.meta.strategy_market import StrategyProofMarket, StrategyMarketConfig
from ouroboros.environments.base import Environment


@dataclass
class Layer3AgentConfig:
    """Configuration for Layer3Agent."""
    agent_id: str = "L3_AGENT_00"
    strategy_proposal_interval: int = 8   # try to improve strategy every N rounds
    min_cost_improvement_bits: float = 3.0
    search_time_budget: float = 3.0
    search_beam_width: int = 20
    search_const_range: int = 20
    random_seed: int = 42


@dataclass
class Layer3Stats:
    """Statistics tracked by a Layer3Agent."""
    total_rounds: int = 0
    strategy_proposals: int = 0
    strategy_approvals: int = 0
    strategy_history: List[str] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)


class Layer3Agent:
    """
    An OUROBOROS agent capable of all 4 layers of self-improvement.
    
    On each round:
      - Runs its current search strategy to find the best expression
      - Every K rounds, proposes a strategy change if one improves results
    """

    def __init__(
        self,
        config: Layer3AgentConfig = None,
        initial_strategy: SearchStrategy = None,
    ):
        self.cfg = config or Layer3AgentConfig()
        self.current_strategy: SearchStrategy = (
            initial_strategy or BeamSearchStrategy()
        )
        self.stats = Layer3Stats()
        self.stats.strategy_history.append(self.current_strategy.name())

    def _make_search_config(self) -> SearchConfig:
        return SearchConfig(
            time_budget_seconds=self.cfg.search_time_budget,
            beam_width=self.cfg.search_beam_width,
            const_range=self.cfg.search_const_range,
            random_seed=self.cfg.random_seed,
        )

    def search(
        self,
        observations: List[int],
        verbose: bool = False,
    ) -> SearchResult:
        """Run the current strategy on observations."""
        config = self._make_search_config()
        result = self.current_strategy.search(observations, config)
        self.stats.cost_history.append(result.best_mdl_cost)
        if verbose:
            print(
                f"  [{self.cfg.agent_id}] {self.current_strategy.name()}: "
                f"cost={result.best_mdl_cost:.2f} bits in {result.wall_time_seconds:.2f}s"
            )
        return result

    def propose_strategy_change(
        self,
        env: Environment,
        stream_length: int = 400,
        verbose: bool = False,
    ) -> Optional[StrategyProposal]:
        """
        Enumerate alternative strategies, run each, propose the best one
        if it outperforms the current strategy.
        """
        self.stats.strategy_proposals += 1
        observations = env.generate(stream_length)
        config = self._make_search_config()

        # Evaluate current strategy
        current_result = self.current_strategy.search(observations, config)
        current_cost = current_result.best_mdl_cost
        current_time = current_result.wall_time_seconds

        if verbose:
            print(
                f"\n  [{self.cfg.agent_id}] Evaluating strategy alternatives "
                f"(current={self.current_strategy.name()}, cost={current_cost:.2f})"
            )

        # Try all alternatives
        best_alt: Optional[SearchStrategy] = None
        best_alt_cost = current_cost
        best_alt_time = current_time

        for alt_strategy in STRATEGY_LIBRARY.get_alternatives_to(
            self.current_strategy.name()
        ):
            alt_result = alt_strategy.search(observations, config)
            if verbose:
                print(
                    f"    Alt {alt_strategy.name()}: "
                    f"cost={alt_result.best_mdl_cost:.2f} in {alt_result.wall_time_seconds:.2f}s"
                )
            if alt_result.best_mdl_cost < best_alt_cost:
                best_alt_cost = alt_result.best_mdl_cost
                best_alt_time = alt_result.wall_time_seconds
                best_alt = alt_strategy

        if best_alt is None:
            if verbose:
                print(f"  [{self.cfg.agent_id}] No strategy improvement found")
            return None

        improvement = current_cost - best_alt_cost
        if improvement < self.cfg.min_cost_improvement_bits:
            return None

        proposal = StrategyProposal(
            proposing_agent=self.cfg.agent_id,
            current_strategy_name=self.current_strategy.name(),
            proposed_strategy_name=best_alt.name(),
            training_env_name=env.name,
            current_best_cost=current_cost,
            proposed_best_cost=best_alt_cost,
            current_time_seconds=current_time,
            proposed_time_seconds=best_alt_time,
        )

        if verbose:
            print(f"\n  [{self.cfg.agent_id}] STRATEGY PROPOSAL:")
            print(f"    {proposal.description()}")

        return proposal

    def apply_approved_strategy(self, new_strategy_name: str) -> None:
        """Switch to the approved strategy."""
        new_strategy = STRATEGY_LIBRARY.get(new_strategy_name)
        if new_strategy is None:
            return
        self.current_strategy = new_strategy
        self.stats.strategy_history.append(new_strategy_name)
        self.stats.strategy_approvals += 1

    def run_round(
        self,
        env: Environment,
        strategy_market: StrategyProofMarket,
        validation_envs: List[Environment],
        round_num: int,
        stream_length: int = 400,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run one round of Layer 3 self-improvement."""
        self.stats.total_rounds += 1

        # Search for best expression with current strategy
        observations = env.generate(stream_length)
        result = self.search(observations, verbose=verbose)

        round_result = {
            "round": round_num,
            "agent_id": self.cfg.agent_id,
            "current_strategy": self.current_strategy.name(),
            "best_cost": result.best_mdl_cost,
            "strategy_proposed": False,
            "strategy_approved": False,
        }

        # Every K rounds, try to improve the strategy
        if round_num % self.cfg.strategy_proposal_interval == 0:
            proposal = self.propose_strategy_change(env, stream_length, verbose)
            if proposal:
                round_result["strategy_proposed"] = True
                current_s = STRATEGY_LIBRARY.get(proposal.current_strategy_name) or \
                            BeamSearchStrategy()
                proposed_s = STRATEGY_LIBRARY.get(proposal.proposed_strategy_name) or \
                             BeamSearchStrategy()

                evaluation = strategy_market.evaluate_proposal(
                    proposal, current_s, proposed_s, env
                )

                if evaluation.approved:
                    self.apply_approved_strategy(proposal.proposed_strategy_name)
                    round_result["strategy_approved"] = True
                    if verbose:
                        print(
                            f"  [{self.cfg.agent_id}] ✅ Strategy approved! "
                            f"Now using: {self.current_strategy.name()}"
                        )
                elif verbose:
                    print(
                        f"  [{self.cfg.agent_id}] ❌ Strategy rejected: "
                        f"{evaluation.rejection_reason}"
                    )

        return round_result