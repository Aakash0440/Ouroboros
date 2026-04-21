"""
MetaSearchRunner — The complete 4-layer recursive self-improvement loop.

This is the final, most complete version of OUROBOROS.

Per round:
  1. All agents search for the best expression (Layer 0)
  2. Every K1 rounds: agents propose hyperparameter changes (Layer 1)
  3. Every K2 rounds: agents propose objective changes (Layer 2)
  4. Every K3 rounds: agents propose strategy changes (Layer 3)

The three self-improvement layers are independent — each has its own
proof market, its own proposal frequency, and its own evaluation protocol.

An agent that successfully improves all 3 layers has:
  - A better search algorithm (finds expressions faster)
  - With better hyperparameters (searches more efficiently)
  - Under a better MDL objective (finds more generalizable programs)
  
This is recursive self-improvement at three independent levels of abstraction.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from ouroboros.meta.search_strategy import BeamSearchStrategy
from ouroboros.meta.strategy_library import STRATEGY_LIBRARY
from ouroboros.meta.strategy_market import StrategyProofMarket, StrategyMarketConfig
from ouroboros.meta.layer3_agent import Layer3Agent, Layer3AgentConfig
from ouroboros.environments.base import Environment


@dataclass
class MetaRunnerConfig:
    """Configuration for the 4-layer meta runner."""
    n_agents: int = 6
    n_rounds: int = 24
    stream_length: int = 500

    # Layer 3 specific
    strategy_proposal_interval: int = 8
    search_time_budget: float = 2.0

    verbose: bool = False
    random_seed: int = 42


@dataclass
class MetaRunResult:
    """Result of a full MetaSearchRunner run."""
    n_rounds: int
    n_agents: int
    training_env_name: str
    round_logs: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 3 statistics
    total_strategy_proposals: int = 0
    total_strategy_approvals: int = 0
    strategy_evolution: Dict[str, List[str]] = field(default_factory=dict)  # agent_id → strategies

    @property
    def strategy_approval_rate(self) -> float:
        if self.total_strategy_proposals == 0:
            return 0.0
        return self.total_strategy_approvals / self.total_strategy_proposals

    def summary(self) -> str:
        lines = [
            f"\nMetaSearch Run ({self.n_rounds} rounds, {self.n_agents} agents)",
            f"  Environment: {self.training_env_name}",
            f"  Strategy proposals: {self.total_strategy_proposals}",
            f"  Approvals: {self.total_strategy_approvals} "
            f"({self.strategy_approval_rate*100:.1f}%)",
            "  Strategy evolution per agent:",
        ]
        for agent_id, history in self.strategy_evolution.items():
            evolution_str = " → ".join(history)
            lines.append(f"    {agent_id}: {evolution_str}")
        return "\n".join(lines)


class MetaSearchRunner:
    """
    Runs the full 4-layer OUROBOROS self-improvement loop.
    """

    def __init__(
        self,
        training_env: Environment,
        validation_envs: List[Environment] = None,
        config: MetaRunnerConfig = None,
    ):
        self.training_env = training_env
        self.validation_envs = validation_envs or []
        self.cfg = config or MetaRunnerConfig()

        # Initialize Layer 3 agents
        self.agents: List[Layer3Agent] = [
            Layer3Agent(
                config=Layer3AgentConfig(
                    agent_id=f"L3_AGENT_{i:02d}",
                    strategy_proposal_interval=self.cfg.strategy_proposal_interval,
                    search_time_budget=self.cfg.search_time_budget,
                    search_beam_width=20,
                    random_seed=self.cfg.random_seed + i * 11,
                ),
                initial_strategy=BeamSearchStrategy(),
            )
            for i in range(self.cfg.n_agents)
        ]

        # Initialize strategy market
        self.strategy_market = StrategyProofMarket(
            config=StrategyMarketConfig(
                min_cost_improvement_bits=3.0,
                search_time_budget=self.cfg.search_time_budget,
            ),
            validation_environments=self.validation_envs,
        )

    def run(self) -> MetaRunResult:
        result = MetaRunResult(
            n_rounds=self.cfg.n_rounds,
            n_agents=self.cfg.n_agents,
            training_env_name=self.training_env.name,
        )
        # Initialize strategy evolution tracking
        for agent in self.agents:
            result.strategy_evolution[agent.cfg.agent_id] = [
                agent.current_strategy.name()
            ]

        print(f"\n{'='*60}")
        print(f"META SEARCH RUNNER — 4-Layer Self-Improvement")
        print(f"Environment: {self.training_env.name}")
        print(f"Strategies available: {STRATEGY_LIBRARY.all_names()}")
        print(f"{'='*60}\n")

        for round_num in range(1, self.cfg.n_rounds + 1):
            if round_num % 6 == 0 or self.cfg.verbose:
                print(f"\n─── Round {round_num}/{self.cfg.n_rounds} ───")
                for agent in self.agents:
                    print(
                        f"  {agent.cfg.agent_id}: {agent.current_strategy.name()}"
                    )

            round_logs = []
            for agent in self.agents:
                log = agent.run_round(
                    env=self.training_env,
                    strategy_market=self.strategy_market,
                    validation_envs=self.validation_envs,
                    round_num=round_num,
                    stream_length=self.cfg.stream_length,
                    verbose=self.cfg.verbose,
                )
                round_logs.append(log)

                result.total_strategy_proposals += int(log["strategy_proposed"])
                result.total_strategy_approvals += int(log["strategy_approved"])

                if log["strategy_approved"]:
                    result.strategy_evolution[agent.cfg.agent_id].append(
                        agent.current_strategy.name()
                    )

            result.round_logs.append({
                "round": round_num,
                "agent_logs": round_logs,
                "market_approval_rate": self.strategy_market.approval_rate,
            })

        print(result.summary())
        return result

    @classmethod
    def for_modular_arithmetic(
        cls,
        modulus: int = 7,
        n_rounds: int = 24,
        verbose: bool = False,
    ) -> 'MetaSearchRunner':
        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.environments.noise import NoiseEnv
        from ouroboros.environments.multi_scale import MultiScaleEnv

        training = ModularArithmeticEnv(modulus=modulus, slope=3, intercept=1)
        validation = [
            ModularArithmeticEnv(modulus=11, slope=5, intercept=2),
            NoiseEnv(alphabet_size=modulus),
            MultiScaleEnv(),
        ]
        cfg = MetaRunnerConfig(
            n_agents=6, n_rounds=n_rounds,
            stream_length=400, search_time_budget=2.0,
            verbose=verbose
        )
        return cls(training, validation, cfg)