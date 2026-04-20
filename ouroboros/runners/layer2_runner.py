"""
Layer2Runner — Runs the full 3-layer self-improvement loop.

Layer 0: Expression modification (Days 1–12)
Layer 1: Hyperparameter modification (Day 20)
Layer 2: Objective modification (Day 23)

In each round:
  1. All agents search for better expressions (Layer 0)
  2. Every K rounds, agents propose hyperparameter changes (Layer 1)
  3. Every M rounds, agents propose objective changes (Layer 2)

This is the most complete version of OUROBOROS's recursive self-improvement.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.agents.mdl_objective import MDLObjective, DEFAULT_OBJECTIVE
from ouroboros.agents.objective_market import (
    ObjectiveProofMarket, ObjectiveMarketConfig,
)
from ouroboros.agents.layer2_agent import Layer2Agent, Layer2AgentConfig
from ouroboros.environmentss.base import Environment


@dataclass
class Layer2RunnerConfig:
    """Configuration for the full 3-layer runner."""
    n_agents: int = 6
    n_rounds: int = 20
    stream_length: int = 500
    
    # Layer 2 specific
    objective_proposal_interval: int = 5
    n_validation_envs: int = 3
    
    verbose: bool = False
    random_seed: int = 42


@dataclass
class Layer2RunResult:
    """Result of a full Layer2Runner run."""
    n_rounds: int
    n_agents: int
    training_env_name: str
    
    # Per-round data
    round_logs: List[Dict[str, Any]] = field(default_factory=list)
    
    # Objective evolution
    initial_objective: MDLObjective = field(default_factory=MDLObjective)
    final_objective: MDLObjective = field(default_factory=MDLObjective)
    
    # Summary statistics
    total_objective_proposals: int = 0
    total_objective_approvals: int = 0
    lambda_prog_start: float = 2.0
    lambda_prog_end: float = 2.0
    lambda_const_start: float = 8.0
    lambda_const_end: float = 8.0

    @property
    def objective_approval_rate(self) -> float:
        if self.total_objective_proposals == 0:
            return 0.0
        return self.total_objective_approvals / self.total_objective_proposals

    def summary(self) -> str:
        return (
            f"Layer 2 Run Summary ({self.n_rounds} rounds, {self.n_agents} agents)\n"
            f"  Environment: {self.training_env_name}\n"
            f"  Objective proposals: {self.total_objective_proposals}\n"
            f"  Approvals: {self.total_objective_approvals} "
            f"({self.objective_approval_rate*100:.1f}%)\n"
            f"  λ_prog: {self.lambda_prog_start:.3f} → {self.lambda_prog_end:.3f}\n"
            f"  λ_const: {self.lambda_const_start:.3f} → {self.lambda_const_end:.3f}\n"
            f"  Final objective: {self.final_objective.description()}"
        )


class Layer2Runner:
    """
    Runs the 3-layer OUROBOROS self-improvement loop.
    
    This is the most complete implementation of OUROBOROS's
    recursive self-improvement architecture.
    """

    def __init__(
        self,
        training_env: Environment,
        validation_envs: List[Environment] = None,
        config: Layer2RunnerConfig = None,
    ):
        self.training_env = training_env
        self.validation_envs = validation_envs or []
        self.cfg = config or Layer2RunnerConfig()

        # Initialize agents
        self.agents: List[Layer2Agent] = [
            Layer2Agent(
                config=Layer2AgentConfig(
                    agent_id=f"L2_AGENT_{i:02d}",
                    beam_width=25,
                    objective_proposal_interval=self.cfg.objective_proposal_interval,
                    random_seed=self.cfg.random_seed + i * 13,
                ),
                initial_objective=copy.deepcopy(DEFAULT_OBJECTIVE),
            )
            for i in range(self.cfg.n_agents)
        ]

        # Initialize objective market
        market_config = ObjectiveMarketConfig(
            n_adversaries=4,
            min_improvement_bits=5.0,
            validation_stream_length=self.cfg.stream_length,
        )
        self.objective_market = ObjectiveProofMarket(
            config=market_config,
            validation_environments=self.validation_envs,
        )

    def run(self) -> Layer2RunResult:
        """Run the full Layer 2 experiment."""
        initial_obj = copy.deepcopy(DEFAULT_OBJECTIVE)
        result = Layer2RunResult(
            n_rounds=self.cfg.n_rounds,
            n_agents=self.cfg.n_agents,
            training_env_name=self.training_env.name,
            initial_objective=initial_obj,
            lambda_prog_start=initial_obj.lambda_prog,
            lambda_const_start=initial_obj.lambda_const,
        )

        print(f"\n{'='*60}")
        print(f"LAYER 2 SELF-IMPROVEMENT EXPERIMENT")
        print(f"Environment: {self.training_env.name}")
        print(f"Agents: {self.cfg.n_agents}, Rounds: {self.cfg.n_rounds}")
        print(f"Objective proposals every {self.cfg.objective_proposal_interval} rounds")
        print(f"{'='*60}\n")

        for round_num in range(1, self.cfg.n_rounds + 1):
            if self.cfg.verbose or round_num % 5 == 0:
                print(f"\n─── Round {round_num}/{self.cfg.n_rounds} ───")

            round_logs = []
            for agent in self.agents:
                agent_log = agent.run_round(
                    env=self.training_env,
                    objective_market=self.objective_market,
                    validation_envs=self.validation_envs,
                    round_num=round_num,
                    stream_length=self.cfg.stream_length,
                    verbose=self.cfg.verbose,
                )
                round_logs.append(agent_log)

                result.total_objective_proposals += int(agent_log["objective_proposed"])
                result.total_objective_approvals += int(agent_log["objective_approved"])

            result.round_logs.append({
                "round": round_num,
                "agent_logs": round_logs,
                "approval_rate": self.objective_market.approval_rate,
            })

        # Final state
        # Use the best agent's objective as the result
        best_agent = max(
            self.agents,
            key=lambda a: a.stats.objective_proposals_approved,
        )
        result.final_objective = copy.deepcopy(best_agent.current_objective)
        result.lambda_prog_end = best_agent.current_objective.lambda_prog
        result.lambda_const_end = best_agent.current_objective.lambda_const

        print(f"\n{result.summary()}")
        return result

    @classmethod
    def for_modular_arithmetic(
        cls,
        modulus: int = 7,
        slope: int = 3,
        intercept: int = 1,
        n_rounds: int = 20,
        verbose: bool = False,
    ) -> 'Layer2Runner':
        """Factory method for standard ModularArithmetic experiment."""
        from ouroboros.environmentss.modular import ModularArithmeticEnv
        from ouroboros.environmentss.noise import NoiseEnv
        from ouroboros.environmentss.binary_repeat import BinaryRepeatEnv

        training = ModularArithmeticEnv(modulus=modulus, slope=slope, intercept=intercept)
        validation = [
            ModularArithmeticEnv(modulus=11, slope=5, intercept=2),
            NoiseEnv(alphabet_size=modulus),
            BinaryRepeatEnv(),
        ]
        config = Layer2RunnerConfig(
            n_agents=6,
            n_rounds=n_rounds,
            stream_length=500,
            verbose=verbose,
        )
        return cls(training, validation, config)

    @classmethod
    def for_continuous_sine(
        cls,
        frequency: float = 1.0/7.0,
        n_rounds: int = 15,
        verbose: bool = False,
    ) -> 'Layer2Runner':
        """Factory method for SineEnv experiment."""
        from ouroboros.continuous.environments import (
            SineEnv, ContinuousNoiseEnv, PolynomialEnv
        )

        training = SineEnv(frequency=frequency, noise_sigma=0.0)
        validation = [
            SineEnv(frequency=1/11.0, noise_sigma=0.02),
            PolynomialEnv(coefficients=[1.0, -2.0, 0.5]),
            ContinuousNoiseEnv(sigma=1.0),
        ]
        config = Layer2RunnerConfig(
            n_agents=4,
            n_rounds=n_rounds,
            stream_length=200,
            verbose=verbose,
        )
        return cls(training, validation, config)