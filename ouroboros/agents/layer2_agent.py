"""
Layer2Agent — Recursive Self-Improvement at the Objective Level

Extends HyperparameterAgent (Layer 1) with Layer 2 capability:
  Layer 1: agent modifies beam_width, mcmc_iterations (HOW to search)
  Layer 2: agent modifies lambda_prog, lambda_const (WHAT to optimize)

The three layers of recursive self-improvement in OUROBOROS:
  Layer 0 (Days 1–12): Agent modifies its symbolic expressions
  Layer 1 (Day 20):    Agent modifies its search hyperparameters
  Layer 2 (Day 23):    Agent modifies its MDL objective function
  Layer 3 (Future):    Agent modifies its search algorithm itself

At Layer 2, the agent reasons about whether a different MDL cost function
would lead to discovering better patterns. For example:
  - Lower lambda_prog → favors LONGER expressions (better fit, more complex)
  - Higher lambda_prog → favors SHORTER expressions (simpler, generalizes better)
  - Higher axiom_r2_threshold → promotes only very accurate continuous axioms
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ouroboros.agents.mdl_objective import (
    MDLObjective, ObjectiveProposal, DEFAULT_OBJECTIVE,
)
from ouroboros.agents.objective_market import ObjectiveProofMarket
from ouroboros.environments.base import ObservationEnvironment


@dataclass
class Layer2AgentConfig:
    """Configuration for Layer2Agent."""
    agent_id: str = "LAYER2_AGENT_00"
    
    # Layer 1 hyperparameters (inherited)
    beam_width: int = 25
    mcmc_iterations: int = 150
    const_range: int = 30
    max_depth: int = 4
    max_lag: int = 3
    
    # Layer 2 objective tuning
    objective_proposal_interval: int = 5    # try to improve objective every N rounds
    objective_delta_lambda: float = 0.5     # step size for lambda adjustments
    objective_min_improvement_bits: float = 5.0
    
    # Shared
    random_seed: int = 42


@dataclass
class Layer2Stats:
    """Statistics tracked by a Layer2Agent across its lifetime."""
    total_rounds: int = 0
    objective_proposals_made: int = 0
    objective_proposals_approved: int = 0
    objective_evolution: List[MDLObjective] = field(default_factory=list)
    mdl_cost_history: List[float] = field(default_factory=list)
    lambda_prog_history: List[float] = field(default_factory=list)
    lambda_const_history: List[float] = field(default_factory=list)


class Layer2Agent:
    """
    An OUROBOROS agent capable of Layer 2 self-improvement.
    
    In addition to searching for better expressions (Layer 0) and
    tuning its search hyperparameters (Layer 1), this agent can
    also propose changes to its MDL objective function.
    
    The key insight: the MDL objective is an inductive bias.
    A better inductive bias leads to faster convergence, better
    generalization, and more accurate axiom promotion.
    
    By allowing agents to update their inductive bias via the
    ObjectiveProofMarket, we close the loop: the agent learns
    not just mathematical patterns but also learns HOW to learn them.
    """

    def __init__(
        self,
        config: Layer2AgentConfig = None,
        initial_objective: MDLObjective = None,
    ):
        self.cfg = config or Layer2AgentConfig()
        self.current_objective = copy.deepcopy(initial_objective or DEFAULT_OBJECTIVE)
        self._rng = random.Random(self.cfg.random_seed)
        self.stats = Layer2Stats()

        # Keep the objective history
        self.stats.objective_evolution.append(copy.deepcopy(self.current_objective))

    def _generate_objective_candidates(self) -> List[MDLObjective]:
        """
        Generate candidate MDL objectives by perturbing the current one.
        
        Perturbation strategy:
        - Vary lambda_prog by ±delta (lower → prefer complex programs)
        - Vary lambda_const by ±delta (lower → more constants allowed)
        - Vary axiom_r2_threshold by ±0.05 (raise → stricter axiom promotion)
        - Vary complexity_penalty_exp by ±0.1 (raise → penalize complexity superlinearly)
        """
        delta = self.cfg.objective_delta_lambda
        candidates = []

        for lp_delta in [-delta, -delta/2, 0, delta/2, delta]:
            for lc_delta in [-delta, 0, delta]:
                new_obj = MDLObjective(
                    lambda_prog=self.current_objective.lambda_prog + lp_delta,
                    lambda_const=self.current_objective.lambda_const + lc_delta,
                    gaussian_min_sigma=self.current_objective.gaussian_min_sigma,
                    axiom_consensus_fraction=self.current_objective.axiom_consensus_fraction,
                    axiom_r2_threshold=self.current_objective.axiom_r2_threshold,
                    complexity_penalty_exp=self.current_objective.complexity_penalty_exp,
                )
                clamped = new_obj.clamp()
                if clamped.is_valid():
                    candidates.append(clamped)

        # Also try a few random perturbations
        for _ in range(5):
            rand_obj = MDLObjective(
                lambda_prog=self.current_objective.lambda_prog + self._rng.gauss(0, delta),
                lambda_const=self.current_objective.lambda_const + self._rng.gauss(0, delta),
                gaussian_min_sigma=self.current_objective.gaussian_min_sigma,
                axiom_consensus_fraction=max(0.1, min(1.0,
                    self.current_objective.axiom_consensus_fraction + self._rng.gauss(0, 0.05))),
                axiom_r2_threshold=max(0.5, min(1.0,
                    self.current_objective.axiom_r2_threshold + self._rng.gauss(0, 0.02))),
                complexity_penalty_exp=max(0.5, min(3.0,
                    self.current_objective.complexity_penalty_exp + self._rng.gauss(0, 0.1))),
            ).clamp()
            if rand_obj.is_valid():
                candidates.append(rand_obj)

        return candidates

    def _estimate_cost_under_objective(
        self,
        env: Environment,
        objective: MDLObjective,
        stream_length: int = 300,
    ) -> float:
        """
        Estimate the MDL cost achievable under the given objective
        by running a quick beam search and computing the cost.
        
        This is used to compare objective candidates.
        """
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine

        observations = env.generate(stream_length)

        # Run quick search
        config = BeamConfig(
            beam_width=self.cfg.beam_width,
            const_range=self.cfg.const_range,
            max_depth=self.cfg.max_depth,
            max_lag=self.cfg.max_lag,
        )
        synthesizer = BeamSearchSynthesizer(config)
        best_expr = synthesizer.search(observations)

        if best_expr is None:
            return float('inf')

        # Score under the given objective
        mdl_engine = MDLEngine()
        predictions = [best_expr.evaluate(t, observations[:t]) for t in range(len(observations))]

        # Program cost under this objective
        prog_bits = objective.compute_program_bits(
            best_expr.node_count(),
            best_expr.constant_count()
        )

        # Data cost (standard Shannon bits — objective doesn't change this)
        from collections import Counter
        import math
        errors = [p - a for p, a in zip(predictions, observations)]
        n = len(errors)
        counts = Counter(errors)
        data_bits = -sum(
            (c / n) * math.log2(c / n) for c in counts.values() if c > 0
        ) * n

        return prog_bits + data_bits

    def propose_objective_modification(
        self,
        env: Environment,
        stream_length: int = 300,
        verbose: bool = False,
    ) -> Optional[ObjectiveProposal]:
        """
        Generate an objective modification proposal if a better objective is found.
        
        Returns None if no improvement found.
        """
        self.stats.objective_proposals_made += 1

        # Evaluate current objective
        current_cost = self._estimate_cost_under_objective(
            env, self.current_objective, stream_length
        )
        self.stats.mdl_cost_history.append(current_cost)
        self.stats.lambda_prog_history.append(self.current_objective.lambda_prog)
        self.stats.lambda_const_history.append(self.current_objective.lambda_const)

        # Try candidate objectives
        candidates = self._generate_objective_candidates()
        best_candidate = None
        best_cost = current_cost

        for candidate in candidates:
            candidate_cost = self._estimate_cost_under_objective(
                env, candidate, stream_length // 2  # use shorter stream for speed
            )
            if candidate_cost < best_cost:
                best_cost = candidate_cost
                best_candidate = candidate

        if best_candidate is None:
            if verbose:
                print(f"  [{self.cfg.agent_id}] No objective improvement found (current={current_cost:.2f})")
            return None

        improvement = current_cost - best_cost
        if improvement < self.cfg.objective_min_improvement_bits:
            return None

        proposal = ObjectiveProposal(
            proposing_agent=self.cfg.agent_id,
            current_objective=self.current_objective,
            proposed_objective=best_candidate,
            training_env_name=env.name,
            current_total_bits=current_cost,
            proposed_total_bits=best_cost,
        )

        if verbose:
            print(f"\n  [{self.cfg.agent_id}] OBJECTIVE PROPOSAL:")
            print(f"    {proposal.description()}")

        return proposal

    def apply_approved_objective(self, new_objective: MDLObjective) -> None:
        """Apply an approved objective modification."""
        self.current_objective = copy.deepcopy(new_objective)
        self.stats.objective_evolution.append(copy.deepcopy(new_objective))
        self.stats.objective_proposals_approved += 1

    def run_round(
        self,
        env: Environment,
        objective_market: ObjectiveProofMarket,
        validation_envs: List[Environment],
        round_num: int,
        stream_length: int = 300,
        verbose: bool = False,
    ) -> dict:
        """
        Run one full round of Layer 2 self-improvement.
        
        Returns a dict with round results.
        """
        self.stats.total_rounds += 1
        round_result = {
            "round": round_num,
            "agent_id": self.cfg.agent_id,
            "objective_proposed": False,
            "objective_approved": False,
            "lambda_prog": self.current_objective.lambda_prog,
            "lambda_const": self.current_objective.lambda_const,
            "current_cost": 0.0,
        }

        # Every N rounds, try to improve the objective
        should_propose = (round_num % self.cfg.objective_proposal_interval == 0)

        if should_propose:
            proposal = self.propose_objective_modification(env, stream_length, verbose)

            if proposal:
                round_result["objective_proposed"] = True
                evaluation = objective_market.evaluate_proposal(proposal, env)

                if evaluation.approved:
                    self.apply_approved_objective(proposal.proposed_objective)
                    round_result["objective_approved"] = True
                    if verbose:
                        print(f"  [{self.cfg.agent_id}] ✅ Objective approved!")
                        print(f"    New: {self.current_objective.description()}")
                else:
                    if verbose:
                        print(f"  [{self.cfg.agent_id}] ❌ Rejected: {evaluation.rejection_reason}")

        round_result["lambda_prog"] = self.current_objective.lambda_prog
        round_result["lambda_const"] = self.current_objective.lambda_const

        return round_result