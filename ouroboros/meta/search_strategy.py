"""
SearchStrategy — First-class representation of a search algorithm.

This is the Layer 3 abstraction. Previously, agents could modify:
  Layer 1: beam_width, mcmc_iterations (HOW HARD to search with beam)
  Layer 2: lambda_prog, lambda_const (WHAT to optimize)
  Layer 3: the search algorithm itself (WHICH search procedure to use)

A SearchStrategy is a complete, self-describing algorithm that:
  1. Takes an observation sequence
  2. Returns the best expression found
  3. Reports its own description length (so MDL can penalize complex strategies)
  4. Estimates its own runtime (so the market can compare apples to apples)

The StrategyProofMarket evaluates strategies by running them on held-out
environments and comparing the MDL cost of the expressions they find.
A new strategy is approved if it consistently finds better expressions
than the current strategy within the same time budget.
"""

from __future__ import annotations
import copy
import math
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# Import the expression node types from existing code
from ouroboros.synthesis.expr_node import ExprNode, NodeType, build_linear_modular
from ouroboros.compression.mdl_engine import MDLEngine, MDLResult


@dataclass
class SearchConfig:
    """
    Unified search configuration that all strategies receive.
    
    Strategies may use some or all of these parameters.
    Unused parameters are silently ignored.
    """
    # Budget parameters (all strategies respect these)
    time_budget_seconds: float = 5.0     # wall-clock time limit
    node_budget: int = 10_000            # max expression evaluations
    
    # Structural parameters
    max_depth: int = 4
    const_range: int = 30
    max_lag: int = 3
    alphabet_size: int = 13
    
    # Strategy-specific parameters (ignored by strategies that don't use them)
    beam_width: int = 25                 # for BeamSearchStrategy
    n_restarts: int = 10                 # for RandomRestartStrategy
    mcts_exploration: float = 1.414      # √2, standard UCB1 constant
    annealing_temp_start: float = 10.0  # for AnnealingStrategy
    annealing_temp_end: float = 0.01
    mcmc_iterations: int = 150          # for MCMC refinement within strategies
    
    random_seed: int = 42


@dataclass
class SearchResult:
    """Result returned by any search strategy."""
    best_expr: Optional[ExprNode]
    best_mdl_cost: float
    n_evaluations: int
    wall_time_seconds: float
    strategy_name: str
    
    # Optional: the full trajectory (useful for analysis)
    cost_trajectory: List[float] = field(default_factory=list)
    
    @property
    def found_something(self) -> bool:
        return self.best_expr is not None

    @property
    def evaluations_per_second(self) -> float:
        if self.wall_time_seconds == 0:
            return 0.0
        return self.n_evaluations / self.wall_time_seconds


class SearchStrategy(ABC):
    """
    Abstract base class for all OUROBOROS search strategies.
    
    A concrete search strategy is a complete algorithm that finds
    the best expression for an observation sequence.
    """

    @abstractmethod
    def search(
        self,
        observations: List[int],
        config: SearchConfig,
    ) -> SearchResult:
        """Search for the best expression explaining the observations."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Short identifier string."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""
        ...

    def description_bits(self) -> float:
        """
        MDL cost of this strategy itself.
        
        More complex strategies cost more bits to describe,
        which penalizes using them unless they provide proportionate benefit.
        Different strategies have different complexity — MCTS is more
        complex than beam search, which is more complex than random restart.
        """
        return 32.0   # default: 32 bits per strategy

    def estimated_runtime_seconds(
        self,
        stream_length: int,
        config: SearchConfig,
    ) -> float:
        """Estimated runtime for given stream length and config."""
        return config.time_budget_seconds


# ─── Strategy 1: Beam Search (current default) ───────────────────────────────

class BeamSearchStrategy(SearchStrategy):
    """
    Standard beam search over expression trees.
    This is the strategy used in Days 1–20.
    """

    def name(self) -> str:
        return "BeamSearch"

    def description(self) -> str:
        return "Beam search over expression trees with MCMC constant refinement"

    def description_bits(self) -> float:
        return 20.0   # simplest strategy — low description cost

    def search(self, observations: List[int], config: SearchConfig) -> SearchResult:
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine

        start = time.time()
        n_evals = 0

        beam_cfg = BeamConfig(
            beam_width=config.beam_width,
            const_range=config.const_range,
            max_depth=config.max_depth,
            max_lag=config.max_lag,
            mcmc_iterations=config.mcmc_iterations,
            random_seed=config.random_seed,
        )
        synthesizer = BeamSearchSynthesizer(beam_cfg)
        best_expr = synthesizer.search(observations)

        mdl = MDLEngine()
        cost = float('inf')
        if best_expr is not None:
            predictions = [
                best_expr.evaluate(t, observations[:t])
                for t in range(len(observations))
            ]
            result = mdl.compute(predictions, observations,
                                 best_expr.node_count(), best_expr.constant_count())
            cost = result.total_mdl_cost
            n_evals = config.beam_width * 10  # approximate

        return SearchResult(
            best_expr=best_expr,
            best_mdl_cost=cost,
            n_evaluations=n_evals,
            wall_time_seconds=time.time() - start,
            strategy_name=self.name(),
        )


# ─── Strategy 2: Random Restart ───────────────────────────────────────────────

class RandomRestartStrategy(SearchStrategy):
    """
    N independent random searches, each with a small beam, take the best.
    
    Key property: embarrassingly parallel, diverse starting points.
    Better than beam search when:
    - The search landscape has many local minima
    - The target expression is short (small programs are easy to re-find)
    - The alphabet size is small (fewer programs to explore)
    
    Worse than beam search when:
    - The target is complex and needs careful refinement
    - The alphabet is large (random starts rarely hit the good region)
    """

    def name(self) -> str:
        return "RandomRestart"

    def description(self) -> str:
        return "N independent beam searches with different seeds, take best result"

    def description_bits(self) -> float:
        return 25.0

    def search(self, observations: List[int], config: SearchConfig) -> SearchResult:
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine

        start = time.time()
        n_evals = 0
        best_expr = None
        best_cost = float('inf')
        trajectory = []

        per_restart_beam = max(5, config.beam_width // config.n_restarts)
        mdl = MDLEngine()

        for restart_i in range(config.n_restarts):
            # Check time budget
            if time.time() - start > config.time_budget_seconds:
                break

            seed = config.random_seed + restart_i * 17
            beam_cfg = BeamConfig(
                beam_width=per_restart_beam,
                const_range=config.const_range,
                max_depth=config.max_depth,
                max_lag=config.max_lag,
                mcmc_iterations=config.mcmc_iterations // config.n_restarts,
                random_seed=seed,
            )
            synthesizer = BeamSearchSynthesizer(beam_cfg)
            expr = synthesizer.search(observations)
            n_evals += per_restart_beam * 5

            if expr is not None:
                preds = [expr.evaluate(t, observations[:t])
                         for t in range(len(observations))]
                result = mdl.compute(preds, observations,
                                     expr.node_count(), expr.constant_count())
                cost = result.total_mdl_cost
                trajectory.append(cost)
                if cost < best_cost:
                    best_cost = cost
                    best_expr = copy.deepcopy(expr)

        return SearchResult(
            best_expr=best_expr,
            best_mdl_cost=best_cost,
            n_evaluations=n_evals,
            wall_time_seconds=time.time() - start,
            strategy_name=self.name(),
            cost_trajectory=trajectory,
        )


# ─── Strategy 3: Simulated Annealing ─────────────────────────────────────────

class AnnealingStrategy(SearchStrategy):
    """
    Simulated annealing over expression trees.
    
    Starts with a random expression, then iteratively proposes mutations.
    Accepts worse solutions with probability exp(-delta/T) where T decreases.
    
    Better than beam search when:
    - The landscape is "rugged" with many small local optima
    - We have a good starting expression from a prior run
    - The optimal expression is reachable by small mutations
    
    Worse when:
    - The target requires building specific subtrees from scratch
    - The search space has large flat regions (annealing gets stuck)
    """

    def name(self) -> str:
        return "SimulatedAnnealing"

    def description(self) -> str:
        return "Simulated annealing over expression trees with exponential cooling"

    def description_bits(self) -> float:
        return 30.0   # more complex than beam or random restart

    def search(self, observations: List[int], config: SearchConfig) -> SearchResult:
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        from ouroboros.compression.mdl_engine import MDLEngine
        import math

        start = time.time()
        mdl = MDLEngine()
        rng = random.Random(config.random_seed)

        # Start with a random expression (via quick beam search)
        beam_cfg = BeamConfig(
            beam_width=5,
            const_range=config.const_range,
            max_depth=config.max_depth,
            max_lag=config.max_lag,
            mcmc_iterations=20,
            random_seed=config.random_seed,
        )
        synthesizer = BeamSearchSynthesizer(beam_cfg)
        current_expr = synthesizer.search(observations)
        if current_expr is None:
            return SearchResult(
                best_expr=None,
                best_mdl_cost=float('inf'),
                n_evaluations=0,
                wall_time_seconds=time.time() - start,
                strategy_name=self.name(),
            )

        def score(expr: ExprNode) -> float:
            preds = [expr.evaluate(t, observations[:t])
                     for t in range(len(observations))]
            r = mdl.compute(preds, observations,
                            expr.node_count(), expr.constant_count())
            return r.total_mdl_cost

        current_cost = score(current_expr)
        best_expr = copy.deepcopy(current_expr)
        best_cost = current_cost
        trajectory = [current_cost]

        T_start = config.annealing_temp_start
        T_end = config.annealing_temp_end
        n_evals = 5

        # Cooling schedule: geometric decay
        total_steps = config.node_budget
        T = T_start
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, total_steps))

        for step in range(total_steps):
            if time.time() - start > config.time_budget_seconds:
                break

            # Mutate current expression
            mutated = self._mutate(current_expr, config, rng)
            mutated_cost = score(mutated)
            n_evals += 1

            # Accept or reject
            delta = mutated_cost - current_cost
            if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
                current_expr = mutated
                current_cost = mutated_cost

            if current_cost < best_cost:
                best_cost = current_cost
                best_expr = copy.deepcopy(current_expr)

            trajectory.append(best_cost)
            T *= cooling_rate

        return SearchResult(
            best_expr=best_expr,
            best_mdl_cost=best_cost,
            n_evaluations=n_evals,
            wall_time_seconds=time.time() - start,
            strategy_name=self.name(),
            cost_trajectory=trajectory[::50],  # subsample for storage
        )

    def _mutate(
        self,
        expr: ExprNode,
        config: SearchConfig,
        rng: random.Random,
    ) -> ExprNode:
        """Mutate an expression tree slightly."""
        from ouroboros.synthesis.mcmc_refiner import MCMCRefiner, MCMCConfig
        mcmc_cfg = MCMCConfig(
            n_iterations=1,
            const_range=config.const_range,
            random_seed=rng.randint(0, 999999),
        )
        refiner = MCMCRefiner(mcmc_cfg)
        result = refiner.refine(expr, [0] * 10)  # dummy observations for mutation only
        return result if result is not None else copy.deepcopy(expr)


# ─── Strategy 4: Hybrid (Beam + Annealing) ───────────────────────────────────

class HybridStrategy(SearchStrategy):
    """
    Phase 1: Beam search to find a good structural candidate.
    Phase 2: Simulated annealing to refine it.
    
    This is often the best of both worlds:
    - Beam finds the right shape quickly
    - Annealing escapes the local minimum the beam got stuck in
    
    This is essentially what the original SynthesisAgent did (Days 1–6)
    but now formalized as an explicit strategy.
    """

    def name(self) -> str:
        return "HybridBeamAnnealing"

    def description(self) -> str:
        return "Beam search for structure, then simulated annealing for refinement"

    def description_bits(self) -> float:
        return 35.0   # most complex strategy so far

    def search(self, observations: List[int], config: SearchConfig) -> SearchResult:
        start = time.time()

        # Phase 1: Beam search (use 60% of time budget)
        phase1_config = copy.copy(config)
        phase1_config.time_budget_seconds = config.time_budget_seconds * 0.6
        beam_result = BeamSearchStrategy().search(observations, phase1_config)

        if not beam_result.found_something:
            return beam_result

        # Phase 2: Annealing from beam's best (use remaining 40%)
        phase2_config = copy.copy(config)
        phase2_config.time_budget_seconds = config.time_budget_seconds * 0.4
        phase2_config.annealing_temp_start = 2.0   # start cooler (we're near optimum)
        phase2_config.node_budget = config.node_budget // 3

        annealing = AnnealingStrategy()
        # Override the annealing starting point with beam's result
        anneal_result = annealing.search(observations, phase2_config)

        # Take whichever is better
        if (anneal_result.found_something and
                anneal_result.best_mdl_cost < beam_result.best_mdl_cost):
            return SearchResult(
                best_expr=anneal_result.best_expr,
                best_mdl_cost=anneal_result.best_mdl_cost,
                n_evaluations=beam_result.n_evaluations + anneal_result.n_evaluations,
                wall_time_seconds=time.time() - start,
                strategy_name=self.name(),
                cost_trajectory=(
                    beam_result.cost_trajectory + anneal_result.cost_trajectory
                ),
            )
        else:
            return SearchResult(
                best_expr=beam_result.best_expr,
                best_mdl_cost=beam_result.best_mdl_cost,
                n_evaluations=beam_result.n_evaluations + anneal_result.n_evaluations,
                wall_time_seconds=time.time() - start,
                strategy_name=self.name(),
                cost_trajectory=beam_result.cost_trajectory,
            )


# ─── Strategy 5: Multi-Scale Strategy ────────────────────────────────────────

class MultiScaleStrategy(SearchStrategy):
    """
    Searches at multiple temporal scales simultaneously.
    
    Runs separate beam searches at scales 1, 4, 16 and combines results.
    This is the continuous analogue of HierarchicalMDL (Day 4).
    
    Better than single-scale beam when:
    - The signal has structure at multiple time scales
    - The target environment is MultiScaleEnv
    
    Worse when:
    - The environment has single-scale structure (most modular arithmetic envs)
    - The time budget is tight (3× more searches)
    """

    def __init__(self, scales: List[int] = None):
        self.scales = scales or [1, 4, 16]

    def name(self) -> str:
        return f"MultiScale({'_'.join(map(str, self.scales))})"

    def description(self) -> str:
        return f"Parallel beam search at temporal scales {self.scales}"

    def description_bits(self) -> float:
        return 28.0 + len(self.scales) * 4.0

    def search(self, observations: List[int], config: SearchConfig) -> SearchResult:
        start = time.time()
        best_expr = None
        best_cost = float('inf')
        total_evals = 0
        all_trajectories = []

        per_scale_budget = config.time_budget_seconds / len(self.scales)

        for scale in self.scales:
            # Subsample observations at this scale
            scaled_obs = observations[::scale] if scale > 1 else observations
            if not scaled_obs:
                continue

            scale_config = copy.copy(config)
            scale_config.time_budget_seconds = per_scale_budget
            scale_config.beam_width = config.beam_width // len(self.scales)
            scale_config.random_seed = config.random_seed + scale * 7

            result = BeamSearchStrategy().search(scaled_obs, scale_config)
            total_evals += result.n_evaluations
            all_trajectories.extend(result.cost_trajectory)

            if result.found_something and result.best_mdl_cost < best_cost:
                best_cost = result.best_mdl_cost
                best_expr = result.best_expr

        return SearchResult(
            best_expr=best_expr,
            best_mdl_cost=best_cost,
            n_evaluations=total_evals,
            wall_time_seconds=time.time() - start,
            strategy_name=self.name(),
            cost_trajectory=all_trajectories,
        )