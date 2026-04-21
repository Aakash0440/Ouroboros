"""
SearchStrategyLibrary — Registry of available search strategies.

Agents propose switching to a different strategy from this library.
The library is the "search algorithm space" that Layer 3 explores.

Think of it as:
  Layer 1: explores hyperparameter space (beam_width in [5, 100])
  Layer 2: explores objective space (lambda_prog in [0.1, 20.0])
  Layer 3: explores algorithm space (strategy in {BeamSearch, RandomRestart, ...})

The key difference from Layers 1 and 2: the search space is DISCRETE
(a finite set of strategies), not continuous. This means Layer 3 uses
a different proposal mechanism -- enumeration + ranking rather than
gradient-like perturbation.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ouroboros.meta.search_strategy import (
    SearchStrategy, SearchConfig, SearchResult,
    BeamSearchStrategy, RandomRestartStrategy,
    AnnealingStrategy, HybridStrategy, MultiScaleStrategy,
)


@dataclass
class StrategyProposal:
    proposing_agent: str
    current_strategy_name: str
    proposed_strategy_name: str
    training_env_name: str
    current_best_cost: float
    proposed_best_cost: float
    current_time_seconds: float
    proposed_time_seconds: float
    cost_improvement: float = field(init=False)
    time_overhead_fraction: float = field(init=False)

    def __post_init__(self):
        self.cost_improvement = self.current_best_cost - self.proposed_best_cost
        if self.current_time_seconds > 0:
            self.time_overhead_fraction = (
                (self.proposed_time_seconds - self.current_time_seconds)
                / self.current_time_seconds
            )
        else:
            self.time_overhead_fraction = 0.0

    @property
    def is_improvement(self) -> bool:
        return self.cost_improvement > 0.0

    @property
    def cost_efficiency(self) -> float:
        extra_time = max(0.01, self.proposed_time_seconds - self.current_time_seconds)
        return self.cost_improvement / extra_time


@dataclass
class StrategyEvaluationResult:
    proposal: StrategyProposal
    approved: bool
    validation_env_name: str
    validation_current_cost: float
    validation_proposed_cost: float
    validation_improvement: float = field(init=False)
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        self.validation_improvement = (
            self.validation_current_cost - self.validation_proposed_cost
        )


class SearchStrategyLibrary:
    def __init__(self):
        self._strategies: Dict[str, SearchStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        strategies = [
            BeamSearchStrategy(),
            RandomRestartStrategy(),
            AnnealingStrategy(),
            HybridStrategy(),
            MultiScaleStrategy(scales=[1, 4, 16]),
            MultiScaleStrategy(scales=[1, 2, 4, 8]),
        ]
        for s in strategies:
            self._strategies[s.name()] = s

    def register(self, strategy: SearchStrategy) -> None:
        self._strategies[strategy.name()] = strategy

    def get(self, name: str) -> Optional[SearchStrategy]:
        return self._strategies.get(name)

    def all_strategies(self) -> List[SearchStrategy]:
        return list(self._strategies.values())

    def all_names(self) -> List[str]:
        return list(self._strategies.keys())

    def get_alternatives_to(self, current_name: str) -> List[SearchStrategy]:
        return [s for name, s in self._strategies.items() if name != current_name]

    def description_bits_for(self, strategy_name: str) -> float:
        s = self.get(strategy_name)
        return s.description_bits() if s else 64.0


STRATEGY_LIBRARY = SearchStrategyLibrary()
