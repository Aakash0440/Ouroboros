cat > ouroboros/meta/strategy_library.py << 'EOF'
"""
SearchStrategyLibrary — Registry of available search strategies.

Agents propose switching to a different strategy from this library.
The library is the "search algorithm space" that Layer 3 explores.

Think of it as:
  Layer 1: explores hyperparameter space (beam_width ∈ [5, 100])
  Layer 2: explores objective space (lambda_prog ∈ [0.1, 20.0])
  Layer 3: explores algorithm space (strategy ∈ {BeamSearch, RandomRestart, ...})

The key difference from Layers 1 and 2: the search space is DISCRETE
(a finite set of strategies), not continuous. This means Layer 3 uses
a different proposal mechanism — enumeration + ranking rather than
gradient-like perturbation.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from ouroboros.meta.search_strategy import (
    SearchStrategy, SearchConfig, SearchResult,
    BeamSearchStrategy, RandomRestartStrategy,
    AnnealingStrategy, HybridStrategy, MultiScaleStrategy,
)


# ─── Strategy Proposal ────────────────────────────────────────────────────────

@dataclass
class StrategyProposal:
    """
    An agent's proposal to switch to a different search strategy.
    
    The proposal contains evidence: the current strategy achieves
    cost C1 on training data, the proposed strategy achieves cost C2 < C1.
    """
    proposing_agent: str
    current_strategy_name: str
    proposed_strategy_name: str
    training_env_name: str

    # Evidence from training environment
    current_best_cost: float
    proposed_best_cost: float

    # Runtime comparison
    current_time_seconds: float
    proposed_time_seconds: float

    # Derived
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
        """Bits saved per second of extra runtime."""
        extra_time = max(0.01, self.proposed_time_seconds - self.current_time_seconds)
        return self.cost_improvement / extra_time

    def description(self) -> str:
        return (
            f"StrategyProposal: {self.current_strategy_name} → {self.proposed_strategy_name}\n"
            f"  Cost: {self.current_best_cost:.2f} → {self.proposed_best_cost:.2f} bits "
            f"(saved {self.cost_improvement:.2f})\n"
            f"  Time: {self.current_time_seconds:.2f}s → {self.proposed_time_seconds:.2f}s "
            f"({self.time_overhead_fraction*100:+.1f}%)"
        )


@dataclass
class StrategyEvaluationResult:
    """Result of the StrategyProofMarket evaluating a proposal."""
    proposal: StrategyProposal
    approved: bool
    validation_env_name: str

    # MDL costs on validation environment
    validation_current_cost: float
    validation_proposed_cost: float

    # Did proposed strategy perform better OOD?
    validation_improvement: float = field(init=False)
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        self.validation_improvement = (
            self.validation_current_cost - self.validation_proposed_cost
        )

    def description(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        return (
            f"{status}: {self.proposal.current_strategy_name} → "
            f"{self.proposal.proposed_strategy_name}\n"
            f"  Training improvement: {self.proposal.cost_improvement:.2f} bits\n"
            f"  Validation improvement: {self.validation_improvement:.2f} bits\n"
            f"  Verdict: {self.rejection_reason or 'Genuine improvement'}"
        )


# ─── Strategy Library ─────────────────────────────────────────────────────────

class SearchStrategyLibrary:
    """
    Registry of all available search strategies.
    
    Agents use this to enumerate candidate strategies when making
    Layer 3 proposals. The library is fixed at compile time but
    could be extended (adding new strategies is itself a research direction).
    """

    def __init__(self):
        self._strategies: Dict[str, SearchStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all built-in strategies."""
        strategies = [
            BeamSearchStrategy(),
            RandomRestartStrategy(),
            AnnealingStrategy(),
            HybridStrategy(),
            MultiScaleStrategy(scales=[1, 4, 16]),
            MultiScaleStrategy(scales=[1, 2, 4, 8]),  # finer multi-scale
        ]
        for s in strategies:
            self._strategies[s.name()] = s

    def register(self, strategy: SearchStrategy) -> None:
        """Add a new strategy to the library."""
        self._strategies[strategy.name()] = strategy

    def get(self, name: str) -> Optional[SearchStrategy]:
        return self._strategies.get(name)

    def all_strategies(self) -> List[SearchStrategy]:
        return list(self._strategies.values())

    def all_names(self) -> List[str]:
        return list(self._strategies.keys())

    def get_alternatives_to(self, current_name: str) -> List[SearchStrategy]:
        """Return all strategies that are NOT the current one."""
        return [s for name, s in self._strategies.items() if name != current_name]

    def description_bits_for(self, strategy_name: str) -> float:
        """MDL cost of naming/describing a strategy."""
        s = self.get(strategy_name)
        if s is None:
            return 64.0   # unknown strategy costs more
        return s.description_bits()


# Global shared library instance
STRATEGY_LIBRARY = SearchStrategyLibrary()
