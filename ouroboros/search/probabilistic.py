"""PosteriorExpressionSampler — softmax distribution over beam candidates."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Any


@dataclass
class WeightedExpression:
    expression: Any
    mdl_cost: float
    probability: float


@dataclass
class PosteriorDistribution:
    expressions: List[WeightedExpression]

    @property
    def best(self) -> WeightedExpression:
        return max(self.expressions, key=lambda e: e.probability)


class PosteriorExpressionSampler:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_distribution(
        self,
        candidates: List[Tuple[Any, float]],
    ) -> PosteriorDistribution:
        """Softmax over negative MDL costs."""
        if not candidates:
            return PosteriorDistribution([])
        costs = [cost for _, cost in candidates]
        min_cost = min(costs)
        weights = [math.exp(-(c - min_cost) / max(self.temperature, 1e-10))
                   for c in costs]
        total = sum(weights)
        return PosteriorDistribution([
            WeightedExpression(expr, cost, w / total)
            for (expr, cost), w in zip(candidates, weights)
        ])