"""
MCMC refinement — updated for extended primitive set.

New mutation operations for new node types:
    LAG_DELTA:    PREV(k) → PREV(k±1)
    LAG_RANDOM:   PREV(k) → PREV(random lag in 1..max_lag)
    IF_SWAP:      swap then/else branches of IF node
    CONST_DELTA:  change CONST by ±1 (original)
    CONST_RANDOM: replace CONST with random value (original)
    SWAP_OP:      cycle through arithmetic ops (extended)
    SWAP_CHILDREN: swap left/right (original)
"""

import copy
import math
import numpy as np
from typing import List, Tuple, Optional
from ouroboros.compression.program_synthesis import (
    ExprNode, NodeType, LEAF_TYPES, BINARY_TYPES, TERNARY_TYPES,
    C, PREV
)
from ouroboros.compression.mdl import MDLCost


class MCMCRefiner:
    """
    Simulated annealing refinement — extended for new primitives.

    Args:
        num_iterations: Total MCMC steps
        temperature: Initial temperature
        cooling_rate: Decay per step
        const_range: Max constant for CONST_RANDOM
        max_lag: Max lag for LAG_RANDOM
        alphabet_size: For prediction clamping
        seed: Random seed
    """

    def __init__(
        self,
        num_iterations: int = 300,
        temperature: float = 20.0,
        cooling_rate: float = 0.985,
        const_range: int = 20,
        max_lag: int = 3,
        alphabet_size: int = 10,
        seed: int = 42
    ):
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.const_range = const_range
        self.max_lag = max_lag
        self.alphabet_size = alphabet_size
        self.rng = np.random.default_rng(seed)
        self.mdl = MDLCost()

    def _score(self, expr: ExprNode, actuals: List[int]) -> float:
        n = len(actuals)
        preds = expr.predict_sequence(n, self.alphabet_size)
        return self.mdl.total_cost(
            expr.to_bytes(), preds, actuals, self.alphabet_size
        )

    def _collect_nodes(self, expr: ExprNode) -> List[ExprNode]:
        nodes = [expr]
        if expr.left:  nodes.extend(self._collect_nodes(expr.left))
        if expr.right: nodes.extend(self._collect_nodes(expr.right))
        if expr.extra: nodes.extend(self._collect_nodes(expr.extra))
        return nodes

    def _mutate(self, expr: ExprNode) -> ExprNode:
        """One of 7 mutation types chosen uniformly."""
        expr_copy = copy.deepcopy(expr)
        nodes = self._collect_nodes(expr_copy)
        target = nodes[int(self.rng.integers(0, len(nodes)))]
        mutation = int(self.rng.integers(0, 7))

        if mutation == 0 and target.node_type == NodeType.CONST:
            # CONST_DELTA: ±1
            target.value = max(0, target.value + int(self.rng.choice([-1, 1])))

        elif mutation == 1 and target.node_type == NodeType.CONST:
            # CONST_RANDOM
            target.value = int(self.rng.integers(0, self.const_range + 1))

        elif mutation == 2 and target.node_type == NodeType.PREV:
            # LAG_DELTA: lag ±1
            current_lag = target.lag or 1
            delta = int(self.rng.choice([-1, 1]))
            target.lag = max(1, min(current_lag + delta, self.max_lag))

        elif mutation == 3 and target.node_type == NodeType.PREV:
            # LAG_RANDOM
            target.lag = int(self.rng.integers(1, self.max_lag + 1))

        elif mutation == 4 and target.node_type in BINARY_TYPES:
            # SWAP_CHILDREN
            target.left, target.right = target.right, target.left

        elif mutation == 5 and target.node_type in BINARY_TYPES - {NodeType.EQ, NodeType.LT}:
            # SWAP_OP: cycle through arithmetic ops
            arith_ops = [NodeType.ADD, NodeType.SUB, NodeType.MUL,
                         NodeType.MOD, NodeType.DIV]
            idx = arith_ops.index(target.node_type) if target.node_type in arith_ops else 0
            target.node_type = arith_ops[(idx + 1) % len(arith_ops)]

        elif mutation == 6 and target.node_type == NodeType.IF:
            # IF_SWAP: swap then/else branches
            target.right, target.extra = target.extra, target.right

        return expr_copy

    def refine(
        self,
        initial_expr: ExprNode,
        actuals: List[int],
        verbose: bool = False
    ) -> Tuple[ExprNode, float]:
        """Refine initial_expr by MCMC simulated annealing."""
        current = copy.deepcopy(initial_expr)
        current_cost = self._score(current, actuals)
        best = copy.deepcopy(current)
        best_cost = current_cost

        T = self.temperature

        for i in range(self.num_iterations):
            mutated = self._mutate(current)
            mutated_cost = self._score(mutated, actuals)
            delta = mutated_cost - current_cost

            if delta < 0 or self.rng.random() < math.exp(-delta / max(T, 1e-9)):
                current = mutated
                current_cost = mutated_cost

            if current_cost < best_cost:
                best = copy.deepcopy(current)
                best_cost = current_cost

                # Early exit: perfect
                preds = best.predict_sequence(len(actuals), self.alphabet_size)
                if all(p == a for p, a in zip(preds, actuals)):
                    if verbose:
                        print(f"  MCMC early exit at step {i}")
                    break

            T *= self.cooling_rate

            if verbose and i % 50 == 0:
                print(f"  MCMC {i:3d}: {best.to_string()!r}  cost={best_cost:.1f}")

        return best, best_cost