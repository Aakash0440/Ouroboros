"""
Shim: wraps ouroboros.compression.mcmc_refiner (if it exists) or provides
a minimal stub so AnnealingStrategy._mutate can call:

    refiner = MCMCRefiner(MCMCConfig(n_iterations=1, const_range=30, random_seed=42))
    result = refiner.refine(expr, observations)
"""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass
from typing import List, Optional

from ouroboros.synthesis.expr_node import ExprNode, NodeType


@dataclass
class MCMCConfig:
    n_iterations: int = 150
    const_range: int = 30
    random_seed: int = 42


try:
    # Use the real refiner if it exists and has the right interface
    from ouroboros.compression.mcmc_refiner import MCMCRefiner as _RealRefiner  # type: ignore
    _HAS_REAL = True
except ImportError:
    _HAS_REAL = False


class MCMCRefiner:
    """
    Minimal MCMC refiner: randomly mutates one node in the expression tree.
    Used by AnnealingStrategy for single-step mutations.
    """

    def __init__(self, config: MCMCConfig = None):
        self.cfg = config or MCMCConfig()
        self._rng = random.Random(self.cfg.random_seed)

    def refine(
        self,
        expr: ExprNode,
        observations: List[int],
    ) -> Optional[ExprNode]:
        """Return a mutated copy of expr (single random perturbation)."""
        if _HAS_REAL:
            try:
                r = _RealRefiner(
                    n_iterations=self.cfg.n_iterations,
                    const_range=self.cfg.const_range,
                    random_seed=self.cfg.random_seed,
                )
                return r.refine(expr, observations)
            except Exception:
                pass  # fall through to stub

        return self._mutate(copy.deepcopy(expr))

    def _mutate(self, node: ExprNode) -> ExprNode:
        """Walk the tree and randomly perturb one CONST node."""
        nodes: list[ExprNode] = []
        self._collect(node, nodes)
        const_nodes = [n for n in nodes if n.node_type == NodeType.CONST]
        if const_nodes:
            target = self._rng.choice(const_nodes)
            delta = self._rng.randint(-3, 3)
            target.value = max(0, (target.value or 0) + delta)
        return node

    def _collect(self, node: ExprNode, out: list) -> None:
        out.append(node)
        for child in (node.left, node.right, node.extra):
            if child is not None:
                self._collect(child, out)