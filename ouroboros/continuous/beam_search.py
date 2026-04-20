"""
Continuous beam search + L-BFGS constant refinement.

Phase 1 — Structural beam search:
  Like the discrete beam search, we enumerate expression tree structures
  and score each using GaussianMDL. The top-K structures survive each round.
  
Phase 2 — Constant tuning (L-BFGS):
  For the top-B structures, we treat all CONST_REAL nodes as free variables
  and run gradient-based optimization (via scipy) to minimize MDL cost.
  This is the key difference from discrete synthesis — continuous constants
  can be tuned analytically.

The combination of structural search (finds the right shape) +
L-BFGS tuning (finds the right constants) is what lets agents
discover sin(1/7 * t) rather than sin(0.1000 * t).
"""

from __future__ import annotations
import copy
import itertools
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ouroboros.continuous.environments import ContinuousEnvironment
from ouroboros.continuous.expr_nodes import (
    ContinuousExprNode, ContinuousNodeType, ARITY
)
from ouroboros.continuous.mdl import compute_gaussian_mdl, GaussianMDLResult


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ContinuousBeamConfig:
    beam_width: int = 20
    max_depth: int = 4
    n_initial_constants: int = 15    # how many constant values to try initially
    constant_range: float = 10.0     # CONST_REAL values in [-range, range]
    lbfgs_iterations: int = 50       # gradient steps for constant tuning
    lbfgs_top_k: int = 5            # tune constants for top-K beam survivors
    random_seed: int = 42
    enable_lbfgs: bool = True        # set False to skip constant tuning

    # Which node types to include in search
    allow_sin: bool = True
    allow_cos: bool = True
    allow_exp: bool = True
    allow_log: bool = True
    allow_sqrt: bool = False         # off by default (sqrt(|x|) is rare)
    allow_prev: bool = False         # off by default (needs previous obs)
    max_lag: int = 3


@dataclass
class ContinuousCandidate:
    """A beam search candidate: expression + its MDL score."""
    expr: ContinuousExprNode
    mdl: GaussianMDLResult
    source: str = "beam"    # "beam", "lbfgs", "mutation"


class ContinuousBeamSearch:
    """
    Beam search over continuous expression trees, followed by L-BFGS
    constant optimization for the top survivors.
    """

    def __init__(self, config: ContinuousBeamConfig = None):
        self.cfg = config or ContinuousBeamConfig()
        self._rng = random.Random(self.cfg.random_seed)

    def _allowed_node_types(self) -> List[ContinuousNodeType]:
        """Which node types are enabled in this search."""
        types = [
            ContinuousNodeType.CONST_REAL,
            ContinuousNodeType.TIME_REAL,
            ContinuousNodeType.ADD_REAL,
            ContinuousNodeType.SUB_REAL,
            ContinuousNodeType.MUL_REAL,
            ContinuousNodeType.DIV_REAL,
        ]
        if self.cfg.allow_sin:
            types.append(ContinuousNodeType.SIN)
        if self.cfg.allow_cos:
            types.append(ContinuousNodeType.COS)
        if self.cfg.allow_exp:
            types.append(ContinuousNodeType.EXP)
        if self.cfg.allow_log:
            types.append(ContinuousNodeType.LOG)
        if self.cfg.allow_sqrt:
            types.append(ContinuousNodeType.SQRT)
        if self.cfg.allow_prev:
            types.append(ContinuousNodeType.PREV_REAL)
        return types

    def _random_terminal(self) -> ContinuousExprNode:
        """Generate a random terminal: CONST_REAL, TIME_REAL, or PREV_REAL."""
        choice = self._rng.random()
        if choice < 0.4:
            v = self._rng.uniform(-self.cfg.constant_range, self.cfg.constant_range)
            return ContinuousExprNode.const(round(v, 3))
        elif choice < 0.85:
            return ContinuousExprNode.time()
        else:
            if self.cfg.allow_prev:
                lag = self._rng.randint(1, self.cfg.max_lag)
                return ContinuousExprNode.prev(lag)
            return ContinuousExprNode.time()

    def _random_expr(self, depth: int, max_depth: int) -> ContinuousExprNode:
        """Recursively build a random expression tree up to max_depth."""
        allowed = self._allowed_node_types()

        if depth >= max_depth:
            # Force terminal
            terminals = [
                t for t in allowed
                if ARITY.get(t, 0) == 0
            ]
            nt = self._rng.choice(terminals)
            node = ContinuousExprNode(nt)
            if nt == ContinuousNodeType.CONST_REAL:
                node.value_f = round(
                    self._rng.uniform(-self.cfg.constant_range, self.cfg.constant_range),
                    3
                )
            elif nt == ContinuousNodeType.PREV_REAL:
                node.lag = self._rng.randint(1, self.cfg.max_lag)
            return node

        # Pick a random node type (bias toward lower arity at deep levels)
        nt = self._rng.choice(allowed)
        arity = ARITY.get(nt, 0)
        node = ContinuousExprNode(nt)

        if arity == 0:
            if nt == ContinuousNodeType.CONST_REAL:
                node.value_f = round(
                    self._rng.uniform(-self.cfg.constant_range, self.cfg.constant_range),
                    3
                )
            elif nt == ContinuousNodeType.PREV_REAL:
                node.lag = self._rng.randint(1, self.cfg.max_lag)
        elif arity == 1:
            node.left = self._random_expr(depth + 1, max_depth)
        elif arity == 2:
            node.left = self._random_expr(depth + 1, max_depth)
            node.right = self._random_expr(depth + 1, max_depth)

        return node

    def _score(
        self,
        expr: ContinuousExprNode,
        observations: List[float]
    ) -> GaussianMDLResult:
        """Score an expression against a sequence of continuous observations."""
        history: List[float] = []
        predictions: List[float] = []

        for t, actual in enumerate(observations):
            pred = expr.evaluate(t, history)
            predictions.append(pred)
            history.append(actual)  # use actual (not prediction) as history

        return compute_gaussian_mdl(
            predictions=predictions,
            actuals=observations,
            program_node_count=expr.node_count(),
            program_constant_count=expr.constant_count(),
        )

    def _mutate(self, expr: ContinuousExprNode) -> ContinuousExprNode:
        """
        Produce a mutated copy of an expression.
        Mutations: perturb a constant, swap a unary op, replace a subtree.
        """
        expr_copy = copy.deepcopy(expr)
        self._mutate_inplace(expr_copy)
        return expr_copy

    def _mutate_inplace(self, node: ContinuousExprNode) -> None:
        """Randomly mutate one node in the tree."""
        # 40% chance: perturb a constant if this is one
        if node.node_type == ContinuousNodeType.CONST_REAL and self._rng.random() < 0.4:
            delta = self._rng.gauss(0, 0.5)
            node.value_f = round(node.value_f + delta, 4)
            return

        # 20% chance: replace this node's subtree with a fresh random one
        if self._rng.random() < 0.2:
            fresh = self._random_expr(0, 2)
            node.node_type = fresh.node_type
            node.value_f = fresh.value_f
            node.lag = fresh.lag
            node.left = fresh.left
            node.right = fresh.right
            return

        # Otherwise recurse into children
        if node.left is not None:
            self._mutate_inplace(node.left)
        if node.right is not None and self._rng.random() < 0.5:
            self._mutate_inplace(node.right)

    def _lbfgs_tune(
        self,
        expr: ContinuousExprNode,
        observations: List[float]
    ) -> Tuple[ContinuousExprNode, GaussianMDLResult]:
        """
        Tune all CONST_REAL values in expr using scipy L-BFGS-B.
        Returns the tuned expression and its MDL score.
        
        If scipy is not available or optimization fails, returns
        the original expression unchanged.
        """
        try:
            from scipy.optimize import minimize
            import numpy as np
        except ImportError:
            return expr, self._score(expr, observations)

        # Extract all CONST_REAL nodes
        const_nodes: List[ContinuousExprNode] = []
        self._collect_const_nodes(expr, const_nodes)

        if not const_nodes:
            return expr, self._score(expr, observations)

        initial_values = np.array([n.value_f for n in const_nodes])

        def objective(params):
            # Temporarily set constant values
            for node, val in zip(const_nodes, params):
                node.value_f = float(val)
            result = self._score(expr, observations)
            return result.total_mdl_cost

        try:
            opt_result = minimize(
                objective,
                initial_values,
                method='L-BFGS-B',
                options={'maxiter': self.cfg.lbfgs_iterations, 'ftol': 1e-10}
            )
            # Apply optimal values
            for node, val in zip(const_nodes, opt_result.x):
                node.value_f = round(float(val), 6)
        except Exception:
            # Restore initial values on failure
            for node, val in zip(const_nodes, initial_values):
                node.value_f = float(val)

        final_result = self._score(expr, observations)
        return expr, final_result

    def _collect_const_nodes(
        self,
        node: ContinuousExprNode,
        collector: List[ContinuousExprNode]
    ) -> None:
        if node.node_type == ContinuousNodeType.CONST_REAL:
            collector.append(node)
        if node.left:
            self._collect_const_nodes(node.left, collector)
        if node.right:
            self._collect_const_nodes(node.right, collector)

    def search(
        self,
        observations: List[float],
        verbose: bool = False
    ) -> List[ContinuousCandidate]:
        """
        Full beam search + L-BFGS constant tuning.
        
        Returns the top beam_width candidates sorted by total MDL cost (ascending).
        Lower MDL cost = better compression = stronger mathematical discovery.
        """
        beam_width = self.cfg.beam_width

        # ── Phase 1: Initialize beam with random candidates ────────────────
        candidates: List[ContinuousCandidate] = []

        # Seed with target-shaped expressions (heuristic warm start)
        seeds = self._warm_start_seeds()
        for seed_expr in seeds:
            mdl = self._score(seed_expr, observations)
            candidates.append(ContinuousCandidate(seed_expr, mdl, "seed"))

        # Fill rest of beam with random expressions
        n_random = beam_width * 3
        for _ in range(n_random):
            expr = self._random_expr(0, self.cfg.max_depth)
            mdl = self._score(expr, observations)
            candidates.append(ContinuousCandidate(expr, mdl, "random"))

        # Sort and keep top beam_width
        candidates.sort(key=lambda c: c.mdl.total_mdl_cost)
        beam = candidates[:beam_width]

        if verbose:
            print(f"  Initial beam best MDL: {beam[0].mdl.total_mdl_cost:.2f} bits")
            print(f"  Initial beam best R²: {beam[0].mdl.r_squared:.4f}")

        # ── Phase 2: Iterative mutation ────────────────────────────────────
        for iteration in range(10):  # 10 mutation rounds
            new_candidates = list(beam)  # keep existing beam

            for cand in beam:
                for _ in range(3):  # 3 mutations per beam member
                    mutated = self._mutate(cand.expr)
                    mdl = self._score(mutated, observations)
                    new_candidates.append(ContinuousCandidate(mutated, mdl, "mutation"))

            new_candidates.sort(key=lambda c: c.mdl.total_mdl_cost)
            beam = new_candidates[:beam_width]

        if verbose:
            print(f"  After mutation beam best MDL: {beam[0].mdl.total_mdl_cost:.2f} bits")
            print(f"  After mutation best R²: {beam[0].mdl.r_squared:.4f}")

        # ── Phase 3: L-BFGS constant tuning on top candidates ─────────────
        if self.cfg.enable_lbfgs:
            tuned_beam = []
            for cand in beam[:self.cfg.lbfgs_top_k]:
                expr_copy = copy.deepcopy(cand.expr)
                tuned_expr, tuned_mdl = self._lbfgs_tune(expr_copy, observations)
                tuned_beam.append(ContinuousCandidate(tuned_expr, tuned_mdl, "lbfgs"))
            for cand in beam[self.cfg.lbfgs_top_k:]:
                tuned_beam.append(cand)
            tuned_beam.sort(key=lambda c: c.mdl.total_mdl_cost)
            beam = tuned_beam

        if verbose:
            best = beam[0]
            print(f"  Final best: {best.expr.to_string()}")
            print(f"  Final MDL: {best.mdl.total_mdl_cost:.2f} bits")
            print(f"  Final R²: {best.mdl.r_squared:.4f}")
            print(f"  Compression ratio: {best.mdl.compression_ratio:.4f}")

        return beam

    def _warm_start_seeds(self) -> List[ContinuousExprNode]:
        """
        Return a set of heuristic seed expressions that cover
        common patterns. These are explored in Phase 1 to give the
        beam a good starting point.
        """
        seeds = []
        C = ContinuousExprNode

        # Constant seeds
        for v in [0.0, 1.0, -1.0, 0.5, 2.0, math.pi]:
            seeds.append(C.const(v))

        # Linear: t, 2*t, 0.5*t
        seeds.append(C.time())
        for v in [0.1, 0.5, 1.0, 2.0, -1.0]:
            seeds.append(C.mul(C.const(v), C.time()))

        # Sinusoidal: sin(f*t) for various f
        for f in [1/3, 1/5, 1/7, 1/11, 0.5, 1.0, 2.0]:
            seeds.append(C.sin(C.mul(C.const(f), C.time())))
            seeds.append(C.cos(C.mul(C.const(f), C.time())))

        # Exponential: exp(r*t) for various r
        for r in [0.01, 0.05, 0.1, 0.5, -0.1, -0.05]:
            seeds.append(C.exp(C.mul(C.const(r), C.time())))

        # Polynomial-like: t^2, t^0.5
        seeds.append(
            C.mul(C.time(), C.time())  # t²
        )
        seeds.append(
            C.add(C.mul(C.const(0.5), C.mul(C.time(), C.time())),
                  C.mul(C.const(-2.0), C.time()))   # 0.5*t² - 2t
        )

        return seeds


import math   # needed for pi in warm_start_seed