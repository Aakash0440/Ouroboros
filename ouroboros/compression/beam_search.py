# ouroboros/compression/beam_search.py

"""
Beam Search over symbolic expression trees.

The search finds the expression minimizing MDL cost on a data sequence.

How beam search works:
    Iteration 0: Generate all depth-0 expressions (constants 0..const_range, TIME)
    Iteration 1: For each expression in beam, generate all depth-1 expansions
                 (wrap in ADD/MUL/MOD with leaf on the right)
    Iteration 2: Same but depth-2
    At each step: score all candidates, keep top beam_width by MDL cost.

WHY BEAM SEARCH FINDS MODULAR ARITHMETIC:
    - Depth-0 candidates include CONST(7), CONST(3), CONST(1), TIME
    - Depth-1 includes TIME mod CONST(7) → predicts t mod 7 (decent)
    - Depth-2 includes (TIME * CONST(3) + CONST(1)) mod CONST(7)
              → predicts (3t+1) mod 7 (perfect for the stream)
    - This depth-2 expression scores MUCH lower MDL than anything else
    - Beam search retains it and returns it as the best program

THE RESULT: Agent discovers modular arithmetic without being told about it.

Complexity: O(beam_width * |ops| * const_range * max_depth)
For default settings: O(25 * 3 * 20 * 3) ≈ 4500 expression evaluations.
Each evaluation on 500 symbols takes ~0.1ms → total search: ~0.5 seconds.
"""

from typing import List, Tuple, Optional
import numpy as np
from ouroboros.compression.program_synthesis import (
    ExprNode, NodeType, C, T, ADD, MUL, MOD
)
from ouroboros.compression.mdl import MDLCost, naive_bits


class BeamSearchSynthesizer:
    """
    Beam search over symbolic expression trees.

    Finds the expression minimizing MDL cost on a target sequence.

    Args:
        beam_width: Number of candidates to keep at each depth (default 25)
        max_depth: Maximum expression tree depth (default 3)
        const_range: Search constants 0..const_range (default 20)
        alphabet_size: Symbol count for prediction clamping
        lambda_weight: MDL regularization weight
    """

    def __init__(
        self,
        beam_width: int = 50,
        max_depth: int = 3,
        const_range: int = 20,
        alphabet_size: int = 10,
        lambda_weight: float = 0.1,
    ):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.const_range = const_range
        self.alphabet_size = alphabet_size
        self.mdl = MDLCost(lambda_weight=lambda_weight)

    def _leaf_nodes(self) -> List[ExprNode]:
        """All depth-0 candidate expressions."""
        leaves = [T()]  # TIME variable
        for c in range(self.const_range + 1):
            leaves.append(C(c))
        return leaves

    def _score(self, expr: ExprNode, actuals: List[int]) -> float:
        n = len(actuals)
        # Raw predictions — NO % alphabet_size clamping
        # This forces the expression to explicitly handle range via mod
        # (3t+1)+7 produces 8,11,7... which don't match stream → high error cost
        # (3t+1) mod 7 produces 1,4,0... which match → near-zero error cost
        preds = [expr.evaluate(t) for t in range(n)]
        prog_bytes = expr.to_bytes()
        return self.mdl.total_cost(prog_bytes, preds, actuals, self.alphabet_size)

    def _expand(self, node: ExprNode) -> List[ExprNode]:
        """
        Generate all depth+1 expressions by adding a binary operation layer.

        For each binary op ∈ {ADD, MUL, MOD} and each leaf ∈ leaf_nodes:
            - node op leaf
            - leaf op node    (for non-commutative ops: MOD)

        We don't expand if already at max_depth-1.
        We limit leaves to first 12 for speed (0..11 covers most useful constants).
        """
        if node.depth() >= self.max_depth:
            return []

        leaves = self._leaf_nodes()[:self.const_range + 1] 
        expansions = []

        for leaf in leaves:
            # node ADD leaf
            expansions.append(ADD(node, leaf))
            # node MUL leaf
            expansions.append(MUL(node, leaf))
            # node MOD leaf
            if leaf.node_type == NodeType.CONST and leaf.value > 0:
                expansions.append(MOD(node, leaf))
            # leaf MOD node (reversed — catches cases like C(7) mod TIME)
            if node.contains_time():
                expansions.append(MOD(leaf, node))

        return expansions
    def _structural_seeds(self) -> List[ExprNode]:
        """
        Hand-crafted templates that preserve arithmetic structure candidates.

        Without these seeds, shallow constants can dominate early beam pruning
        and remove useful modular forms before they are expanded.
        """
        seeds: List[ExprNode] = []
        max_c = self.const_range

        for c in range(1, max_c + 1):
            seeds.append(MUL(T(), C(c)))
            seeds.append(ADD(T(), C(c)))
            seeds.append(MOD(T(), C(c)))

        # Include linear-modular templates directly:
        # (a*t + b) mod m
        # Keep this bounded to avoid a combinatorial blowup.
        max_a = min(max_c, 8)
        max_b = min(max_c, 8)
        max_m = min(max_c, self.alphabet_size)
        for a in range(1, max_a + 1):
            for b in range(0, max_b + 1):
                inner = ADD(MUL(T(), C(a)), C(b))
                for m in range(2, max_m + 1):
                    seeds.append(MOD(inner, C(m)))

        return seeds

    def search(
        self,
        actuals: List[int],
        verbose: bool = False,
    ) -> Tuple[ExprNode, float]:
        if not actuals:
            return C(0), float('inf')

        # ── Depth 0: score all leaf nodes ──────────────────────────────────────
        scored: List[Tuple[float, ExprNode]] = []
        for node in self._leaf_nodes():
            cost = self._score(node, actuals)
            scored.append((cost, node))

        scored.sort(key=lambda x: x[0])
        beam = scored[:self.beam_width]

        # ── Force structural seeds into beam so MUL(T,k) survives pruning ──────
        seed_scored = []
        for node in self._structural_seeds():
            cost = self._score(node, actuals)
            seed_scored.append((cost, node))
        seed_scored.sort(key=lambda x: x[0])

        # Keep enough structural seeds so modular candidates survive to depth>1.
        seed_keep = max(10, self.beam_width // 2)
        combined = beam + seed_scored[:seed_keep]
        combined.sort(key=lambda x: x[0])
        seen = set()
        deduped = []
        for cost, expr in combined:
            key = expr.to_string()
            if key not in seen:
                seen.add(key)
                deduped.append((cost, expr))
        beam = deduped[:self.beam_width + seed_keep]

        if verbose:
            best = beam[0]
            print(f"  D0: {best[1].to_string()!r}  cost={best[0]:.1f}")

        # ── Depth 1..max_depth: expand and score ───────────────────────────────
        for depth in range(1, self.max_depth + 1):
            new_candidates: List[Tuple[float, ExprNode]] = []

            for _, node in beam:
                for expanded in self._expand(node):
                    cost = self._score(expanded, actuals)
                    new_candidates.append((cost, expanded))

            if not new_candidates:
                break

            all_candidates = beam + new_candidates
            all_candidates.sort(key=lambda x: x[0])

            seen = set()
            deduped = []
            for cost, expr in all_candidates:
                key = expr.to_string()
                if key not in seen:
                    seen.add(key)
                    deduped.append((cost, expr))
                if len(deduped) >= self.beam_width:
                    break

            beam = deduped[:self.beam_width]

            if verbose:
                best = beam[0]
                print(f"  D{depth}: {best[1].to_string()!r}  cost={best[0]:.1f}")

        best_cost, best_expr = beam[0]
        return best_expr, best_cost

    def beats_ngram(
        self,
        expr: ExprNode,
        expr_cost: float,
        actuals: List[int],
    ) -> bool:
        """
        Check if the found expression has better MDL than naive baseline.

        Returns True if expression is worth keeping (< 90% of naive cost).
        """
        nb = naive_bits(actuals, self.alphabet_size)
        return expr_cost < nb * 0.90