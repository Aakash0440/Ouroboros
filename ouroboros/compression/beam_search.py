import numpy as np
from typing import List, Tuple, Optional
from ouroboros.compression.program_synthesis import (
    ExprNode, NodeType, C, T, ADD, MUL, MOD, PREV, PRIME
)
from ouroboros.compression.mdl import MDLCost, naive_bits

class BeamSearchSynthesizer:
    def __init__(
        self,
        beam_width: int = 50,
        max_depth: int = 3,
        const_range: int = 20,
        alphabet_size: int = 256,
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
        # PREV nodes for recurrence
        for lag in range(1, 4):
            leaves.append(PREV(lag))
        
        # ✅ FIX: Prime is now a fundamental leaf starting point
        leaves.append(PRIME(T())) 
        return leaves

    def _score(self, expr: ExprNode, actuals: List[int]) -> float:
        n = len(actuals)
        if expr.has_prev():
            seeds = actuals[:3]
            preds = expr.predict_sequence(n, self.alphabet_size, initial_history=seeds)
        else:
            preds = [expr.evaluate(t) % self.alphabet_size for t in range(n)]
        prog_bytes = expr.to_bytes()
        return self.mdl.total_cost(prog_bytes, preds, actuals, self.alphabet_size)

    def _expand(self, node: ExprNode) -> List[ExprNode]:
        """Generate depth+1 expressions."""
        if node.depth() >= self.max_depth:
            return []

        leaves = self._leaf_nodes()[:self.const_range + 1]
        expansions = []

        # ✅ FIX: Allow any numeric node to be wrapped in PRIME
        if node.node_type != NodeType.PRIME:
            expansions.append(PRIME(node))

        for leaf in leaves:
            expansions.append(ADD(node, leaf))
            expansions.append(MUL(node, leaf))
            if leaf.node_type == NodeType.CONST and leaf.value > 0:
                expansions.append(MOD(node, leaf))
            if node.contains_time():
                expansions.append(MOD(leaf, node))

        return expansions

    def _structural_seeds(self) -> List[ExprNode]:
        """Templates that preserve prime and modular structure."""
        seeds: List[ExprNode] = []
        max_c = self.const_range

        # Basic structural seeds
        for c in range(1, max_c + 1):
            seeds.append(MUL(T(), C(c)))
            seeds.append(MOD(T(), C(c)))
            # ✅ FIX: Prime-based seeds
            seeds.append(PRIME(T()))
            seeds.append(MOD(PRIME(T()), C(c)))

        # Complex templates: (a * prime(t) + b) mod m
        for a in [1, 3]:
            for b in [0, 7]:
                inner = ADD(MUL(PRIME(T()), C(a)), C(b))
                for m in [13, 17, 19]:
                    seeds.append(MOD(inner, C(m)))

        return seeds

    def search(self, actuals: List[int], verbose: bool = False) -> Tuple[ExprNode, float]:
        if not actuals: return C(0), float('inf')
        
        # Initial scoring
        scored = [(self._score(n, actuals), n) for n in self._leaf_nodes()]
        scored.sort(key=lambda x: x[0])
        beam = scored[:self.beam_width]

        # Inject structural seeds
        seed_scored = [(self._score(n, actuals), n) for n in self._structural_seeds()]
        seed_scored.sort(key=lambda x: x[0])
        
        combined = beam + seed_scored[:max(10, self.beam_width // 2)]
        combined.sort(key=lambda x: x[0])
        
        seen = set()
        deduped = []
        for cost, expr in combined:
            key = expr.to_string()
            if key not in seen:
                seen.add(key); deduped.append((cost, expr))
        beam = deduped[:self.beam_width]

        # Expansion loop
        for depth in range(1, self.max_depth + 1):
            new_candidates = []
            for _, node in beam:
                for expanded in self._expand(node):
                    new_candidates.append((self._score(expanded, actuals), expanded))
            
            if not new_candidates: break
            
            all_c = beam + new_candidates
            all_c.sort(key=lambda x: x[0])
            
            seen, deduped = set(), []
            for cost, expr in all_c:
                key = expr.to_string()
                if key not in seen:
                    seen.add(key); deduped.append((cost, expr))
                if len(deduped) >= self.beam_width: break
            beam = deduped
            
        return beam[0][1], beam[0][0]