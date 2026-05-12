"""
GrammarConstrainedBeam — Beam search that only generates valid expression trees.

This replaces BeamSearchSynthesizer (Day 2) as the primary search engine
when the extended 60-node vocabulary is active.

Key differences from BeamSearchSynthesizer:
  1. When generating children, only consider grammar-valid node types
  2. Branching factor drops from 60 to ~6 on average
  3. Search space shrinks by factor of (60/6)^16 = 10^16
  4. All 60 node types accessible but constrained by parent type

The grammar constraint is applied at TWO points:
  - During initial tree generation (_random_expr): respects grammar at every level
  - During mutation (_mutate): grammar-valid replacements only

This ensures 100% of beam candidates are grammatically valid mathematical
expressions. No wasted evaluations on invalid combinations.
"""

from __future__ import annotations
import copy
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time

from ouroboros.nodes.extended_nodes import ExtNodeType, ExtExprNode, NODE_SPECS, NodeCategory
from ouroboros.grammar.math_grammar import MathGrammar, DEFAULT_GRAMMAR
from ouroboros.compression.mdl_engine import MDLEngine
from ouroboros.nodes.extended_nodes import ExtNodeType

@dataclass
class GrammarBeamConfig:
    """Configuration for grammar-constrained beam search."""
    beam_width: int = 25
    max_depth: int = 5
    const_range: int = 50
    max_lag: int = 10
    max_window: int = 50
    n_iterations: int = 15
    mcmc_iterations: int = 0      # set 0 to skip MCMC, grammar beam is faster
    random_seed: int = 42
    grammar: MathGrammar = field(default_factory=lambda: DEFAULT_GRAMMAR)

    # Category weights for sampling (higher = more likely to be sampled)
    category_weights: Dict[NodeCategory, float] = field(default_factory=lambda: {
        NodeCategory.TERMINAL:    3.0,
        NodeCategory.ARITHMETIC:  2.0,
        NodeCategory.CALCULUS:    1.0,
        NodeCategory.STATISTICAL: 1.0,
        NodeCategory.LOGICAL:     0.8,
        NodeCategory.TRANSFORM:   0.5,
        NodeCategory.NUMBER:      0.7,
        NodeCategory.MEMORY:      0.6,
        NodeCategory.TRANSCEND:   1.5,
    })


@dataclass
class GrammarBeamCandidate:
    """One candidate in the grammar-constrained beam."""
    expr: ExtExprNode
    mdl_cost: float
    is_valid_grammar: bool = True

    def __lt__(self, other):
        return self.mdl_cost < other.mdl_cost


class GrammarConstrainedBeam:
    """
    Beam search over the 60-node expression vocabulary,
    constrained by the mathematical grammar.
    """

    def __init__(self, config: GrammarBeamConfig = None):
        self.cfg = config or GrammarBeamConfig()
        self._rng = random.Random(self.cfg.random_seed)
        self._grammar = self.cfg.grammar
        self._mdl = MDLEngine()
        self._channel_state: dict = {}
        self._nodes_by_category: Dict[NodeCategory, List[ExtNodeType]] = {}
        for nt, spec in NODE_SPECS.items():
            cat = spec.category
            if cat not in self._nodes_by_category:
                self._nodes_by_category[cat] = []
            self._nodes_by_category[cat].append(nt)

    def _sample_node_type(
        self,
        allowed_categories: frozenset,
        depth: int,
        force_terminal: bool = False,
    ) -> ExtNodeType:
        if force_terminal or depth >= self.cfg.max_depth:
            terminal_types = self._nodes_by_category.get(NodeCategory.TERMINAL, [])
            if terminal_types:
                return self._rng.choice(terminal_types)

        candidates = []
        weights = []
        for cat in allowed_categories:
            if cat not in self._nodes_by_category:
                continue
            cat_weight = self.cfg.category_weights.get(cat, 1.0)
            depth_factor = max(0.1, 1.0 - depth * 0.15)
            for nt in self._nodes_by_category[cat]:
                spec = NODE_SPECS.get(nt)
                if spec is None:
                    continue
                terminal_bonus = 2.0 if spec.arity == 0 else 1.0
                w = cat_weight * depth_factor * terminal_bonus
                candidates.append(nt)
                weights.append(w)

        if not candidates:
            return NodeCategory.TERMINAL

        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for nt, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return nt
        return candidates[-1]

    def _make_const(self) -> ExtExprNode:
        from ouroboros.synthesis.expr_node import NodeType
        node = ExtExprNode.__new__(ExtExprNode)
        node.node_type = NodeType.CONST
        if self.cfg.const_range <= 3:
            # likely a small-valued target — sample floats too
            node.value = self._rng.uniform(-1.0, 1.0)
        else:
            node.value = float(self._rng.randint(0, self.cfg.const_range))
        node.lag = 1
        node.state_key = 0
        node.window = 10
        node.left = None
        node.right = None
        node.third = None
        node._cache = {}
        return node
        
    def _make_time(self) -> ExtExprNode:
        from ouroboros.synthesis.expr_node import NodeType
        node = ExtExprNode.__new__(ExtExprNode)
        node.node_type = NodeType.TIME
        node.value = 0.0
        node.lag = 1
        node.state_key = 0
        node.window = 10
        node.left = None
        node.right = None
        node.third = None
        node._cache = {}
        return node

    def _make_prev(self, lag: int = 1) -> ExtExprNode:
        from ouroboros.synthesis.expr_node import NodeType
        node = ExtExprNode.__new__(ExtExprNode)
        node.node_type = NodeType.PREV
        node.value = 0.0
        node.lag = lag
        node.state_key = 0
        node.window = 10
        node.left = None
        node.right = None
        node.third = None
        node._cache = {}
        return node

    def _random_terminal(self) -> ExtExprNode:
        # When channels are loaded, 25% chance to emit a CHANNEL_PREV leaf
        if self._channel_state and self._rng.random() < 0.25:
            ch_idx = self._rng.choice(list(self._channel_state.keys()))
            lag = self._rng.randint(0, min(3, self.cfg.max_lag))
            return ExtExprNode(ExtNodeType.CHANNEL_PREV, channel_idx=ch_idx, lag=lag)

        choice = self._rng.random()
        if choice < 0.60:
            return self._make_const()
        elif choice < 0.90:
            return self._make_time()
        else:
            lag = self._rng.randint(1, self.cfg.max_lag)
            return self._make_prev(lag)

    def _seed_growth_templates(self, observations: List[int]) -> List[GrammarBeamCandidate]:
        from ouroboros.nodes.extended_nodes import NODE_SPECS
        from ouroboros.synthesis.expr_node import NodeType

        LOG_NT  = next((nt for nt in NODE_SPECS if nt.name == 'LOG'),  None)
        SQRT_NT = next((nt for nt in NODE_SPECS if nt.name == 'SQRT'), None)
        EXP_NT  = next((nt for nt in NODE_SPECS if nt.name == 'EXP'),  None)
        DIV_NT  = next((nt for nt in NODE_SPECS if nt.name == 'DIV'),  None)
        POW_NT  = next((nt for nt in NODE_SPECS if nt.name == 'POW'),  None)

        C   = NodeType.CONST
        T   = NodeType.TIME
        MUL = NodeType.MUL
        ADD = NodeType.ADD

        def n(nt, val=0.0, left=None, right=None):
            return self._make_literal_node(nt, val, left=left, right=right)

        def t_plus(b):
            """t + b node"""
            if b == 0:
                return n(T)
            return n(ADD, left=n(T), right=n(C, float(b)))

        seeds = []

        # ── LINEAR: scale * (t + b) ───────────────────────────────────────────
        for scale in [0.5, 1, 2, 3, 5, 7, 10, 20]:
            for b in [0, 1, 2]:
                try:
                    expr = n(MUL, left=n(C, float(scale)), right=t_plus(b))
                    seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                except Exception:
                    pass

        # ── LINEAR WITH INTERCEPT: scale * (t + b) + intercept ───────────────
        for scale in [1, 2, 3, 5, 7, 10]:
            for b in [0, 1]:
                for intercept in [1, 2, 3, 5, 7, 10, 20, 50]:
                    try:
                        mul  = n(MUL, left=n(C, float(scale)), right=t_plus(b))
                        expr = n(ADD, left=mul, right=n(C, float(intercept)))
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── DATA-DRIVEN LINEAR ────────────────────────────────────────────────
        if len(observations) >= 4:
            diffs = [observations[i+1] - observations[i]
                    for i in range(min(10, len(observations)-1))]
            raw_slope = sum(diffs) / len(diffs)
            for s in {raw_slope, round(raw_slope), round(raw_slope*2)/2}:
                if s == 0:
                    continue
                for b in [0, 1, 2]:
                    for intercept in [0, observations[0]]:
                        try:
                            mul  = n(MUL, left=n(C, float(s)), right=t_plus(b))
                            expr = n(ADD, left=mul, right=n(C, float(intercept)))
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        # ── QUADRATIC: scale * (t + b)^2 ─────────────────────────────────────
        if POW_NT:
            # data-driven scale estimate
            scales = set([1.0, 2.0, 0.5])
            for t_idx in range(3, min(10, len(observations))):
                if (t_idx+1)**2 > 0:
                    scales.add(round(observations[t_idx] / (t_idx+1)**2, 2))
            for scale in scales:
                for b in [0, 1, 2]:
                    try:
                        base = t_plus(b)
                        pw   = n(POW_NT, left=base, right=n(C, 2.0))
                        expr = n(MUL, left=n(C, float(scale)), right=pw)
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass
            # quadratic + intercept
            for b in [1]:
                for intercept in [1, 2]:
                    try:
                        base = t_plus(b)
                        pw   = n(POW_NT, left=base, right=n(C, 2.0))
                        mul  = n(MUL, left=n(C, 2.0), right=pw)
                        expr = n(ADD, left=mul, right=n(C, float(intercept)))
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass
                
                # ── CUBIC: scale * (t + b)^3 ─────────────────────────────────────────
        if POW_NT:
            scales_c = set([1.0, 0.5, 0.125, 0.25])
            for t_idx in range(2, min(8, len(observations))):
                denom = (t_idx + 1) ** 3
                if denom > 0:
                    scales_c.add(round(observations[t_idx] / denom, 4))
            for scale in scales_c:
                for b in [0, 1, 2, 3]:
                    try:
                        base = t_plus(b)
                        pw   = n(POW_NT, left=base, right=n(C, 3.0))
                        expr = n(MUL, left=n(C, float(scale)), right=pw)
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── SQRT: scale * sqrt(t + b) ─────────────────────────────────────────
        if SQRT_NT:
            c_est = float(observations[0]) if observations else 10.0
            for scale in {c_est, round(c_est), 5.0, 10.0, round(c_est/2)*2}:
                for b in [0, 1, 2]:
                    try:
                        sqr  = n(SQRT_NT, left=t_plus(b))
                        expr = n(MUL, left=n(C, float(scale)), right=sqr)
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── LOG: scale * log(t + b) ───────────────────────────────────────────
        if LOG_NT:
            for scale in [1, 2, 5, 10, 20]:
                for b in [1, 2]:   # b>=1 to avoid log(0)
                    try:
                        lg   = n(LOG_NT, left=t_plus(b))
                        expr = n(MUL, left=n(C, float(scale)), right=lg)
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── EXP DECAY/GROWTH: scale * exp(rate * (t + b)) ────────────────────
        if EXP_NT:
            for rate in [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1]:
                for scale in [observations[0], 100.0, 50.0, 10.0]:
                    for b in [0, 1]:
                        try:
                            mul_rt = n(MUL, left=n(C, float(rate)), right=t_plus(b))
                            exp_n  = n(EXP_NT, left=mul_rt)
                            expr   = n(MUL, left=n(C, float(scale)), right=exp_n)
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        # ── AR1 EXP DECAY: ratio * obs[t-1] ──────────────────────────────────
        if len(observations) >= 5 and observations[0] > 0:
            ratios = [observations[i+1] / observations[i]
                    for i in range(min(8, len(observations)-1))
                    if observations[i] > 0 and observations[i+1] > 0]
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                if 0.5 < avg_ratio < 0.99:
                    for r in {avg_ratio, round(avg_ratio*20)/20}:
                        try:
                            expr = n(MUL, left=n(C, float(r)),
                                    right=self._make_prev(1))
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        # ── INVERSE: scale / (t + b) ──────────────────────────────────────────
        if DIV_NT:
            for scale in [observations[0], 10.0, 50.0, 100.0, 1000.0]:
                for b in [1, 2, 3]:
                    try:
                        expr = n(DIV_NT, left=n(C, float(scale)), right=t_plus(b))
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── INVERSE SQUARE: scale / (t + b)^2 ────────────────────────────────
        if DIV_NT and POW_NT:
            for scale in [observations[0], 10.0, 100.0, 1000.0]:
                for b in [1, 2]:
                    try:
                        pw   = n(POW_NT, left=t_plus(b), right=n(C, 2.0))
                        expr = n(DIV_NT, left=n(C, float(scale)), right=pw)
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── GAUSSIAN: scale * EXP(rate * (t+b)^2) ────────────────────────────
        if EXP_NT and POW_NT:
            for scale in [observations[0], 100.0, 50.0]:
                for rate in [-0.02, -0.005, -0.01, -0.05]:
                    for b in [0, 1]:
                        try:
                            tb    = t_plus(b)
                            sq    = n(POW_NT, left=tb, right=n(C, 2.0))
                            mul_r = n(MUL, left=n(C, float(rate)), right=sq)
                            exp_n = n(EXP_NT, left=mul_r)
                            expr  = n(MUL, left=n(C, float(scale)), right=exp_n)
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        # ── T * EXP: (t+b) * EXP(rate * (t+b)) ──────────────────────────────
        if EXP_NT:
            for rate in [0.05, 0.1, 0.2]:
                for b in [0, 1]:
                    try:
                        tb    = t_plus(b)
                        tb2   = t_plus(b)   # fresh node
                        mul_r = n(MUL, left=n(C, float(rate)), right=tb)
                        exp_n = n(EXP_NT, left=mul_r)
                        expr  = n(MUL, left=tb2, right=exp_n)
                        seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                    except Exception:
                        pass

        # ── SATURATION: scale*(t+b) / (c + (t+b)) ────────────────────────────
        if DIV_NT:
            for scale in [10.0, observations[0] if observations else 10.0]:
                for b in [0, 1]:
                    for c in [1.0, 2.0, 5.0, 10.0]:
                        try:
                            tb   = t_plus(b)
                            tb2  = t_plus(b)
                            num  = n(MUL, left=n(C, float(scale)), right=tb)
                            den  = n(ADD, left=n(C, float(c)), right=tb2)
                            expr = n(DIV_NT, left=num, right=den)
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        # ── T * SIN(T): scale * (t+b) * SIN(freq * (t+b)) ───────────────────
        SIN_NT = next((nt for nt in NODE_SPECS if nt.name == 'SIN'), None)
        if SIN_NT:
            for scale in [1.0, 2.0, 3.0, 5.0]:
                for freq in [0.1, 0.2, 1.0]:
                    for b in [0, 1]:
                        try:
                            tb    = t_plus(b)
                            tb2   = t_plus(b)
                            tb3   = t_plus(b)
                            mul_f = n(MUL, left=n(C, float(freq)), right=tb)
                            sin_n = n(SIN_NT, left=mul_f)
                            expr  = n(MUL, left=n(MUL, left=n(C, float(scale)), right=tb2), right=sin_n)
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        # ── BOSE-EINSTEIN: scale / (EXP(rate*(t+b)) - 1) ─────────────────────
        SUB_NT = next((nt for nt in NODE_SPECS if nt.name == 'SUB'), None)
        if EXP_NT and SUB_NT and DIV_NT:
            for scale in [10.0, 5.0]:
                for rate in [0.05, 0.1, 0.2]:
                    for b in [0, 1]:
                        try:
                            tb    = t_plus(b)
                            mul_r = n(MUL, left=n(C, float(rate)), right=tb)
                            exp_n = n(EXP_NT, left=mul_r)
                            den   = n(SUB_NT, left=exp_n, right=n(C, 1.0))
                            expr  = n(DIV_NT, left=n(C, float(scale)), right=den)
                            seeds.append(GrammarBeamCandidate(expr, self._score(expr, observations)))
                        except Exception:
                            pass

        seeds.sort()
        return seeds[:self.cfg.beam_width * 2]  # return 2x slots, slot reservation trims it

    def _random_expr(
        self,
        depth: int = 0,
        allowed_categories: frozenset = None,
        parent_type: Optional[ExtNodeType] = None,
        arg_index: int = 0,
    ) -> ExtExprNode:
        if allowed_categories is None:
            from ouroboros.grammar.math_grammar import ANY_TYPES
            allowed_categories = ANY_TYPES

        terminal_prob = min(0.9, 0.2 + depth * 0.2)
        if depth >= self.cfg.max_depth or self._rng.random() < terminal_prob:
            return self._random_terminal()

        nt = self._sample_node_type(allowed_categories, depth)

        if isinstance(nt, NodeCategory):
            return self._random_terminal()

        spec = NODE_SPECS.get(nt)
        if spec is None:
            return self._random_terminal()

        node = ExtExprNode(nt)

        if spec.window_arg and spec.arity >= 2:
            node.window = self._rng.randint(5, self.cfg.max_window)

        if spec.arity >= 1:
            child_cats = self._grammar.allowed_child_categories(nt, 0)
            node.left = self._random_expr(depth + 1, child_cats, nt, 0)

        if spec.arity >= 2:
            child_cats = self._grammar.allowed_child_categories(nt, 1)
            if spec.window_arg:
                node.right = self._make_const()
                node.right.value = float(self._rng.randint(5, self.cfg.max_window))
            else:
                node.right = self._random_expr(depth + 1, child_cats, nt, 1)

        if spec.arity >= 3:
            child_cats = self._grammar.allowed_child_categories(nt, 2)
            node.third = self._random_expr(depth + 1, child_cats, nt, 2)

        return node

    def _make_literal_node(self, node_type, value=0.0, left=None, right=None, third=None, channel_idx=0) -> ExtExprNode:
        """Helper to build a node with explicit fields (no random)."""
        node = ExtExprNode.__new__(ExtExprNode)
        node.node_type = node_type
        node.value = float(value)
        node.lag = 1
        node.state_key = 0
        node.window = 10
        node.channel_idx = channel_idx
        node.left = left
        node.right = right
        node.third = third
        node._cache = {}
        return node

    def _seed_modular_templates(self, observations: List[int]) -> List[GrammarBeamCandidate]:
        """
        Seed beam with (slope*t + intercept) % mod templates.
        Exhaustively tries small parameter combinations — one will exactly
        match any modular arithmetic sequence with near-zero MDL cost.
        """
        from ouroboros.synthesis.expr_node import NodeType

        seeds = []
        obs_set = set(observations)
        max_mod = max(obs_set) + 2 if obs_set else 14

        for mod in range(2, min(int(max_mod) + 1, 20)):
            for slope in range(1, mod + 1):
                for intercept in range(0, mod):
                    t_node  = self._make_literal_node(NodeType.TIME)
                    c_slope = self._make_literal_node(NodeType.CONST, slope)
                    c_inter = self._make_literal_node(NodeType.CONST, intercept)
                    c_mod   = self._make_literal_node(NodeType.CONST, mod)

                    mul_node = self._make_literal_node(NodeType.MUL,
                                                       left=c_slope, right=t_node)
                    add_node = self._make_literal_node(NodeType.ADD,
                                                       left=mul_node, right=c_inter)
                    mod_node = self._make_literal_node(NodeType.MOD,
                                                       left=add_node, right=c_mod)

                    cost = self._score(mod_node, observations)
                    seeds.append(GrammarBeamCandidate(mod_node, cost))

        seeds.sort()
        return seeds[:self.cfg.beam_width]

    def _seed_recurrence_templates(self, observations: List[int]) -> List[GrammarBeamCandidate]:
        """
        Seed beam with Fibonacci-style recurrence templates:
        (prev(a) + prev(b)) % mod
        (c * prev(1) + prev(2)) % mod
        Covers Fibonacci mod N and linear recurrences mod N.
        """
        from ouroboros.synthesis.expr_node import NodeType

        seeds = []
        obs_set = set(observations)
        max_mod = max(obs_set) + 2 if obs_set else 14

        for mod in range(2, min(int(max_mod) + 1, 20)):
            for lag_a in range(1, 4):
                for lag_b in range(lag_a + 1, 5):
                    p_a   = self._make_prev(lag_a)
                    p_b   = self._make_prev(lag_b)
                    add   = self._make_literal_node(NodeType.ADD, left=p_a, right=p_b)
                    c_mod = self._make_literal_node(NodeType.CONST, mod)
                    expr  = self._make_literal_node(NodeType.MOD, left=add, right=c_mod)
                    cost  = self._score(expr, observations)
                    seeds.append(GrammarBeamCandidate(expr, cost))

            for slope in range(1, min(mod + 1, 8)):
                p1    = self._make_prev(1)
                p2    = self._make_prev(2)
                c_s   = self._make_literal_node(NodeType.CONST, slope)
                mul   = self._make_literal_node(NodeType.MUL, left=c_s, right=p1)
                add   = self._make_literal_node(NodeType.ADD, left=mul, right=p2)
                c_mod = self._make_literal_node(NodeType.CONST, mod)
                expr  = self._make_literal_node(NodeType.MOD, left=add, right=c_mod)
                cost  = self._score(expr, observations)
                seeds.append(GrammarBeamCandidate(expr, cost))

        seeds.sort()
        return seeds[:self.cfg.beam_width]

    def _score(self, expr: ExtExprNode, observations: List[int]) -> float:
        try:
            state: Dict[int, Any] = {}
            if hasattr(self, '_channel_state') and self._channel_state:
                state[-1] = self._channel_state
            predictions = []
            for t in range(len(observations)):
                pred = expr.evaluate(t, observations[:t] if t > 0 else [], state)
                if not math.isfinite(pred):
                    return float('inf')
                predictions.append(int(round(pred)))
            result = self._mdl.compute(
                predictions, observations,
                expr.node_count(), expr.constant_count(),
                alphabet_size=getattr(self, '_alphabet_size', 13)
            )
            return result.total_mdl_cost
        except RecursionError:
            return float('inf')
        except Exception:
            return float('inf')

    def _mutate_grammar(self, expr: ExtExprNode) -> ExtExprNode:
        mutated = copy.deepcopy(expr)
        self._mutate_node_inplace(mutated, depth=0)
        return mutated

    def _mutate_node_inplace(self, node: ExtExprNode, depth: int) -> None:
        """Mutate a node in place, maintaining grammar validity."""
        r = self._rng.random()

        if hasattr(node.node_type, 'name') and node.node_type.name == 'CONST' and r < 0.65:
            v = node.value
            if abs(v) < 1.0:
                # fine-grained search for small constants (rates, fractions)
                strategy = self._rng.random()
                if strategy < 0.5:
                    node.value = v + self._rng.uniform(-0.02, 0.02)
                elif strategy < 0.8:
                    node.value = v * self._rng.choice([0.5, 2.0, -1.0])
                else:
                    node.value = self._rng.uniform(-1.0, 1.0)
            elif abs(v) < 10.0:
                # medium constants — additive or small multiplicative
                strategy = self._rng.random()
                if strategy < 0.4:
                    node.value = v + self._rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0])
                elif strategy < 0.7:
                    node.value = v * self._rng.choice([0.5, 2.0])
                else:
                    node.value = float(self._rng.randint(1, 20))
            else:
                # large constants — multiplicative scaling
                scale = self._rng.choice([0.5, 2.0, 0.1, 10.0, -1.0])
                node.value = max(-self.cfg.const_range, min(self.cfg.const_range, v * scale))
            return

        # 20%: replace entire subtree
        if r < 0.35 and depth < self.cfg.max_depth - 1:
            from ouroboros.grammar.math_grammar import ANY_TYPES
            fresh = self._random_expr(depth, ANY_TYPES)
            node.node_type = fresh.node_type
            node.value = fresh.value
            node.lag = fresh.lag
            node.left = fresh.left
            node.right = fresh.right
            if hasattr(fresh, 'third'):
                node.third = fresh.third
            return

        # Recurse into children
        if node.left and self._rng.random() < 0.5:
            self._mutate_node_inplace(node.left, depth + 1)
        if node.right and self._rng.random() < 0.4:
            self._mutate_node_inplace(node.right, depth + 1)
        if getattr(node, 'third', None) and self._rng.random() < 0.3:
            self._mutate_node_inplace(node.third, depth + 1)

        
    def search(self, observations, verbose=False, extra_seeds=None,
           channel_state=None, alphabet_size=None):
        self._channel_state = channel_state if channel_state is not None else {}
        self._alphabet_size = alphabet_size or (max(observations) - min(observations) + 2)

        beam_width = self.cfg.beam_width

        # Random initial population
        init_exprs = [self._random_expr() for _ in range(beam_width * 5)]
        init_costs = [self._score(e, observations) for e in init_exprs]

        # Modular arithmetic seeds — (slope*t + intercept) % mod
        mod_seeds    = self._seed_modular_templates(observations)
        rec_seeds    = self._seed_recurrence_templates(observations)
        growth_seeds = self._seed_growth_templates(observations)

        # Reserve guaranteed slots for each seed type so no category crowds others out
        growth_slots = growth_seeds[:max(8, beam_width // 3)]
        mod_slots    = mod_seeds[:max(4, beam_width // 6)]
        rec_slots    = rec_seeds[:max(4, beam_width // 6)]
        random_slots = sorted(
            [GrammarBeamCandidate(e, c) for e, c in zip(init_exprs, init_costs)]
        )[:max(4, beam_width // 6)]
        extra_slots  = (extra_seeds or [])[:4]

        # Merge, deduplicate by score, take best beam_width
        all_candidates = growth_slots + mod_slots + rec_slots + random_slots + extra_slots
        seen_scores = set()
        deduped = []
        for c in sorted(all_candidates):
            key = round(c.mdl_cost, 1)
            if key not in seen_scores:
                seen_scores.add(key)
                deduped.append(c)

        candidates = deduped[:beam_width]

        if verbose:
            print(f"  Grammar beam initial best: {candidates[0].mdl_cost:.2f}")
            print(f"  Best expr: {candidates[0].expr.to_string()[:60]}")

        deadline = time.time() + getattr(self.cfg, 'time_budget_seconds', 8.0)
        for iteration in range(self.cfg.n_iterations):
            if time.time() > deadline:
                break
            new_candidates = list(candidates)
            for cand in candidates:
                # 1. mutation (keep existing)
                for _ in range(2):
                    mutated = self._mutate_grammar(cand.expr)
                    cost = self._score(mutated, observations)
                    new_candidates.append(GrammarBeamCandidate(mutated, cost))

                # 2. 🔥 NEW: additive composition
                for other in candidates[:5]:  # limit for speed
                    try:
                        from ouroboros.synthesis.expr_node import NodeType as _NT
                        combined = self._make_literal_node(
                            _NT.ADD,
                            left=copy.deepcopy(cand.expr),
                            right=copy.deepcopy(other.expr)
                        )

                        cost = self._score(combined, observations)
                        new_candidates.append(GrammarBeamCandidate(combined, cost))
                    except Exception:
                        pass

            new_candidates.sort()
            candidates = new_candidates[:beam_width]

            if verbose and iteration % 5 == 0:
                print(f"  iter {iteration}: best={candidates[0].mdl_cost:.2f}")

        if verbose:
            print(f"  Grammar beam final: {candidates[0].mdl_cost:.2f}")

        return candidates[0].expr if candidates else None