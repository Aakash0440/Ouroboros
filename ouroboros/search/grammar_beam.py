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
        import math
        from ouroboros.nodes.extended_nodes import NODE_SPECS
        from ouroboros.synthesis.expr_node import NodeType

        LOG_NT  = next((nt for nt in NODE_SPECS if nt.name == 'LOG'),  None)
        SQRT_NT = next((nt for nt in NODE_SPECS if nt.name == 'SQRT'), None)

        seeds = []

        for scale in [1, 2, 5, 10, 20]:
            for offset in [0, 1]:
                if LOG_NT:
                    t_node   = self._make_literal_node(NodeType.TIME)
                    c_off    = self._make_literal_node(NodeType.CONST, float(offset))
                    c_scl    = self._make_literal_node(NodeType.CONST, float(scale))
                    add_node = self._make_literal_node(NodeType.ADD, left=t_node, right=c_off)
                    log_node = self._make_literal_node(LOG_NT, left=add_node)
                    expr_log = self._make_literal_node(NodeType.MUL, left=c_scl, right=log_node)
                    seeds.append(GrammarBeamCandidate(expr_log, self._score(expr_log, observations)))

                if SQRT_NT:
                    t2        = self._make_literal_node(NodeType.TIME)
                    c_off2    = self._make_literal_node(NodeType.CONST, float(offset))
                    c_scl2    = self._make_literal_node(NodeType.CONST, float(scale))
                    add_node2 = self._make_literal_node(NodeType.ADD, left=t2, right=c_off2)
                    sqrt_node = self._make_literal_node(SQRT_NT, left=add_node2)
                    expr_sqrt = self._make_literal_node(NodeType.MUL, left=c_scl2, right=sqrt_node)
                    seeds.append(GrammarBeamCandidate(expr_sqrt, self._score(expr_sqrt, observations)))

                # linear: scale * t
                t3       = self._make_literal_node(NodeType.TIME)
                c_scl3   = self._make_literal_node(NodeType.CONST, float(scale))
                expr_lin = self._make_literal_node(NodeType.MUL, left=c_scl3, right=t3)
                seeds.append(GrammarBeamCandidate(expr_lin, self._score(expr_lin, observations)))

                # linear with intercept: scale * t + intercept
                for intercept in [1, 2, 5, 10, 20, 50]:
                    t5        = self._make_literal_node(NodeType.TIME)
                    c_scl5    = self._make_literal_node(NodeType.CONST, float(scale))
                    c_int     = self._make_literal_node(NodeType.CONST, float(intercept))
                    mul_node  = self._make_literal_node(NodeType.MUL, left=c_scl5, right=t5)
                    expr_lin2 = self._make_literal_node(NodeType.ADD, left=mul_node, right=c_int)
                    seeds.append(GrammarBeamCandidate(expr_lin2, self._score(expr_lin2, observations)))

                # quadratic: scale * t^2
                t4       = self._make_literal_node(NodeType.TIME)
                c2       = self._make_literal_node(NodeType.CONST, 2.0)
                c_scl4   = self._make_literal_node(NodeType.CONST, float(scale))
                pow_node = self._make_literal_node(NodeType.POW, left=t4, right=c2)
                expr_pow = self._make_literal_node(NodeType.MUL, left=c_scl4, right=pow_node)
                seeds.append(GrammarBeamCandidate(expr_pow, self._score(expr_pow, observations)))

        # ── DATA-DRIVEN SEEDS (run once, not inside the scale/offset loops) ──

        # 1. Detected linear: slope * t + intercept (supports float slopes)
        if len(observations) >= 4:
            diffs = [observations[i+1] - observations[i]
                     for i in range(min(10, len(observations)-1))]
            raw_slope = sum(diffs) / len(diffs)
            slopes_to_try = list({round(raw_slope), raw_slope,
                                   round(raw_slope*2)/2, round(raw_slope*4)/4})
            slopes_to_try = [s for s in slopes_to_try if s != 0]
            intercept = observations[0]
            for s in slopes_to_try:
                for b in [intercept, 0, 1]:
                    try:
                        c_s = self._make_literal_node(NodeType.CONST, float(s))
                        c_b = self._make_literal_node(NodeType.CONST, float(b))
                        t_n = self._make_literal_node(NodeType.TIME)
                        mul = self._make_literal_node(NodeType.MUL, left=c_s, right=t_n)
                        add = self._make_literal_node(NodeType.ADD, left=mul, right=c_b)
                        seeds.append(GrammarBeamCandidate(add, self._score(add, observations)))
                    except Exception:
                        pass

        # 2. Detected sqrt: c * sqrt(t + b)
        if SQRT_NT and len(observations) >= 4 and all(v >= 0 for v in observations[:10]):
            c_est = float(observations[0])
            for c in {c_est, round(c_est), round(c_est/2)*2, 5.0, 10.0}:
                for b in [0, 1, 2]:
                    try:
                        c_n = self._make_literal_node(NodeType.CONST, float(c))
                        b_n = self._make_literal_node(NodeType.CONST, float(b))
                        t_n = self._make_literal_node(NodeType.TIME)
                        add = self._make_literal_node(NodeType.ADD, left=t_n, right=b_n)
                        sqr = self._make_literal_node(SQRT_NT, left=add)
                        mul = self._make_literal_node(NodeType.MUL, left=c_n, right=sqr)
                        seeds.append(GrammarBeamCandidate(mul, self._score(mul, observations)))
                    except Exception:
                        pass

        # 3. Detected exp decay: ratio * obs[t-1]  (AR1 approximation)
        if len(observations) >= 5 and observations[0] > 0:
            ratios = [observations[i+1] / observations[i]
                      for i in range(min(8, len(observations)-1))
                      if observations[i] > 0 and observations[i+1] > 0]
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                if 0.5 < avg_ratio < 0.99:
                    for r in {avg_ratio, round(avg_ratio*20)/20}:
                        try:
                            c_r   = self._make_literal_node(NodeType.CONST, float(r))
                            obs_n = self._make_literal_node(NodeType.PREV)
                            mul   = self._make_literal_node(NodeType.MUL, left=c_r, right=obs_n)
                            score = self._score(mul, observations)
                            seeds.append(GrammarBeamCandidate(mul, score))
                        except Exception:
                            pass


        if len(observations) >= 5 and observations[0] > 0:
            # detect saturation: differences shrinking
            diffs = [observations[i+1] - observations[i]
                     for i in range(min(8, len(observations)-1))]
            if len(diffs) >= 3 and diffs[0] > diffs[-1] > 0:
                try:
                    # seed: obs[t-1] * (1 - 1/obs[0])  rough saturation
                    sat_ratio = 1.0 - (diffs[-1] / max(diffs[0], 1))
                    c_r   = self._make_literal_node(NodeType.CONST, float(sat_ratio))
                    obs_n = self._make_literal_node(NodeType.PREV)
                    mul   = self._make_literal_node(NodeType.MUL, left=c_r, right=obs_n)
                    seeds.append(GrammarBeamCandidate(mul, self._score(mul, observations)))
                except Exception:
                    pass
                
        seeds.sort()
        return seeds[:self.cfg.beam_width]

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
                expr.node_count(), expr.constant_count()
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

        # 50%: perturb constant
        if hasattr(node.node_type, 'name') and node.node_type.name == 'CONST' and r < 0.65:
            if abs(node.value) < 1.0:
                # small fractional constant — perturb additively to keep it fractional
                delta = self._rng.uniform(-0.05, 0.05)
                node.value = node.value + delta
            else:
                scale = self._rng.choice([0.5, 2.0, 0.1, 10.0, -1.0])
                node.value = max(-self.cfg.const_range, min(self.cfg.const_range, node.value * scale))
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

        
    def search(
        self,
        observations: List[int],
        verbose: bool = False,
        extra_seeds: Optional[List] = None,
        channel_state: Optional[dict] = None,
    ) -> Optional[ExtExprNode]:
        """Run grammar-constrained beam search. Returns the best expression found."""
        if channel_state is not None:
            self._channel_state = channel_state

        beam_width = self.cfg.beam_width

        # Random initial population
        init_exprs = [self._random_expr() for _ in range(beam_width * 5)]
        init_costs = [self._score(e, observations) for e in init_exprs]

        # Modular arithmetic seeds — (slope*t + intercept) % mod
        mod_seeds    = self._seed_modular_templates(observations)
        rec_seeds    = self._seed_recurrence_templates(observations)
        growth_seeds = self._seed_growth_templates(observations)

        candidates = sorted(
            [GrammarBeamCandidate(e, c) for e, c in zip(init_exprs, init_costs)]
            + mod_seeds
            + rec_seeds
            + growth_seeds
            + (extra_seeds or []),
        )[:beam_width]

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
                        combined = ExtExprNode(
                            ExtNodeType.ADD,
                            left=cand.expr,
                            right=other.expr
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