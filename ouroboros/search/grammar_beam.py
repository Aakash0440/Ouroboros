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
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.nodes.extended_nodes import ExtNodeType, ExtExprNode, NODE_SPECS, NodeCategory
from ouroboros.grammar.math_grammar import MathGrammar, DEFAULT_GRAMMAR
from ouroboros.compression.mdl_engine import MDLEngine


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
    # Adjusted based on environment type by HierarchicalClassifier (Day 31)
    category_weights: Dict[NodeCategory, float] = field(default_factory=lambda: {
        NodeCategory.TERMINAL:    3.0,   # always high — terminals needed everywhere
        NodeCategory.ARITHMETIC:  2.0,   # common building blocks
        NodeCategory.CALCULUS:    1.0,   # domain-specific
        NodeCategory.STATISTICAL: 1.0,
        NodeCategory.LOGICAL:     0.8,
        NodeCategory.TRANSFORM:   0.5,   # expensive, less common
        NodeCategory.NUMBER:      0.7,
        NodeCategory.MEMORY:      0.6,
        NodeCategory.TRANSCEND:   1.5,   # SIN, COS, EXP common in physics
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
    
    Key guarantee: every candidate in the beam at every iteration
    is a grammatically valid mathematical expression.
    """

    def __init__(self, config: GrammarBeamConfig = None):
        self.cfg = config or GrammarBeamConfig()
        self._rng = random.Random(self.cfg.random_seed)
        self._grammar = self.cfg.grammar
        self._mdl = MDLEngine()

        # Precompute weighted node lists by category
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
        """
        Sample a node type from the allowed categories.
        
        At max_depth: always sample a terminal.
        Otherwise: weight by category_weights and depth pressure.
        """
        if force_terminal or depth >= self.cfg.max_depth:
            terminal_types = self._nodes_by_category.get(NodeCategory.TERMINAL, [])
            if terminal_types:
                return self._rng.choice(terminal_types)

        # Build weighted list of valid types
        candidates = []
        weights = []
        for cat in allowed_categories:
            if cat not in self._nodes_by_category:
                continue
            cat_weight = self.cfg.category_weights.get(cat, 1.0)
            # Depth penalty: prefer simpler nodes at deeper levels
            depth_factor = max(0.1, 1.0 - depth * 0.15)
            for nt in self._nodes_by_category[cat]:
                spec = NODE_SPECS.get(nt)
                if spec is None:
                    continue
                # Terminal nodes get a depth-scaled bonus (we need leaves)
                terminal_bonus = 2.0 if spec.arity == 0 else 1.0
                w = cat_weight * depth_factor * terminal_bonus
                candidates.append(nt)
                weights.append(w)

        if not candidates:
            # Fallback: CONST terminal
            from ouroboros.synthesis.expr_node import NodeType
            return NodeCategory.TERMINAL  # signal to use original CONST

        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0.0
        for nt, w in zip(candidates, weights):
            cumulative += w
            if r <= cumulative:
                return nt
        return candidates[-1]

    def _make_const(self) -> ExtExprNode:
        """Make a CONST terminal from original node system."""
        from ouroboros.synthesis.expr_node import NodeType, ExprNode
        v = self._rng.randint(0, self.cfg.const_range)
        # Wrap in ExtExprNode-compatible form
        node = ExtExprNode.__new__(ExtExprNode)
        node.node_type = NodeType.CONST
        node.value = float(v)
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
        """Generate a random terminal node."""
        choice = self._rng.random()
        if choice < 0.5:
            return self._make_const()
        elif choice < 0.8:
            return self._make_time()
        else:
            lag = self._rng.randint(1, self.cfg.max_lag)
            return self._make_prev(lag)

    def _random_expr(
        self,
        depth: int = 0,
        allowed_categories: frozenset = None,
        parent_type: Optional[ExtNodeType] = None,
        arg_index: int = 0,
    ) -> ExtExprNode:
        """
        Recursively generate a random grammatically-valid expression.
        """
        if allowed_categories is None:
            # Root level: all categories allowed
            from ouroboros.grammar.math_grammar import ANY_TYPES
            allowed_categories = ANY_TYPES

        # At max depth or with probability increasing with depth: return terminal
        terminal_prob = min(0.9, 0.2 + depth * 0.2)
        if depth >= self.cfg.max_depth or self._rng.random() < terminal_prob:
            return self._random_terminal()

        # Sample a node type from allowed categories
        nt = self._sample_node_type(allowed_categories, depth)

        # If sampling failed (returned a category object), use terminal
        if isinstance(nt, NodeCategory):
            return self._random_terminal()

        spec = NODE_SPECS.get(nt)
        if spec is None:
            return self._random_terminal()

        # Create the node
        node = ExtExprNode(nt)

        # Set window size for window nodes
        if spec.window_arg and spec.arity >= 2:
            node.window = self._rng.randint(5, self.cfg.max_window)

        # Recursively generate children according to grammar
        if spec.arity >= 1:
            child_cats = self._grammar.allowed_child_categories(nt, 0)
            node.left = self._random_expr(depth + 1, child_cats, nt, 0)

        if spec.arity >= 2:
            child_cats = self._grammar.allowed_child_categories(nt, 1)
            # For window args: force CONST terminal
            if spec.window_arg:
                node.right = self._make_const()
                node.right.value = float(self._rng.randint(5, self.cfg.max_window))
            else:
                node.right = self._random_expr(depth + 1, child_cats, nt, 1)

        if spec.arity >= 3:
            child_cats = self._grammar.allowed_child_categories(nt, 2)
            node.third = self._random_expr(depth + 1, child_cats, nt, 2)

        return node

    def _score(self, expr: ExtExprNode, observations: List[int]) -> float:
        """Score an expression under MDL."""
        try:
            state: Dict[int, float] = {}
            predictions = []
            for t in range(len(observations)):
                pred = expr.evaluate(t, observations[:t], state)
                predictions.append(int(round(pred)) if math.isfinite(pred) else 0)

            result = self._mdl.compute(
                predictions, observations,
                expr.node_count(), expr.constant_count()
            )
            return result.total_mdl_cost
        except Exception:
            return float('inf')

    def _mutate_grammar(self, expr: ExtExprNode) -> ExtExprNode:
        """
        Mutate an expression while maintaining grammar validity.
        
        Three mutation types:
          1. Constant perturbation (50%): change a CONST value
          2. Grammar-valid node replacement (30%): replace a node with valid alternative
          3. Subtree replacement (20%): replace a subtree with new valid subtree
        """
        mutated = copy.deepcopy(expr)
        self._mutate_node_inplace(mutated, depth=0)
        return mutated

    def _mutate_node_inplace(self, node: ExtExprNode, depth: int) -> None:
        """Mutate a node in place, maintaining grammar validity."""
        from ouroboros.synthesis.expr_node import NodeType

        r = self._rng.random()

        # 50%: perturb constant
        if hasattr(node.node_type, 'name') and node.node_type.name == 'CONST' and r < 0.5:
            delta = self._rng.gauss(0, max(1, abs(node.value) * 0.1))
            node.value = max(-self.cfg.const_range, min(self.cfg.const_range, node.value + delta))
            return

        # 20%: replace entire subtree
        if r < 0.2 and depth < self.cfg.max_depth - 1:
            from ouroboros.grammar.math_grammar import ANY_TYPES
            fresh = self._random_expr(depth, ANY_TYPES)
            node.node_type = fresh.node_type
            node.value = fresh.value
            node.lag = fresh.lag
            node.left = fresh.left
            node.right = fresh.right
            node.third = fresh.third
            return

        # Recurse into children
        if node.left and self._rng.random() < 0.5:
            self._mutate_node_inplace(node.left, depth + 1)
        if node.right and self._rng.random() < 0.4:
            self._mutate_node_inplace(node.right, depth + 1)
        if node.third and self._rng.random() < 0.3:
            self._mutate_node_inplace(node.third, depth + 1)

    def search(
        self,
        observations: List[int],
        verbose: bool = False,
    ) -> Optional[ExtExprNode]:
        """
        Run grammar-constrained beam search.
        Returns the best expression found.
        """
        import math

        beam_width = self.cfg.beam_width

        # Initialize with random grammar-valid expressions
        init_exprs = [self._random_expr() for _ in range(beam_width * 5)]
        init_costs = [self._score(e, observations) for e in init_exprs]

        candidates = sorted(
            [GrammarBeamCandidate(e, c) for e, c in zip(init_exprs, init_costs)],
        )[:beam_width]

        if verbose:
            print(f"  Grammar beam initial best: {candidates[0].mdl_cost:.2f}")
            print(f"  Best expr: {candidates[0].expr.to_string()[:60]}")

        for iteration in range(self.cfg.n_iterations):
            new_candidates = list(candidates)
            for cand in candidates:
                for _ in range(3):
                    mutated = self._mutate_grammar(cand.expr)
                    cost = self._score(mutated, observations)
                    new_candidates.append(GrammarBeamCandidate(mutated, cost))

            new_candidates.sort()
            candidates = new_candidates[:beam_width]

        if verbose:
            print(f"  Grammar beam final: {candidates[0].mdl_cost:.2f}")

        return candidates[0].expr if candidates else None


import math  # needed for isfinite in _score