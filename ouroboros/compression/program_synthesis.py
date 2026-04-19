"""
Symbolic program synthesis for OUROBOROS agents.

Extended node types (v2):
    CONST(n)       â€” integer constant n
    TIME           â€” timestep variable t
    ADD(l, r)      â€” l + r
    MUL(l, r)      â€” l * r
    MOD(l, r)      â€” l mod r  (r=0 â†’ 0)
    SUB(l, r)      â€” l - r    (NEW: enables negative residues)
    DIV(l, r)      â€” l // r   (NEW: integer floor division, r=0 â†’ 0)
    POW(l, r)      â€” l ** r   (NEW: polynomial sequences, r clamped â‰¤5)
    PREV(k)        â€” obs[t-k] (NEW: recurrence relations â€” k is lag)
    IF(c, t, e)    â€” câ‰ 0 ? t : e (NEW: piecewise rules, 3-child node)
    EQ(l, r)       â€” 1 if l==r else 0 (NEW: equality test, useful in IF)
    LT(l, r)       â€” 1 if l<r  else 0 (NEW: less-than test)

Why these and not more?
    These cover: modular arithmetic (MOD), recurrence (PREV),
    polynomial (POW), piecewise (IF+EQ/LT), and basic arithmetic.
    Together they express the vast majority of simple mathematical
    sequences. Adding more primitives beyond this set increases
    search space without proportionally increasing expressiveness.

PREV semantics:
    PREV(k) evaluates to observation_history[t-k].
    If t-k < 0, returns 0 (boundary condition).
    The observation_history is passed as context to evaluate().
    For prediction (no history), PREV uses the expression's own
    previous predictions â€” this is the recurrence mode.

IF semantics:
    IF has THREE children: condition, then_branch, else_branch.
    evaluate() checks if condition != 0, returns then or else.
    This requires a special 3-child node structure.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


class NodeType(Enum):
    # Original
    CONST = auto()
    TIME  = auto()
    ADD   = auto()
    MUL   = auto()
    MOD   = auto()
    # New arithmetic
    SUB   = auto()
    DIV   = auto()
    POW   = auto()
    # New structural
    PREV  = auto()
    IF    = auto()
    EQ    = auto()
    LT    = auto()


# Node category helpers
LEAF_TYPES    = {NodeType.CONST, NodeType.TIME, NodeType.PREV}
BINARY_TYPES  = {NodeType.ADD, NodeType.MUL, NodeType.MOD,
                 NodeType.SUB, NodeType.DIV, NodeType.POW,
                 NodeType.EQ, NodeType.LT}
TERNARY_TYPES = {NodeType.IF}


@dataclass
class ExprNode:
    """
    Node in a symbolic expression tree.

    Leaf:    CONST(value), TIME, PREV(lag)
    Binary:  ADD/MUL/MOD/SUB/DIV/POW/EQ/LT (left, right)
    Ternary: IF (condition=left, then=right, else=extra)

    evaluate() needs observation_history for PREV nodes.
    If history is None, PREV returns 0 (safe default for synthesis mode).
    """
    node_type: NodeType
    value: Optional[int] = None         # CONST: the integer value
    lag: Optional[int] = None           # PREV: lag k (PREV(k) = obs[t-k])
    left: Optional['ExprNode'] = None   # Binary: left child / IF: condition
    right: Optional['ExprNode'] = None  # Binary: right child / IF: then-branch
    extra: Optional['ExprNode'] = None  # IF: else-branch (third child)

    def evaluate(
        self,
        t: int,
        history: Optional[List[int]] = None,
        alphabet_size: int = 256
    ) -> int:
        """
        Evaluate expression at timestep t.

        Args:
            t: Current timestep
            history: Observation history (needed for PREV nodes)
                     If None, PREV returns 0
            alphabet_size: For clamping (not applied here â€” caller clamps)

        Returns:
            Integer value of expression at t
        """
        # â”€â”€ Leaf nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.node_type == NodeType.CONST:
            return self.value

        if self.node_type == NodeType.TIME:
            return t

        if self.node_type == NodeType.PREV:
            lag = self.lag or 1
            idx = t - lag
            if history is None or idx < 0 or idx >= len(history):
                return 0
            return history[idx]

        # â”€â”€ Binary nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        lv = self.left.evaluate(t, history, alphabet_size)
        rv = self.right.evaluate(t, history, alphabet_size)

        if self.node_type == NodeType.ADD:
            return lv + rv
        if self.node_type == NodeType.SUB:
            return lv - rv
        if self.node_type == NodeType.MUL:
            return lv * rv
        if self.node_type == NodeType.MOD:
            return lv % rv if rv != 0 else 0
        if self.node_type == NodeType.DIV:
            return lv // rv if rv != 0 else 0
        if self.node_type == NodeType.POW:
            # Clamp exponent to prevent explosion
            exp = max(0, min(rv, 5))
            base = max(-100, min(lv, 100))
            try:
                return int(base ** exp)
            except (OverflowError, ValueError):
                return 0
        if self.node_type == NodeType.EQ:
            return 1 if lv == rv else 0
        if self.node_type == NodeType.LT:
            return 1 if lv < rv else 0

        # â”€â”€ Ternary node â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if self.node_type == NodeType.IF:
            condition_val = self.left.evaluate(t, history, alphabet_size)
            if condition_val != 0:
                return self.right.evaluate(t, history, alphabet_size)
            else:
                return self.extra.evaluate(t, history, alphabet_size)

        raise ValueError(f"Unknown node type: {self.node_type}")

    def predict_sequence(
        self,
        length: int,
        alphabet_size: int,
        initial_history: Optional[List[int]] = None
    ) -> List[int]:
        seeds = list(initial_history) if initial_history else []
        full_history = list(seeds)
        predictions = list(seeds)

        remaining = length - len(seeds)
        for _ in range(remaining):
            t = len(full_history)  # absolute timestep
            raw = self.evaluate(t, full_history, alphabet_size)
            clamped = raw % alphabet_size if alphabet_size > 0 else raw
            predictions.append(clamped)
            full_history.append(clamped)

        return predictions[:length]

    def to_string(self) -> str:
        """Human-readable infix representation."""
        if self.node_type == NodeType.CONST:
            return str(self.value)
        if self.node_type == NodeType.TIME:
            return 't'
        if self.node_type == NodeType.PREV:
            return f"prev({self.lag or 1})"

        ops = {
            NodeType.ADD: '+', NodeType.SUB: '-',
            NodeType.MUL: '*', NodeType.DIV: '//',
            NodeType.MOD: 'mod', NodeType.POW: '**',
            NodeType.EQ: '==', NodeType.LT: '<',
        }

        if self.node_type in ops:
            return f"({self.left.to_string()} {ops[self.node_type]} {self.right.to_string()})"

        if self.node_type == NodeType.IF:
            return (f"IF({self.left.to_string()}, "
                    f"{self.right.to_string()}, "
                    f"{self.extra.to_string()})")

        return f"UNKNOWN({self.node_type})"

    def to_bytes(self) -> bytes:
        return self.to_string().encode('utf-8')

    def depth(self) -> int:
        if self.node_type in LEAF_TYPES:
            return 0
        d = 0
        if self.left:
            d = max(d, self.left.depth())
        if self.right:
            d = max(d, self.right.depth())
        if self.extra:
            d = max(d, self.extra.depth())
        return d + 1

    def num_nodes(self) -> int:
        n = 1
        if self.left:  n += self.left.num_nodes()
        if self.right: n += self.right.num_nodes()
        if self.extra: n += self.extra.num_nodes()
        return n

    def has_prev(self) -> bool:
        """Check if expression tree contains any PREV nodes."""
        if self.node_type == NodeType.PREV:
            return True
        return any(
            child.has_prev()
            for child in [self.left, self.right, self.extra]
            if child is not None
        )

    def contains_time(self) -> bool:
        """Check if expression tree contains any TIME nodes."""
        if self.node_type == NodeType.TIME:
            return True
        return any(
            child.contains_time()
            for child in [self.left, self.right, self.extra]
            if child is not None
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExprNode):
            return False
        return self.to_string() == other.to_string()

    def __hash__(self) -> int:
        return hash(self.to_string())

    def __repr__(self) -> str:
        return f"ExprNode({self.to_string()!r})"


# â”€â”€ Convenience constructors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def C(n: int) -> ExprNode:
    return ExprNode(NodeType.CONST, value=n)

def T() -> ExprNode:
    return ExprNode(NodeType.TIME)

def PREV(lag: int = 1) -> ExprNode:
    return ExprNode(NodeType.PREV, lag=lag)

def ADD(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.ADD, left=l, right=r)

def SUB(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.SUB, left=l, right=r)

def MUL(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.MUL, left=l, right=r)

def DIV(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.DIV, left=l, right=r)

def POW(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.POW, left=l, right=r)

def MOD(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.MOD, left=l, right=r)

def EQ(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.EQ, left=l, right=r)

def LT(l: ExprNode, r: ExprNode) -> ExprNode:
    return ExprNode(NodeType.LT, left=l, right=r)

def IF(condition: ExprNode, then_branch: ExprNode,
       else_branch: ExprNode) -> ExprNode:
    return ExprNode(NodeType.IF, left=condition,
                    right=then_branch, extra=else_branch)

def build_linear_modular(slope: int, intercept: int, modulus: int) -> ExprNode:
    """Build (slope * t + intercept) mod modulus."""
    return MOD(ADD(MUL(C(slope), T()), C(intercept)), C(modulus))

def build_fibonacci_mod(modulus: int) -> ExprNode:
    """
    Build the Fibonacci recurrence mod modulus.
    F(t) = (prev(1) + prev(2)) mod modulus
    Requires initial_history=[0, 1] when calling predict_sequence.
    """
    return MOD(ADD(PREV(1), PREV(2)), C(modulus))


def predict_fibonacci_mod(modulus: int, length: int) -> list:
    """Predict Fibonacci mod modulus with correct [0,1] seeds."""
    expr = build_fibonacci_mod(modulus)
    return expr.predict_sequence(length, modulus, initial_history=[0, 1])

def build_piecewise(
    period: int,
    expr1: ExprNode,
    expr2: ExprNode
) -> ExprNode:
    """
    Build IF(t mod period == 0, expr1, expr2).
    Alternates between expr1 and expr2 every `period` steps.
    """
    return IF(EQ(MOD(T(), C(period)), C(0)), expr1, expr2)


# â”€â”€ Extended Beam Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class BeamSearchSynthesizer:
    """
    Beam search over expression trees â€” extended with new primitives.

    New candidates at depth 0:
        All PREV(k) for k in 1..max_lag (alongside CONST and TIME)

    New candidates at expansion:
        SUB, DIV, POW operations (alongside ADD, MUL, MOD)
        IF constructions (condition from EQ/LT, then/else from beam)

    PREV-containing expressions are scored using predict_sequence()
    in recurrence mode â€” the expression's own previous predictions
    are fed back as history. This is how Fibonacci emerges.

    Args:
        beam_width: Candidates kept at each depth
        max_depth: Maximum tree depth
        const_range: Search constants 0..const_range
        max_lag: Maximum PREV lag to try (default 3)
        alphabet_size: For prediction clamping
        mdl_lambda: MDL regularization weight
        enable_prev: Include PREV nodes (default True)
        enable_if: Include IF nodes (default True, slower)
        enable_pow: Include POW nodes (default True)
    """

    def __init__(
        self,
        beam_width: int = 25,
        max_depth: int = 3,
        const_range: int = 16,
        max_lag: int = 3,
        alphabet_size: int = 10,
        mdl_lambda: float = 1.0,
        enable_prev: bool = True,
        enable_if: bool = True,
        enable_pow: bool = True
    ):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.const_range = const_range
        self.max_lag = max_lag
        self.alphabet_size = alphabet_size
        self.mdl_lambda = mdl_lambda
        self.enable_prev = enable_prev
        self.enable_if = enable_if
        self.enable_pow = enable_pow

    def _leaves(self) -> List[ExprNode]:
        """All depth-0 candidates."""
        leaves = [T()]
        leaves += [C(n) for n in range(0, self.const_range + 1)]
        if self.enable_prev:
            leaves += [PREV(k) for k in range(1, self.max_lag + 1)]
        return leaves

    def _score(self, node: ExprNode, actuals: List[int]) -> float:
        """MDL cost of a node on actuals."""
        from ouroboros.compression.mdl import MDLCost
        n = len(actuals)
        preds = node.predict_sequence(n, self.alphabet_size)
        mdl = MDLCost(lambda_weight=self.mdl_lambda)
        return mdl.total_cost(node.to_bytes(), preds, actuals, self.alphabet_size)

    def _binary_ops(self) -> List[NodeType]:
        ops = [NodeType.ADD, NodeType.MUL, NodeType.MOD, NodeType.SUB]
        if self.enable_pow:
            ops.append(NodeType.POW)
        ops += [NodeType.DIV, NodeType.EQ, NodeType.LT]
        return ops

    def _expand(self, node: ExprNode) -> List[ExprNode]:
        """Expand node by one layer."""
        if node.depth() >= self.max_depth - 1:
            return []

        leaves = self._leaves()[:8]
        expansions = []

        for op in self._binary_ops():
            for leaf in leaves:
                expansions.append(ExprNode(op, left=node, right=leaf))
                if op in (NodeType.MOD, NodeType.SUB, NodeType.DIV, NodeType.LT):
                    expansions.append(ExprNode(op, left=leaf, right=node))

        # IF expansions: IF(EQ(node, leaf), leaf2, leaf3)
        if self.enable_if and node.depth() <= 1:
            for leaf1 in leaves[:5]:
                for leaf2 in leaves[:5]:
                    for leaf3 in leaves[:3]:
                        cond = ExprNode(NodeType.EQ, left=node, right=leaf1)
                        expansions.append(IF(cond, leaf2, leaf3))

        return expansions

    def search(
        self,
        actuals: List[int],
        verbose: bool = False
    ) -> Tuple[ExprNode, float]:
        """
        Beam search over expression trees.

        Returns: (best_expression, best_mdl_cost)
        """
        if not actuals:
            return C(0), float('inf')

        # Depth 0
        beam: List[Tuple[float, ExprNode]] = []
        for leaf in self._leaves():
            cost = self._score(leaf, actuals)
            beam.append((cost, leaf))

        beam.sort(key=lambda x: x[0])
        beam = beam[:self.beam_width]

        if verbose:
            print(f"  Depth 0: best={beam[0][1].to_string()!r} cost={beam[0][0]:.1f}")

        # Depth 1..max_depth
        for depth in range(1, self.max_depth + 1):
            new_candidates = []
            for _, node in beam:
                for expanded in self._expand(node):
                    cost = self._score(expanded, actuals)
                    new_candidates.append((cost, expanded))

            if not new_candidates:
                break

            all_candidates = beam + new_candidates
            all_candidates.sort(key=lambda x: x[0])
            beam = all_candidates[:self.beam_width]

            if verbose:
                print(f"  Depth {depth}: best={beam[0][1].to_string()!r} cost={beam[0][0]:.1f}")

            # Early exit: perfect prediction
            top_cost, top_expr = beam[0]
            preds = top_expr.predict_sequence(len(actuals), self.alphabet_size)
            if all(p == a for p, a in zip(preds, actuals)):
                if verbose:
                    print(f"  Perfect prediction at depth {depth}")
                break

        best_cost, best_expr = beam[0]
        return best_expr, best_cost

