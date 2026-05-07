"""
Symbolic program synthesis for OUROBOROS agents.
Extended node types (v3): Adds PRIME support for adversarial discovery.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
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
    PRIME = auto()  # ✅ Added for prime sequence discovery
    # New structural
    PREV  = auto()
    IF    = auto()
    EQ    = auto()
    LT    = auto()


# Node category helpers
LEAF_TYPES    = {NodeType.CONST, NodeType.TIME, NodeType.PREV}
UNARY_TYPES   = {NodeType.PRIME}  # ✅ PRIME only takes one child
BINARY_TYPES  = {NodeType.ADD, NodeType.MUL, NodeType.MOD,
                 NodeType.SUB, NodeType.DIV, NodeType.POW,
                 NodeType.EQ, NodeType.LT}
TERNARY_TYPES = {NodeType.IF}


@dataclass
class ExprNode:
    node_type: NodeType
    value: Optional[int] = None         # CONST: the integer value
    lag: Optional[int] = None           # PREV: lag k (PREV(k) = obs[t-k])
    left: Optional['ExprNode'] = None   # Binary: left child / Unary: child / IF: condition
    right: Optional['ExprNode'] = None  # Binary: right child / IF: then-branch
    extra: Optional['ExprNode'] = None  # IF: else-branch (third child)

    def evaluate(
        self,
        t: int,
        history: Optional[List[int]] = None,
        alphabet_size: int = 256
    ) -> int:
        # ── Leaf nodes ───────────────────────────────────────────────────
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

        # ── Unary nodes ──────────────────────────────────────────────────
        if self.node_type == NodeType.PRIME:
            import sympy
            # Use left child as index into prime sequence
            # We use % 10000 to keep sympy from hanging on massive primes during search
            idx = int(abs(np.round(self.left.evaluate(t, history, alphabet_size)))) % 10000
            return int(sympy.prime(idx + 1))

        # ── Binary nodes ─────────────────────────────────────────────────
        lv = self.left.evaluate(t, history, alphabet_size)
        rv = self.right.evaluate(t, history, alphabet_size) if self.right else 0

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

        # ── Ternary node ─────────────────────────────────────────────────
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
        predictions = []

        for current_t in range(length):
            # If we have seeds for the beginning, use them
            if current_t < len(seeds):
                clamped = seeds[current_t] % alphabet_size
            else:
                raw = self.evaluate(current_t, full_history, alphabet_size)
                clamped = raw % alphabet_size if alphabet_size > 0 else raw
            
            predictions.append(clamped)
            if current_t >= len(full_history):
                full_history.append(clamped)

        return predictions

    def to_string(self) -> str:
        if self.node_type == NodeType.CONST:
            return str(self.value)
        if self.node_type == NodeType.TIME:
            return 't'
        if self.node_type == NodeType.PREV:
            return f"prev({self.lag or 1})"
        if self.node_type == NodeType.PRIME:
            return f"prime({self.left.to_string()})"

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
        if self.left: d = max(d, self.left.depth())
        if self.right: d = max(d, self.right.depth())
        if self.extra: d = max(d, self.extra.depth())
        return d + 1

    def num_nodes(self) -> int:
        n = 1
        if self.left:  n += self.left.num_nodes()
        if self.right: n += self.right.num_nodes()
        if self.extra: n += self.extra.num_nodes()
        return n

    def has_prev(self) -> bool:
        if self.node_type == NodeType.PREV: return True
        children = [self.left, self.right, self.extra]
        return any(c.has_prev() for c in children if c is not None)

    def contains_time(self) -> bool:
        if self.node_type == NodeType.TIME: return True
        children = [self.left, self.right, self.extra]
        return any(c.contains_time() for c in children if c is not None)

# ── Convenience constructors ──────────────────────────────────────────────────

def C(n: int) -> ExprNode: return ExprNode(NodeType.CONST, value=n)
def T() -> ExprNode: return ExprNode(NodeType.TIME)
def PREV(lag: int = 1) -> ExprNode: return ExprNode(NodeType.PREV, lag=lag)
def PRIME(node: ExprNode) -> ExprNode: return ExprNode(NodeType.PRIME, left=node) # ✅ Added
def ADD(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.ADD, left=l, right=r)
def SUB(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.SUB, left=l, right=r)
def MUL(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.MUL, left=l, right=r)
def DIV(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.DIV, left=l, right=r)
def POW(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.POW, left=l, right=r)
def MOD(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.MOD, left=l, right=r)
def EQ(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.EQ, left=l, right=r)
def LT(l: ExprNode, r: ExprNode) -> ExprNode: return ExprNode(NodeType.LT, left=l, right=r)
def IF(c: ExprNode, t: ExprNode, e: ExprNode) -> ExprNode:
    return ExprNode(NodeType.IF, left=c, right=t, extra=e)

# ── Program builders ──────────────────────────────────────────────────────────

def build_linear_modular(a: int, b: int, m: int) -> ExprNode:
    """
    Builds the expression tree for (a*t + b) % m.
    
    This is the canonical depth-3 linear modular program:
        MOD(ADD(MUL(CONST(a), TIME), CONST(b)), CONST(m))
    
    Used by expr_node.py and grammar_beam.py as a fast-path constructor
    when the classifier identifies a LINEAR_MODULAR family.
    """
    if m <= 0:
        raise ValueError(f"Modulus m must be positive, got {m}")
    return MOD(ADD(MUL(C(a), T()), C(b)), C(m))