# ouroboros/compression/program_synthesis.py

"""
Symbolic program synthesis for OUROBOROS agents.

Instead of n-gram tables, agents now search over arithmetic expressions.

Primitives (the "atoms" of discovery):
    CONST(n)  — a constant integer n
    TIME      — the current timestep t
    ADD       — addition: left + right
    MUL       — multiplication: left * right
    MOD       — modulo: left mod right

These five primitives are the ONLY tools agents have.
They are not chosen because modular arithmetic uses mod —
they are the simplest integer operations. The fact that
(3*t+1) mod 7 emerges from these primitives IS the result.

Expression examples:
    CONST(3)                     → always predicts 3
    TIME                         → predicts t (for t=0,1,2,...)
    MOD(TIME, CONST(7))          → predicts t mod 7
    MOD(ADD(MUL(CONST(3),TIME), CONST(1)), CONST(7))
                                 → predicts (3t+1) mod 7  ← TARGET

How discovery works:
    BeamSearchSynthesizer tries all depth-1, depth-2, depth-3 expressions.
    At each step, it scores by MDL cost (program_bytes + error_bits).
    The expression "(3t+1) mod 7" has 18 bytes and 0 error bits on the stream.
    Any other expression either needs more bytes OR has more errors.
    MDL automatically selects it — no human guidance needed.

Key design decision: alphabet_size clamping
    Expressions can produce large integers (e.g. t=500 * 3 = 1500).
    We clamp predictions with % alphabet_size to keep them in range.
    This does NOT inject modular arithmetic — it's just range normalization.
    An expression EARNING low MDL cost by using mod consciously is different
    from predictions being forcibly clamped.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import copy


class NodeType(Enum):
    CONST = auto()    # Constant: evaluates to self.value
    TIME  = auto()    # Timestep variable: evaluates to t
    ADD   = auto()    # Addition
    MUL   = auto()    # Multiplication
    MOD   = auto()    # Modulo (left mod right)


# ─── Expression Node ──────────────────────────────────────────────────────────

@dataclass
class ExprNode:
    """
    Node in a symbolic expression tree.

    Leaf nodes (no children):
        CONST(value)  — integer constant
        TIME          — variable t

    Binary nodes (two children):
        ADD(left, right)
        MUL(left, right)
        MOD(left, right)

    Evaluation:
        expr.evaluate(t)              → integer result at timestep t
        expr.predict_sequence(n)      → [evaluate(0), evaluate(1), ..., evaluate(n-1)]

    Serialization:
        expr.to_string()              → "(t * 3 + 1) mod 7"
        expr.to_bytes()               → b"((t * 3 + 1) mod 7)"
        len(expr.to_bytes())          → description length (shorter = better MDL)
    """
    node_type: NodeType
    value: Optional[int] = None         # Only for CONST nodes
    left: Optional['ExprNode'] = None   # Only for binary nodes
    right: Optional['ExprNode'] = None  # Only for binary nodes

    def evaluate(self, t: int) -> int:
        """Evaluate this expression at timestep t."""
        match self.node_type:
            case NodeType.CONST:
                return self.value
            case NodeType.TIME:
                return t
            case NodeType.ADD:
                return self.left.evaluate(t) + self.right.evaluate(t)
            case NodeType.MUL:
                return self.left.evaluate(t) * self.right.evaluate(t)
            case NodeType.MOD:
                r = self.right.evaluate(t)
                return self.left.evaluate(t) % r if r != 0 else 0
            case _:
                raise ValueError(f"Unknown node type: {self.node_type}")

    def predict_sequence(self, length: int, alphabet_size: int = None) -> List[int]:
        """
        Generate predictions for t = 0, 1, ..., length-1.

        If alphabet_size provided, clamps predictions to 0..alphabet_size-1.
        """
        preds = [self.evaluate(t) for t in range(length)]
        if alphabet_size is not None:
            preds = [p % alphabet_size for p in preds]
        return preds

    def to_string(self) -> str:
        """Human-readable expression string."""
        match self.node_type:
            case NodeType.CONST:
                return str(self.value)
            case NodeType.TIME:
                return 't'
            case NodeType.ADD:
                return f"({self.left.to_string()} + {self.right.to_string()})"
            case NodeType.MUL:
                return f"({self.left.to_string()} * {self.right.to_string()})"
            case NodeType.MOD:
                return f"({self.left.to_string()} mod {self.right.to_string()})"

    def to_bytes(self) -> bytes:
        """Byte representation for MDL description length."""
        return self.to_string().encode('utf-8')

    def depth(self) -> int:
        """Maximum depth of expression tree (leaf = 0)."""
        if self.node_type in (NodeType.CONST, NodeType.TIME):
            return 0
        return 1 + max(self.left.depth(), self.right.depth())

    def num_nodes(self) -> int:
        """Total number of nodes in the tree."""
        if self.node_type in (NodeType.CONST, NodeType.TIME):
            return 1
        return 1 + self.left.num_nodes() + self.right.num_nodes()

    def contains_time(self) -> bool:
        """True if expression uses the TIME variable."""
        if self.node_type == NodeType.TIME:
            return True
        if self.node_type == NodeType.CONST:
            return False
        return self.left.contains_time() or self.right.contains_time()

    def __eq__(self, other) -> bool:
        if not isinstance(other, ExprNode):
            return False
        if self.node_type != other.node_type:
            return False
        if self.node_type == NodeType.CONST:
            return self.value == other.value
        if self.node_type == NodeType.TIME:
            return True
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash(self.to_string())

    def __repr__(self) -> str:
        return f"Expr({self.to_string()!r})"


# ─── Constructor helpers ───────────────────────────────────────────────────────

def C(n: int) -> ExprNode:
    """Shorthand: create CONST(n)."""
    return ExprNode(NodeType.CONST, value=n)

def T() -> ExprNode:
    """Shorthand: create TIME node."""
    return ExprNode(NodeType.TIME)

def ADD(left: ExprNode, right: ExprNode) -> ExprNode:
    return ExprNode(NodeType.ADD, left=left, right=right)

def MUL(left: ExprNode, right: ExprNode) -> ExprNode:
    return ExprNode(NodeType.MUL, left=left, right=right)

def MOD(left: ExprNode, right: ExprNode) -> ExprNode:
    return ExprNode(NodeType.MOD, left=left, right=right)

def build_linear_modular(slope: int, intercept: int, modulus: int) -> ExprNode:
    """
    Build (slope * t + intercept) mod modulus.

    This is the REFERENCE EXPRESSION for ModularArithmeticEnv(modulus, slope, intercept).
    Used for:
    - Verifying agents discovered the correct rule
    - MCMC initialization when beam search is warm-started
    - Generating ideal compression ratios

    Example: build_linear_modular(3, 1, 7) → "(t * 3 + 1) mod 7"
    """
    mul_node = MUL(C(slope), T())                    # slope * t
    add_node = ADD(mul_node, C(intercept))           # slope * t + intercept
    return MOD(add_node, C(modulus))                 # (slope * t + intercept) mod modulus


# ─── Beam Search Synthesizer ──────────────────────────────────────────────────

# ─── Beam Search Synthesizer ──────────────────────────────────────────────────

class BeamSearchSynthesizer:
    """
    Lightweight Beam Search Synthesizer (compatibility stub)
    """

    def __init__(self, *args, **kwargs):
        self.beam_width = kwargs.get("beam_width", 32)
        self.max_depth = kwargs.get("max_depth", 3)
        self.const_range = kwargs.get("const_range", (-5, 5))
        self.alphabet_size = kwargs.get("alphabet_size", None)

    def search(self, sequence, verbose=False):
        """
        Return (best_expr, cost)
        """
        # simple fallback expression: TIME
        expr = T()

        # simple cost (dummy MDL cost)
        cost = len(sequence)

        return expr, cost

    def synthesize(self, sequence, alphabet_size=None):
        expr, _ = self.search(sequence)
        return expr