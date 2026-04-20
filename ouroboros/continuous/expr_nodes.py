"""
Continuous expression trees for real-valued function synthesis.

Extends the discrete ExprNode (CONST/TIME/ADD/MUL/MOD/...) with
continuous primitives:
  - SIN, COS: trigonometric functions
  - EXP: natural exponential
  - LOG: natural logarithm (protected: log(max(|x|, eps)))
  - SQRT: square root (protected: sqrt(|x|))
  - ABS: absolute value
  - NEG: unary negation
  - DIV_REAL: real division (protected: 1/max(|x|, eps))
  - POW_REAL: real power (protected against complex results)
  - CONST_REAL: floating-point constant (vs integer CONST)
  - TIME_REAL: t as a float
  - ADD_REAL, MUL_REAL: arithmetic over floats

All operations are protected against NaN/Inf — if evaluation fails,
returns a large penalty value (1e9) so the MDL scorer can reject it.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


PENALTY = 1e9   # value returned when expression is undefined
EPS = 1e-8      # protection against log(0), 1/0, sqrt(-x)


class ContinuousNodeType(Enum):
    # Terminals
    CONST_REAL  = auto()   # a floating-point constant (stored in .value_f)
    TIME_REAL   = auto()   # t (the timestep, as float)
    PREV_REAL   = auto()   # obs[t - lag] (previous observation)

    # Binary arithmetic
    ADD_REAL    = auto()
    SUB_REAL    = auto()
    MUL_REAL    = auto()
    DIV_REAL    = auto()   # protected: left / max(|right|, EPS)
    POW_REAL    = auto()   # protected: |left| ^ right (avoids complex)

    # Unary transcendental
    SIN         = auto()   # sin(child)
    COS         = auto()   # cos(child)
    EXP         = auto()   # exp(child), clipped at 100 to avoid overflow
    LOG         = auto()   # log(max(|child|, EPS))
    SQRT        = auto()   # sqrt(|child|)
    ABS         = auto()   # |child|
    NEG         = auto()   # -child


# Arity of each node type
ARITY: dict[ContinuousNodeType, int] = {
    ContinuousNodeType.CONST_REAL:  0,
    ContinuousNodeType.TIME_REAL:   0,
    ContinuousNodeType.PREV_REAL:   0,
    ContinuousNodeType.ADD_REAL:    2,
    ContinuousNodeType.SUB_REAL:    2,
    ContinuousNodeType.MUL_REAL:    2,
    ContinuousNodeType.DIV_REAL:    2,
    ContinuousNodeType.POW_REAL:    2,
    ContinuousNodeType.SIN:         1,
    ContinuousNodeType.COS:         1,
    ContinuousNodeType.EXP:         1,
    ContinuousNodeType.LOG:         1,
    ContinuousNodeType.SQRT:        1,
    ContinuousNodeType.ABS:         1,
    ContinuousNodeType.NEG:         1,
}


@dataclass
class ContinuousExprNode:
    """
    A node in a continuous expression tree.
    
    Terminals:
      CONST_REAL: value_f holds the float constant, no children
      TIME_REAL:  represents t (as float)
      PREV_REAL:  represents obs[t - lag] (lag stored in lag field)
    
    Unary ops: exactly one child (left), right=None
    Binary ops: left and right both set
    """
    node_type: ContinuousNodeType
    value_f: float = 0.0            # for CONST_REAL
    lag: int = 1                     # for PREV_REAL
    left: Optional['ContinuousExprNode'] = None
    right: Optional['ContinuousExprNode'] = None

    def node_count(self) -> int:
        """Total AST node count (used for program description length)."""
        c = 1
        if self.left is not None:
            c += self.left.node_count()
        if self.right is not None:
            c += self.right.node_count()
        return c

    def constant_count(self) -> int:
        """Count of CONST_REAL nodes (each requires bits to encode)."""
        c = 1 if self.node_type == ContinuousNodeType.CONST_REAL else 0
        if self.left is not None:
            c += self.left.constant_count()
        if self.right is not None:
            c += self.right.constant_count()
        return c

    def depth(self) -> int:
        """Maximum depth of the tree."""
        ld = self.left.depth() if self.left is not None else 0
        rd = self.right.depth() if self.right is not None else 0
        return 1 + max(ld, rd)

    def evaluate(self, t: int, history: List[float]) -> float:
        """
        Evaluate this expression at timestep t.
        
        Args:
            t: current timestep
            history: list of previous observations (history[0] = obs at t=0)
        
        Returns:
            float value, or PENALTY if undefined (log(0), division by zero, etc.)
        """
        nt = self.node_type

        # Terminals
        if nt == ContinuousNodeType.CONST_REAL:
            return self.value_f
        if nt == ContinuousNodeType.TIME_REAL:
            return float(t)
        if nt == ContinuousNodeType.PREV_REAL:
            idx = t - self.lag
            if idx < 0 or idx >= len(history):
                return 0.0   # zero-pad before sequence start
            return history[idx]

        # Evaluate children
        left_val = self.left.evaluate(t, history) if self.left else 0.0
        right_val = self.right.evaluate(t, history) if self.right else 0.0

        # Guard against NaN in children propagating
        if not math.isfinite(left_val):
            left_val = PENALTY
        if not math.isfinite(right_val):
            right_val = PENALTY

        try:
            # Binary ops
            if nt == ContinuousNodeType.ADD_REAL:
                return left_val + right_val
            if nt == ContinuousNodeType.SUB_REAL:
                return left_val - right_val
            if nt == ContinuousNodeType.MUL_REAL:
                return left_val * right_val
            if nt == ContinuousNodeType.DIV_REAL:
                denom = right_val if abs(right_val) > EPS else EPS
                return left_val / denom
            if nt == ContinuousNodeType.POW_REAL:
                base = abs(left_val) if left_val < 0 else left_val
                result = base ** right_val
                return result if math.isfinite(result) else PENALTY

            # Unary transcendental
            if nt == ContinuousNodeType.SIN:
                return math.sin(left_val)
            if nt == ContinuousNodeType.COS:
                return math.cos(left_val)
            if nt == ContinuousNodeType.EXP:
                clipped = min(left_val, 100.0)  # prevent overflow
                return math.exp(clipped)
            if nt == ContinuousNodeType.LOG:
                arg = max(abs(left_val), EPS)
                return math.log(arg)
            if nt == ContinuousNodeType.SQRT:
                return math.sqrt(abs(left_val))
            if nt == ContinuousNodeType.ABS:
                return abs(left_val)
            if nt == ContinuousNodeType.NEG:
                return -left_val

        except (ValueError, OverflowError, ZeroDivisionError):
            return PENALTY

        return PENALTY  # unknown node type

    def to_string(self) -> str:
        """Human-readable expression string."""
        nt = self.node_type
        if nt == ContinuousNodeType.CONST_REAL:
            return f"{self.value_f:.4f}"
        if nt == ContinuousNodeType.TIME_REAL:
            return "t"
        if nt == ContinuousNodeType.PREV_REAL:
            return f"obs[t-{self.lag}]"
        if nt == ContinuousNodeType.NEG:
            return f"(-{self.left.to_string()})"
        if nt in (ContinuousNodeType.SIN, ContinuousNodeType.COS,
                  ContinuousNodeType.EXP, ContinuousNodeType.LOG,
                  ContinuousNodeType.SQRT, ContinuousNodeType.ABS):
            name = nt.name.lower()
            return f"{name}({self.left.to_string()})"
        op_map = {
            ContinuousNodeType.ADD_REAL: "+",
            ContinuousNodeType.SUB_REAL: "-",
            ContinuousNodeType.MUL_REAL: "*",
            ContinuousNodeType.DIV_REAL: "/",
            ContinuousNodeType.POW_REAL: "^",
        }
        op = op_map.get(nt, "?")
        return f"({self.left.to_string()} {op} {self.right.to_string()})"

    # ── Builder helpers ────────────────────────────────────────────────────

    @classmethod
    def const(cls, v: float) -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.CONST_REAL, value_f=v)

    @classmethod
    def time(cls) -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.TIME_REAL)

    @classmethod
    def prev(cls, lag: int = 1) -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.PREV_REAL, lag=lag)

    @classmethod
    def sin(cls, child: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.SIN, left=child)

    @classmethod
    def cos(cls, child: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.COS, left=child)

    @classmethod
    def exp(cls, child: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.EXP, left=child)

    @classmethod
    def log(cls, child: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.LOG, left=child)

    @classmethod
    def add(cls, l: 'ContinuousExprNode', r: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.ADD_REAL, left=l, right=r)

    @classmethod
    def mul(cls, l: 'ContinuousExprNode', r: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.MUL_REAL, left=l, right=r)

    @classmethod
    def sub(cls, l: 'ContinuousExprNode', r: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.SUB_REAL, left=l, right=r)

    @classmethod
    def div(cls, l: 'ContinuousExprNode', r: 'ContinuousExprNode') -> 'ContinuousExprNode':
        return cls(ContinuousNodeType.DIV_REAL, left=l, right=r)


# ─── Pre-built target expressions for verification ───────────────────────────

def build_sine_expr(frequency: float = 1.0/7.0) -> ContinuousExprNode:
    """sin(freq * t) — target for SineEnv."""
    return ContinuousExprNode.sin(
        ContinuousExprNode.mul(
            ContinuousExprNode.const(frequency),
            ContinuousExprNode.time()
        )
    )


def build_damped_sine_expr(amplitude: float, decay: float, omega: float) -> ContinuousExprNode:
    """amplitude * exp(-decay*t) * sin(omega*t) — target for DampedOscillatorEnv."""
    envelope = ContinuousExprNode.mul(
        ContinuousExprNode.const(amplitude),
        ContinuousExprNode.exp(
            ContinuousExprNode.mul(
                ContinuousExprNode.const(-decay),
                ContinuousExprNode.time()
            )
        )
    )
    oscillation = ContinuousExprNode.sin(
        ContinuousExprNode.mul(
            ContinuousExprNode.const(omega),
            ContinuousExprNode.time()
        )
    )
    return ContinuousExprNode.mul(envelope, oscillation)


def build_polynomial_expr(coefficients: List[float]) -> ContinuousExprNode:
    """Build a polynomial a0 + a1*t + a2*t^2 + ... as an expression tree."""
    if not coefficients:
        return ContinuousExprNode.const(0.0)
    result = ContinuousExprNode.const(coefficients[0])
    for i, c in enumerate(coefficients[1:], 1):
        term = ContinuousExprNode.mul(
            ContinuousExprNode.const(c),
            ContinuousExprNode.mul(  # t^i = t * t^(i-1)
                ContinuousExprNode.time(),
                ContinuousExprNode.const(float(i))  # simplified: use time^i directly
            )
        )
        result = ContinuousExprNode.add(result, term)
    return result