"""
Extended node types for OUROBOROS — all 40 new mathematical primitives.

Design principles:
1. Every node has a NodeCategory (used by hierarchical classifier)
2. Every node has an arity (0=terminal, 1=unary, 2=binary, 3=ternary)
3. Every node has a description_bits cost (complex ops cost more to describe)
4. Every node has a protected evaluate() that never raises exceptions

Categories:
  CALCULUS    — derivatives, integrals, convolution
  STATISTICAL — rolling statistics, correlation, z-score
  LOGICAL     — thresholds, comparisons, boolean ops
  TRANSFORM   — FFT coefficients, autocorrelation, wavelets
  NUMBER      — GCD, LCM, totient, primality
  MEMORY      — rolling argmax/min, streaks, state variables
"""

from __future__ import annotations
import math
import cmath
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


class NodeCategory(Enum):
    """High-level category for each node type. Used by hierarchical classifier."""
    TERMINAL    = auto()   # CONST, TIME, PREV, STATE
    ARITHMETIC  = auto()   # ADD, SUB, MUL, DIV, MOD, POW — original ops
    CALCULUS    = auto()   # DERIV, DERIV2, CUMSUM, CUMSUM_WIN, CONVOLVE
    STATISTICAL = auto()   # MEAN_WIN, VAR_WIN, STD_WIN, CORR, ZSCORE, QUANTILE
    LOGICAL     = auto()   # IF, THRESHOLD, SIGN, COMPARE, AND, OR, NOT
    TRANSFORM   = auto()   # FFT_COEFF, AUTOCORR, HILBERT
    NUMBER      = auto()   # GCD, LCM, FLOOR, CEIL, ROUND, FRAC, TOTIENT, ISPRIME
    MEMORY      = auto()   # ARGMAX_WIN, ARGMIN_WIN, COUNT_WIN, STREAK, DELTA_ZERO
    TRANSCEND   = auto()   # SIN, COS, EXP, LOG, SQRT, ABS — original ops


class ExtNodeType(Enum):
    # ── CALCULUS (5 nodes) ────────────────────────────────────────────────────
    DERIV       = auto()   # (f(t) - f(t-1)) / 1  [unary]
    DERIV2      = auto()   # second derivative       [unary]
    CUMSUM      = auto()   # Σᵢ₌₀ᵗ f(i)            [unary]
    CUMSUM_WIN  = auto()   # Σᵢ₌ₜ₋ᵥ₊₁ᵗ f(i)        [binary: f, window_size]
    CONVOLVE    = auto()   # Σₖ f(k)*g(t-k)         [binary: f, g]

    # ── STATISTICAL (6 nodes) ─────────────────────────────────────────────────
    MEAN_WIN    = auto()   # rolling mean over W     [binary: f, W]
    VAR_WIN     = auto()   # rolling variance         [binary: f, W]
    STD_WIN     = auto()   # rolling std dev          [binary: f, W]
    CORR        = auto()   # rolling correlation      [ternary: f, g, W]
    ZSCORE      = auto()   # rolling z-score          [binary: f, W]
    QUANTILE    = auto()   # running quantile          [binary: f, q_const]

    # ── LOGICAL (7 nodes) ─────────────────────────────────────────────────────
    THRESHOLD   = auto()   # f(t) > c ? 1 : 0        [binary: f, c]
    SIGN        = auto()   # -1, 0, or +1             [unary]
    COMPARE     = auto()   # f(t) > g(t) ? 1 : 0     [binary: f, g]
    BOOL_AND    = auto()   # a(t) AND b(t)            [binary: a, b]
    BOOL_OR     = auto()   # a(t) OR b(t)             [binary: a, b]
    BOOL_NOT    = auto()   # NOT a(t)                 [unary]
    CLAMP       = auto()   # clamp(f, lo, hi)         [ternary: f, lo, hi]

    # ── TRANSFORM (4 nodes) ───────────────────────────────────────────────────
    FFT_AMP     = auto()   # amplitude at freq k over window W   [ternary: f, k, W]
    FFT_PHASE   = auto()   # phase at freq k over window W       [ternary: f, k, W]
    AUTOCORR    = auto()   # autocorrelation at lag L            [binary: f, L]
    HILBERT_ENV = auto()   # Hilbert envelope (amplitude)        [unary]

    # ── NUMBER-THEORETIC (8 nodes) ────────────────────────────────────────────
    GCD_NODE    = auto()   # gcd(a, b)               [binary: a, b]
    LCM_NODE    = auto()   # lcm(a, b)               [binary: a, b]
    FLOOR_NODE  = auto()   # floor(f)                [unary]
    CEIL_NODE   = auto()   # ceil(f)                 [unary]
    ROUND_NODE  = auto()   # round(f)                [unary]
    FRAC_NODE   = auto()   # f - floor(f)            [unary]
    TOTIENT     = auto()   # Euler's totient φ(n)    [unary, n integer]
    ISPRIME     = auto()   # 1 if prime else 0       [unary, n integer]

    # ── MEMORY / STATE (6 nodes) ──────────────────────────────────────────────
    ARGMAX_WIN  = auto()   # index of max in last W  [binary: f, W]
    ARGMIN_WIN  = auto()   # index of min in last W  [binary: f, W]
    COUNT_WIN   = auto()   # count(f==c in last W)   [ternary: f, c, W]
    STREAK      = auto()   # length of current run   [unary]
    DELTA_ZERO  = auto()   # steps since last zero   [unary]
    STATE_VAR   = auto()   # persistent state k      [terminal, k=const]

    # ── CALCULUS EXTRAS (4 nodes) ─────────────────────────────────────────────
    DIFF_QUOT   = auto()   # (f(t+h) - f(t-h)) / 2h central difference [binary: f, h]
    RUNNING_MAX = auto()   # max(f(0)..f(t))        [unary]
    RUNNING_MIN = auto()   # min(f(0)..f(t))        [unary]
    EWMA        = auto()   # exponentially weighted moving average [binary: f, alpha]


# ── Node metadata ──────────────────────────────────────────────────────────────

@dataclass
class NodeSpec:
    """Specification for one node type."""
    node_type: ExtNodeType
    category: NodeCategory
    arity: int                  # 0=terminal, 1=unary, 2=binary, 3=ternary
    description_bits: float     # MDL cost of using this node
    window_arg: bool = False    # True if second arg is a window size (integer)
    protected: bool = True      # True if evaluation is always finite


# Build the complete node specification table
NODE_SPECS: Dict[ExtNodeType, NodeSpec] = {
    # Calculus
    ExtNodeType.DERIV:       NodeSpec(ExtNodeType.DERIV,       NodeCategory.CALCULUS,    1, 4.0),
    ExtNodeType.DERIV2:      NodeSpec(ExtNodeType.DERIV2,      NodeCategory.CALCULUS,    1, 5.0),
    ExtNodeType.CUMSUM:      NodeSpec(ExtNodeType.CUMSUM,      NodeCategory.CALCULUS,    1, 4.0),
    ExtNodeType.CUMSUM_WIN:  NodeSpec(ExtNodeType.CUMSUM_WIN,  NodeCategory.CALCULUS,    2, 5.0, window_arg=True),
    ExtNodeType.CONVOLVE:    NodeSpec(ExtNodeType.CONVOLVE,    NodeCategory.CALCULUS,    2, 8.0),
    ExtNodeType.DIFF_QUOT:   NodeSpec(ExtNodeType.DIFF_QUOT,   NodeCategory.CALCULUS,    2, 6.0),
    ExtNodeType.RUNNING_MAX: NodeSpec(ExtNodeType.RUNNING_MAX, NodeCategory.CALCULUS,    1, 3.0),
    ExtNodeType.RUNNING_MIN: NodeSpec(ExtNodeType.RUNNING_MIN, NodeCategory.CALCULUS,    1, 3.0),
    ExtNodeType.EWMA:        NodeSpec(ExtNodeType.EWMA,        NodeCategory.CALCULUS,    2, 5.0),

    # Statistical
    ExtNodeType.MEAN_WIN:    NodeSpec(ExtNodeType.MEAN_WIN,    NodeCategory.STATISTICAL, 2, 5.0, window_arg=True),
    ExtNodeType.VAR_WIN:     NodeSpec(ExtNodeType.VAR_WIN,     NodeCategory.STATISTICAL, 2, 6.0, window_arg=True),
    ExtNodeType.STD_WIN:     NodeSpec(ExtNodeType.STD_WIN,     NodeCategory.STATISTICAL, 2, 6.0, window_arg=True),
    ExtNodeType.CORR:        NodeSpec(ExtNodeType.CORR,        NodeCategory.STATISTICAL, 3, 8.0, window_arg=True),
    ExtNodeType.ZSCORE:      NodeSpec(ExtNodeType.ZSCORE,      NodeCategory.STATISTICAL, 2, 7.0, window_arg=True),
    ExtNodeType.QUANTILE:    NodeSpec(ExtNodeType.QUANTILE,    NodeCategory.STATISTICAL, 2, 7.0),

    # Logical
    ExtNodeType.THRESHOLD:   NodeSpec(ExtNodeType.THRESHOLD,   NodeCategory.LOGICAL,     2, 4.0),
    ExtNodeType.SIGN:        NodeSpec(ExtNodeType.SIGN,        NodeCategory.LOGICAL,     1, 3.0),
    ExtNodeType.COMPARE:     NodeSpec(ExtNodeType.COMPARE,     NodeCategory.LOGICAL,     2, 4.0),
    ExtNodeType.BOOL_AND:    NodeSpec(ExtNodeType.BOOL_AND,    NodeCategory.LOGICAL,     2, 3.0),
    ExtNodeType.BOOL_OR:     NodeSpec(ExtNodeType.BOOL_OR,     NodeCategory.LOGICAL,     2, 3.0),
    ExtNodeType.BOOL_NOT:    NodeSpec(ExtNodeType.BOOL_NOT,    NodeCategory.LOGICAL,     1, 2.0),
    ExtNodeType.CLAMP:       NodeSpec(ExtNodeType.CLAMP,       NodeCategory.LOGICAL,     3, 5.0),

    # Transform
    ExtNodeType.FFT_AMP:     NodeSpec(ExtNodeType.FFT_AMP,     NodeCategory.TRANSFORM,   3, 10.0, window_arg=True),
    ExtNodeType.FFT_PHASE:   NodeSpec(ExtNodeType.FFT_PHASE,   NodeCategory.TRANSFORM,   3, 10.0, window_arg=True),
    ExtNodeType.AUTOCORR:    NodeSpec(ExtNodeType.AUTOCORR,    NodeCategory.TRANSFORM,   2, 8.0),
    ExtNodeType.HILBERT_ENV: NodeSpec(ExtNodeType.HILBERT_ENV, NodeCategory.TRANSFORM,   1, 9.0),

    # Number-theoretic
    ExtNodeType.GCD_NODE:    NodeSpec(ExtNodeType.GCD_NODE,    NodeCategory.NUMBER,      2, 5.0),
    ExtNodeType.LCM_NODE:    NodeSpec(ExtNodeType.LCM_NODE,    NodeCategory.NUMBER,      2, 5.0),
    ExtNodeType.FLOOR_NODE:  NodeSpec(ExtNodeType.FLOOR_NODE,  NodeCategory.NUMBER,      1, 3.0),
    ExtNodeType.CEIL_NODE:   NodeSpec(ExtNodeType.CEIL_NODE,   NodeCategory.NUMBER,      1, 3.0),
    ExtNodeType.ROUND_NODE:  NodeSpec(ExtNodeType.ROUND_NODE,  NodeCategory.NUMBER,      1, 3.0),
    ExtNodeType.FRAC_NODE:   NodeSpec(ExtNodeType.FRAC_NODE,   NodeCategory.NUMBER,      1, 3.0),
    ExtNodeType.TOTIENT:     NodeSpec(ExtNodeType.TOTIENT,     NodeCategory.NUMBER,      1, 7.0),
    ExtNodeType.ISPRIME:     NodeSpec(ExtNodeType.ISPRIME,     NodeCategory.NUMBER,      1, 6.0),

    # Memory
    ExtNodeType.ARGMAX_WIN:  NodeSpec(ExtNodeType.ARGMAX_WIN,  NodeCategory.MEMORY,      2, 6.0, window_arg=True),
    ExtNodeType.ARGMIN_WIN:  NodeSpec(ExtNodeType.ARGMIN_WIN,  NodeCategory.MEMORY,      2, 6.0, window_arg=True),
    ExtNodeType.COUNT_WIN:   NodeSpec(ExtNodeType.COUNT_WIN,   NodeCategory.MEMORY,      3, 7.0, window_arg=True),
    ExtNodeType.STREAK:      NodeSpec(ExtNodeType.STREAK,      NodeCategory.MEMORY,      1, 5.0),
    ExtNodeType.DELTA_ZERO:  NodeSpec(ExtNodeType.DELTA_ZERO,  NodeCategory.MEMORY,      1, 5.0),
    ExtNodeType.STATE_VAR:   NodeSpec(ExtNodeType.STATE_VAR,   NodeCategory.MEMORY,      0, 4.0),
}


# ── Extended ExprNode ──────────────────────────────────────────────────────────

PENALTY = 1e6
EPS = 1e-10

class ExtExprNode:
    """
    Extended expression node supporting all 60 node types.
    
    Combines the original NodeType from Days 1-29 with the new ExtNodeType.
    Both are represented in a single class for compatibility.
    
    The key invariant: evaluate() NEVER raises an exception.
    All operations are numerically protected.
    """
    
    __slots__ = ['node_type', 'value', 'lag', 'state_key', 'window',
                 'left', 'right', 'third', '_cache']

    def __init__(
        self,
        node_type,            # ExtNodeType or original NodeType
        value: float = 0.0,   # for CONST
        lag: int = 1,         # for PREV
        state_key: int = 0,   # for STATE_VAR
        window: int = 10,     # default window size
        left: Optional['ExtExprNode'] = None,
        right: Optional['ExtExprNode'] = None,
        third: Optional['ExtExprNode'] = None,
    ):
        self.node_type = node_type
        self.value = value
        self.lag = lag
        self.state_key = state_key
        self.window = window
        self.left = left
        self.right = right
        self.third = third
        self._cache: Dict[int, float] = {}  # memoization cache per timestep

    def evaluate(
        self,
        t: int,
        history: List[float],
        state: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        Evaluate the expression at timestep t.
        
        history: list of all previous observations [obs[0], obs[1], ..., obs[t-1]]
        state: persistent state dict shared across timesteps (for STATE_VAR)
        
        Never raises. Returns PENALTY if undefined.
        """
        try:
            return self._eval(t, history, state or {})
        except Exception:
            return PENALTY

    def _eval(self, t: int, history: List[float], state: Dict[int, float]) -> float:
        nt = self.node_type

        # ── Terminals ─────────────────────────────────────────────────────────
        if hasattr(nt, 'name') and nt.name == 'CONST':
            return float(self.value)
        if hasattr(nt, 'name') and nt.name == 'TIME':
            return float(t)
        if hasattr(nt, 'name') and nt.name == 'PREV':
            idx = t - self.lag
            return float(history[idx]) if 0 <= idx < len(history) else 0.0

        if nt == ExtNodeType.STATE_VAR:
            return state.get(self.state_key, 0.0)

        # ── Helper to evaluate children ────────────────────────────────────────
        def L() -> float:
            return self.left._eval(t, history, state) if self.left else 0.0
        def R() -> float:
            return self.right._eval(t, history, state) if self.right else 0.0
        def T() -> float:
            return self.third._eval(t, history, state) if self.third else 0.0

        # ── Helper: get a window's worth of history values of a sub-expression ─
        def hist_vals(expr: 'ExtExprNode', w: int) -> List[float]:
            w = max(1, min(int(w), 200))  # cap window at 200
            return [
                expr._eval(max(0, t - w + i + 1), history, state)
                for i in range(min(w, t + 1))
            ]

        # ── Calculus ───────────────────────────────────────────────────────────
        if nt == ExtNodeType.DERIV:
            v_now = L()
            v_prev = self.left._eval(max(0, t-1), history, state) if self.left else 0.0
            return v_now - v_prev

        if nt == ExtNodeType.DERIV2:
            v0 = L()
            v1 = self.left._eval(max(0, t-1), history, state) if self.left else 0.0
            v2 = self.left._eval(max(0, t-2), history, state) if self.left else 0.0
            return v0 - 2*v1 + v2

        if nt == ExtNodeType.CUMSUM:
            return sum(
                self.left._eval(i, history, state)
                for i in range(t + 1)
            ) if self.left else 0.0

        if nt == ExtNodeType.CUMSUM_WIN:
            w = max(1, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            return sum(vals)

        if nt == ExtNodeType.CONVOLVE:
            # Σₖ f(k) * g(t-k) for k in [0, t]
            total = 0.0
            for k in range(min(t + 1, 50)):  # cap at 50 to avoid O(t²)
                fk = self.left._eval(k, history, state) if self.left else 0.0
                gk = self.right._eval(t - k, history, state) if self.right else 0.0
                total += fk * gk
            return total

        if nt == ExtNodeType.DIFF_QUOT:
            h = max(1, int(abs(R()) + 0.5))
            v_plus  = self.left._eval(min(t + h, len(history)), history, state) if self.left else 0.0
            v_minus = self.left._eval(max(0, t - h), history, state) if self.left else 0.0
            return (v_plus - v_minus) / (2.0 * h)

        if nt == ExtNodeType.RUNNING_MAX:
            vals = [self.left._eval(i, history, state) for i in range(t + 1)] if self.left else [0.0]
            return max(vals)

        if nt == ExtNodeType.RUNNING_MIN:
            vals = [self.left._eval(i, history, state) for i in range(t + 1)] if self.left else [0.0]
            return min(vals)

        if nt == ExtNodeType.EWMA:
            alpha = max(0.01, min(0.99, abs(R())))
            result = 0.0
            for i in range(t + 1):
                v = self.left._eval(i, history, state) if self.left else 0.0
                result = alpha * v + (1 - alpha) * result
            return result

        # ── Statistical ───────────────────────────────────────────────────────
        if nt == ExtNodeType.MEAN_WIN:
            w = max(1, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            return sum(vals) / max(1, len(vals))

        if nt == ExtNodeType.VAR_WIN:
            w = max(1, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if len(vals) < 2: return 0.0
            m = sum(vals) / len(vals)
            return sum((v - m)**2 for v in vals) / (len(vals) - 1)

        if nt == ExtNodeType.STD_WIN:
            w = max(1, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if len(vals) < 2: return 0.0
            m = sum(vals) / len(vals)
            var = sum((v - m)**2 for v in vals) / (len(vals) - 1)
            return math.sqrt(max(0, var))

        if nt == ExtNodeType.CORR:
            w = max(2, int(abs(T()) + 0.5)) if self.third else 10
            fvals = hist_vals(self.left, w) if self.left else [0.0]
            gvals = hist_vals(self.right, w) if self.right else [0.0]
            n = min(len(fvals), len(gvals))
            if n < 2: return 0.0
            fvals, gvals = fvals[:n], gvals[:n]
            fm = sum(fvals)/n; gm = sum(gvals)/n
            num = sum((f-fm)*(g-gm) for f,g in zip(fvals, gvals))
            fs = math.sqrt(sum((f-fm)**2 for f in fvals))
            gs = math.sqrt(sum((g-gm)**2 for g in gvals))
            denom = max(fs * gs, EPS)
            return num / denom

        if nt == ExtNodeType.ZSCORE:
            w = max(2, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if len(vals) < 2: return 0.0
            m = sum(vals)/len(vals)
            s = math.sqrt(sum((v-m)**2 for v in vals)/(len(vals)-1))
            return (L() - m) / max(s, EPS)

        if nt == ExtNodeType.QUANTILE:
            q = max(0.01, min(0.99, abs(R()))) if self.right else 0.5
            w = 50  # fixed window for quantile
            vals = sorted(hist_vals(self.left, w)) if self.left else [0.0]
            idx = int(q * (len(vals) - 1))
            return vals[max(0, min(idx, len(vals)-1))]

        # ── Logical ───────────────────────────────────────────────────────────
        if nt == ExtNodeType.THRESHOLD:
            return 1.0 if L() > R() else 0.0

        if nt == ExtNodeType.SIGN:
            v = L()
            return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)

        if nt == ExtNodeType.COMPARE:
            return 1.0 if L() > R() else 0.0

        if nt == ExtNodeType.BOOL_AND:
            return 1.0 if (L() != 0) and (R() != 0) else 0.0

        if nt == ExtNodeType.BOOL_OR:
            return 1.0 if (L() != 0) or (R() != 0) else 0.0

        if nt == ExtNodeType.BOOL_NOT:
            return 0.0 if L() != 0 else 1.0

        if nt == ExtNodeType.CLAMP:
            return max(R(), min(T(), L()))

        # ── Transform ─────────────────────────────────────────────────────────
        if nt == ExtNodeType.FFT_AMP:
            w = max(4, int(abs(T()) + 0.5)) if self.third else 32
            k_freq = max(0, int(abs(R()) + 0.5)) if self.right else 1
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if len(vals) < 4: return 0.0
            # DFT coefficient k via direct computation (no numpy dependency)
            n = len(vals)
            k_freq = k_freq % (n // 2 + 1)
            real = sum(vals[j] * math.cos(2*math.pi*k_freq*j/n) for j in range(n))
            imag = sum(vals[j] * math.sin(2*math.pi*k_freq*j/n) for j in range(n))
            return math.sqrt(real**2 + imag**2) / max(n, 1)

        if nt == ExtNodeType.FFT_PHASE:
            w = max(4, int(abs(T()) + 0.5)) if self.third else 32
            k_freq = max(0, int(abs(R()) + 0.5)) if self.right else 1
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if len(vals) < 4: return 0.0
            n = len(vals)
            k_freq = k_freq % (n // 2 + 1)
            real = sum(vals[j] * math.cos(2*math.pi*k_freq*j/n) for j in range(n))
            imag = sum(vals[j] * math.sin(2*math.pi*k_freq*j/n) for j in range(n))
            return math.atan2(imag, real)

        if nt == ExtNodeType.AUTOCORR:
            lag_val = max(1, int(abs(R()) + 0.5)) if self.right else 1
            w = 50
            vals = hist_vals(self.left, w) if self.left else [0.0]
            n = len(vals)
            if n <= lag_val: return 0.0
            m = sum(vals) / n
            num = sum((vals[i] - m) * (vals[i - lag_val] - m) for i in range(lag_val, n))
            denom = sum((v - m)**2 for v in vals)
            return num / max(denom, EPS)

        if nt == ExtNodeType.HILBERT_ENV:
            # Approximation: instantaneous amplitude via |f + j*H(f)|
            # Use a simple approximation: sqrt(f² + DERIV(f)²)
            v = L()
            v_prev = self.left._eval(max(0, t-1), history, state) if self.left else 0.0
            deriv = v - v_prev
            return math.sqrt(v**2 + deriv**2)

        # ── Number-Theoretic ─────────────────────────────────────────────────
        if nt == ExtNodeType.GCD_NODE:
            a, b = int(abs(L())), int(abs(R()))
            if a == 0 and b == 0: return 0.0
            return float(math.gcd(a, b))

        if nt == ExtNodeType.LCM_NODE:
            a, b = int(abs(L())), int(abs(R()))
            if a == 0 or b == 0: return 0.0
            g = math.gcd(a, b)
            return float(a * b // g) if g > 0 else 0.0

        if nt == ExtNodeType.FLOOR_NODE:
            return math.floor(L())

        if nt == ExtNodeType.CEIL_NODE:
            return math.ceil(L())

        if nt == ExtNodeType.ROUND_NODE:
            return round(L())

        if nt == ExtNodeType.FRAC_NODE:
            v = L()
            return v - math.floor(v)

        if nt == ExtNodeType.TOTIENT:
            n = max(1, int(abs(L())))
            if n == 1: return 1.0
            result = n
            p = 2
            temp = n
            while p * p <= temp:
                if temp % p == 0:
                    while temp % p == 0:
                        temp //= p
                    result -= result // p
                p += 1
            if temp > 1:
                result -= result // temp
            return float(result)

        if nt == ExtNodeType.ISPRIME:
            n = int(abs(L()))
            if n < 2: return 0.0
            if n == 2: return 1.0
            if n % 2 == 0: return 0.0
            for i in range(3, min(int(n**0.5) + 1, 1000), 2):
                if n % i == 0: return 0.0
            return 1.0

        # ── Memory ────────────────────────────────────────────────────────────
        if nt == ExtNodeType.ARGMAX_WIN:
            w = max(1, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if not vals: return 0.0
            max_idx = vals.index(max(vals))
            return float(t - len(vals) + max_idx + 1)

        if nt == ExtNodeType.ARGMIN_WIN:
            w = max(1, int(abs(R()) + 0.5)) if self.right else 10
            vals = hist_vals(self.left, w) if self.left else [0.0]
            if not vals: return 0.0
            min_idx = vals.index(min(vals))
            return float(t - len(vals) + min_idx + 1)

        if nt == ExtNodeType.COUNT_WIN:
            w = max(1, int(abs(T()) + 0.5)) if self.third else 10
            target = R()
            vals = hist_vals(self.left, w) if self.left else [0.0]
            return float(sum(1 for v in vals if abs(v - target) < 0.5))

        if nt == ExtNodeType.STREAK:
            if not history: return 0.0
            current = history[-1] if history else L()
            count = 0
            for v in reversed(history):
                if abs(v - current) < 0.5:
                    count += 1
                else:
                    break
            return float(count)

        if nt == ExtNodeType.DELTA_ZERO:
            for i, v in enumerate(reversed(history)):
                if abs(v) < 0.5:
                    return float(i)
            return float(len(history))

        return PENALTY

    def node_count(self) -> int:
        c = 1
        if self.left: c += self.left.node_count()
        if self.right: c += self.right.node_count()
        if self.third: c += self.third.node_count()
        return c

    def constant_count(self) -> int:
        from ouroboros.synthesis.expr_node import NodeType
        c = 1 if (hasattr(self.node_type, 'name') and
                  self.node_type.name == 'CONST') else 0
        if self.left: c += self.left.constant_count()
        if self.right: c += self.right.constant_count()
        if self.third: c += self.third.constant_count()
        return c

    def depth(self) -> int:
        ld = self.left.depth() if self.left else 0
        rd = self.right.depth() if self.right else 0
        td = self.third.depth() if self.third else 0
        return 1 + max(ld, rd, td)

    def to_string(self) -> str:
        nt = self.node_type
        name = nt.name if hasattr(nt, 'name') else str(nt)
        if self.third:
            return f"{name}({self.left.to_string() if self.left else '?'}, {self.right.to_string() if self.right else '?'}, {self.third.to_string() if self.third else '?'})"
        if self.right:
            return f"{name}({self.left.to_string() if self.left else '?'}, {self.right.to_string() if self.right else '?'})"
        if self.left:
            return f"{name}({self.left.to_string()})"
        if name == 'CONST':
            return f"{self.value:.3f}"
        if name == 'TIME':
            return "t"
        if name == 'PREV':
            return f"obs[t-{self.lag}]"
        return name