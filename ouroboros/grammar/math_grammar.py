"""
MathGrammar — Formal grammar specifying valid parent-child node combinations.

This is the core mechanism that makes 60 node types tractable.

Without grammar: branching factor = 60, depth 4 → 60^16 ≈ 10^28 candidates
With grammar:    branching factor ≈ 6 (average valid children), depth 4 → 6^16 ≈ 10^12
With pruning:    practical beam search explores ~25 * 12 iterations → ~300 candidates

Grammar rules encode mathematical validity:
  ✓ DERIV(SIN(TIME))          — derivative of a continuous function: valid
  ✗ DERIV(ISPRIME(TIME))      — derivative of a discrete indicator: meaningless
  ✓ MEAN_WIN(SIN(TIME), 20)   — rolling mean of a continuous signal: valid
  ✗ MEAN_WIN(ISPRIME, 20)     — rolling mean of a primality check: weird but allowed
  ✓ FFT_AMP(SIN(TIME), 1, 32) — Fourier amplitude of a sinusoid: valid
  ✗ FFT_AMP(ISPRIME, 1, 32)   — Fourier transform of primality: invalid
  ✓ BOOL_AND(THRESHOLD(f,c), COMPARE(f,g)) — logical combination of binary signals
  ✗ BOOL_AND(SIN(TIME), EXP(TIME))         — AND of continuous values: meaningless

The grammar is organized as a dict:
  parent_category → set of allowed child categories for each argument position
"""

from __future__ import annotations
from typing import Set, Dict, List, Optional, FrozenSet
from ouroboros.nodes.extended_nodes import ExtNodeType, NodeCategory, NODE_SPECS

# Also import original node categories for backward compatibility
CONTINUOUS_TYPES = frozenset([
    NodeCategory.CALCULUS,
    NodeCategory.STATISTICAL,
    NodeCategory.TRANSFORM,
    NodeCategory.TRANSCEND,
    NodeCategory.ARITHMETIC,
])

DISCRETE_TYPES = frozenset([
    NodeCategory.NUMBER,
    NodeCategory.LOGICAL,
    NodeCategory.MEMORY,
])

TERMINAL_TYPES = frozenset([NodeCategory.TERMINAL])
ANY_TYPES = frozenset(NodeCategory)


# ── Grammar: allowed child categories per (parent_node, argument_position) ────

# Format: (ExtNodeType, arg_index) → frozenset of allowed NodeCategory for that arg
GRAMMAR: Dict[tuple, FrozenSet[NodeCategory]] = {

    # ── CALCULUS NODES ─────────────────────────────────────────────────────────
    # DERIV makes sense on continuous signals, not discrete indicators
    (ExtNodeType.DERIV, 0):       frozenset([NodeCategory.CALCULUS, NodeCategory.STATISTICAL,
                                              NodeCategory.TRANSCEND, NodeCategory.ARITHMETIC,
                                              NodeCategory.TERMINAL]),
    (ExtNodeType.DERIV2, 0):      frozenset([NodeCategory.CALCULUS, NodeCategory.TRANSCEND,
                                              NodeCategory.ARITHMETIC, NodeCategory.TERMINAL]),
    (ExtNodeType.CUMSUM, 0):      frozenset([NodeCategory.CALCULUS, NodeCategory.STATISTICAL,
                                              NodeCategory.TRANSCEND, NodeCategory.ARITHMETIC,
                                              NodeCategory.TERMINAL, NodeCategory.LOGICAL]),
    (ExtNodeType.CUMSUM_WIN, 0):  frozenset([NodeCategory.CALCULUS, NodeCategory.TRANSCEND,
                                              NodeCategory.ARITHMETIC, NodeCategory.TERMINAL]),
    (ExtNodeType.CUMSUM_WIN, 1):  frozenset([NodeCategory.TERMINAL]),  # window = CONST
    (ExtNodeType.CONVOLVE, 0):    frozenset([NodeCategory.CALCULUS, NodeCategory.TRANSCEND,
                                              NodeCategory.ARITHMETIC, NodeCategory.TERMINAL]),
    (ExtNodeType.CONVOLVE, 1):    frozenset([NodeCategory.CALCULUS, NodeCategory.TRANSCEND,
                                              NodeCategory.ARITHMETIC, NodeCategory.TERMINAL]),
    (ExtNodeType.DIFF_QUOT, 0):   frozenset([NodeCategory.CALCULUS, NodeCategory.TRANSCEND,
                                              NodeCategory.ARITHMETIC, NodeCategory.TERMINAL]),
    (ExtNodeType.DIFF_QUOT, 1):   frozenset([NodeCategory.TERMINAL]),  # step h = CONST
    (ExtNodeType.RUNNING_MAX, 0): ANY_TYPES - frozenset([NodeCategory.TRANSFORM]),
    (ExtNodeType.RUNNING_MIN, 0): ANY_TYPES - frozenset([NodeCategory.TRANSFORM]),
    (ExtNodeType.EWMA, 0):        frozenset([NodeCategory.CALCULUS, NodeCategory.TRANSCEND,
                                              NodeCategory.ARITHMETIC, NodeCategory.TERMINAL]),
    (ExtNodeType.EWMA, 1):        frozenset([NodeCategory.TERMINAL]),  # alpha = CONST

    # ── STATISTICAL NODES ─────────────────────────────────────────────────────
    # Rolling statistics make sense on any continuous signal
    (ExtNodeType.MEAN_WIN, 0):    CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.MEAN_WIN, 1):    TERMINAL_TYPES,  # window = CONST
    (ExtNodeType.VAR_WIN, 0):     CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.VAR_WIN, 1):     TERMINAL_TYPES,
    (ExtNodeType.STD_WIN, 0):     CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.STD_WIN, 1):     TERMINAL_TYPES,
    (ExtNodeType.CORR, 0):        CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.CORR, 1):        CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.CORR, 2):        TERMINAL_TYPES,  # window = CONST
    (ExtNodeType.ZSCORE, 0):      CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.ZSCORE, 1):      TERMINAL_TYPES,
    (ExtNodeType.QUANTILE, 0):    CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.QUANTILE, 1):    TERMINAL_TYPES,  # q in (0,1) = CONST

    # ── LOGICAL NODES ─────────────────────────────────────────────────────────
    # Threshold produces binary 0/1 — can apply to anything
    (ExtNodeType.THRESHOLD, 0):   ANY_TYPES - frozenset([NodeCategory.LOGICAL]),
    (ExtNodeType.THRESHOLD, 1):   TERMINAL_TYPES,  # threshold = CONST
    (ExtNodeType.SIGN, 0):        CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.COMPARE, 0):     CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.COMPARE, 1):     CONTINUOUS_TYPES | TERMINAL_TYPES,
    # BOOL_AND/OR should take binary (logical) inputs
    (ExtNodeType.BOOL_AND, 0):    frozenset([NodeCategory.LOGICAL]),
    (ExtNodeType.BOOL_AND, 1):    frozenset([NodeCategory.LOGICAL]),
    (ExtNodeType.BOOL_OR, 0):     frozenset([NodeCategory.LOGICAL]),
    (ExtNodeType.BOOL_OR, 1):     frozenset([NodeCategory.LOGICAL]),
    (ExtNodeType.BOOL_NOT, 0):    frozenset([NodeCategory.LOGICAL]),
    (ExtNodeType.CLAMP, 0):       CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.CLAMP, 1):       TERMINAL_TYPES,  # lo = CONST
    (ExtNodeType.CLAMP, 2):       TERMINAL_TYPES,  # hi = CONST

    # ── TRANSFORM NODES ───────────────────────────────────────────────────────
    # FFT makes sense only on continuous signals
    (ExtNodeType.FFT_AMP, 0):     CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.FFT_AMP, 1):     TERMINAL_TYPES,   # freq index k = CONST
    (ExtNodeType.FFT_AMP, 2):     TERMINAL_TYPES,   # window W = CONST
    (ExtNodeType.FFT_PHASE, 0):   CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.FFT_PHASE, 1):   TERMINAL_TYPES,
    (ExtNodeType.FFT_PHASE, 2):   TERMINAL_TYPES,
    (ExtNodeType.AUTOCORR, 0):    CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.AUTOCORR, 1):    TERMINAL_TYPES,   # lag L = CONST
    (ExtNodeType.HILBERT_ENV, 0): CONTINUOUS_TYPES | TERMINAL_TYPES,

    # ── NUMBER-THEORETIC NODES ────────────────────────────────────────────────
    # GCD/LCM take integer inputs — floor them first in practice
    (ExtNodeType.GCD_NODE, 0):    ANY_TYPES,
    (ExtNodeType.GCD_NODE, 1):    ANY_TYPES,
    (ExtNodeType.LCM_NODE, 0):    ANY_TYPES,
    (ExtNodeType.LCM_NODE, 1):    ANY_TYPES,
    (ExtNodeType.FLOOR_NODE, 0):  CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.CEIL_NODE, 0):   CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.ROUND_NODE, 0):  CONTINUOUS_TYPES | TERMINAL_TYPES,
    (ExtNodeType.FRAC_NODE, 0):   CONTINUOUS_TYPES | TERMINAL_TYPES,
    # TOTIENT and ISPRIME expect integer inputs
    (ExtNodeType.TOTIENT, 0):     ANY_TYPES,
    (ExtNodeType.ISPRIME, 0):     ANY_TYPES,

    # ── MEMORY NODES ──────────────────────────────────────────────────────────
    (ExtNodeType.ARGMAX_WIN, 0):  ANY_TYPES,
    (ExtNodeType.ARGMAX_WIN, 1):  TERMINAL_TYPES,
    (ExtNodeType.ARGMIN_WIN, 0):  ANY_TYPES,
    (ExtNodeType.ARGMIN_WIN, 1):  TERMINAL_TYPES,
    (ExtNodeType.COUNT_WIN, 0):   ANY_TYPES,
    (ExtNodeType.COUNT_WIN, 1):   TERMINAL_TYPES,   # target value = CONST
    (ExtNodeType.COUNT_WIN, 2):   TERMINAL_TYPES,   # window W = CONST
    (ExtNodeType.STREAK, 0):      ANY_TYPES,
    (ExtNodeType.DELTA_ZERO, 0):  ANY_TYPES,
}


class MathGrammar:
    """
    Enforces the mathematical grammar during expression tree construction.
    
    Usage:
        grammar = MathGrammar()
        allowed = grammar.allowed_child_types(parent_type, arg_index=0)
        # → frozenset of NodeCategory values that are valid children
        
        grammar.is_valid_tree(root_expr)
        # → True if entire tree satisfies grammar rules
    """

    def __init__(self, strict: bool = True):
        """
        strict=True:  enforce all grammar rules (recommended for main search)
        strict=False: allow any combination (used for mutation in beam search)
        """
        self.strict = strict
        self._category_to_types: Dict[NodeCategory, List[ExtNodeType]] = {}
        self._build_category_index()

    def _build_category_index(self) -> None:
        """Index node types by category for fast lookup."""
        for nt, spec in NODE_SPECS.items():
            cat = spec.category
            if cat not in self._category_to_types:
                self._category_to_types[cat] = []
            self._category_to_types[cat].append(nt)

    def allowed_child_categories(
        self,
        parent_type: ExtNodeType,
        arg_index: int,
    ) -> FrozenSet[NodeCategory]:
        """
        Return the set of NodeCategory values allowed as the arg_index-th
        child of parent_type.
        
        Returns ANY_TYPES if no rule is specified (permissive default).
        """
        if not self.strict:
            return ANY_TYPES
        key = (parent_type, arg_index)
        return GRAMMAR.get(key, ANY_TYPES)

    def allowed_child_node_types(
        self,
        parent_type: ExtNodeType,
        arg_index: int,
        include_terminals: bool = True,
    ) -> List[ExtNodeType]:
        """
        Return list of ExtNodeType values that are grammatically valid
        as the arg_index-th child of parent_type.
        """
        allowed_cats = self.allowed_child_categories(parent_type, arg_index)
        result = []
        for nt, spec in NODE_SPECS.items():
            if spec.category in allowed_cats:
                if not include_terminals and spec.arity == 0:
                    continue
                result.append(nt)
        return result

    def category_of(self, node_type: ExtNodeType) -> NodeCategory:
        """Return the NodeCategory for a given ExtNodeType."""
        spec = NODE_SPECS.get(node_type)
        if spec:
            return spec.category
        # Fall back for original NodeTypes
        return NodeCategory.ARITHMETIC

    def is_valid_combination(
        self,
        parent_type: ExtNodeType,
        child_type,
        arg_index: int,
    ) -> bool:
        """Check if a specific parent-child-position combination is valid."""
        if not self.strict:
            return True
        child_cat = self.category_of(child_type) if isinstance(child_type, ExtNodeType) \
                    else NodeCategory.ARITHMETIC
        allowed = self.allowed_child_categories(parent_type, arg_index)
        return child_cat in allowed

    def effective_branching_factor(self, depth: int, arg_index: int = 0) -> float:
        """
        Compute the average number of valid children for a node at this depth.
        Used to estimate the actual search space size.
        """
        total_valid = 0
        n_parents = 0
        for nt, spec in NODE_SPECS.items():
            if spec.arity > arg_index:
                valid_children = len(self.allowed_child_node_types(nt, arg_index))
                total_valid += valid_children
                n_parents += 1
        return total_valid / max(1, n_parents)

    def search_space_size(self, max_depth: int) -> float:
        """Estimate the grammar-constrained search space size."""
        avg_branching = self.effective_branching_factor(0)
        # Binary tree with branching factor b at each level
        return avg_branching ** (2 ** max_depth)


# Global grammar instance
DEFAULT_GRAMMAR = MathGrammar(strict=True)
PERMISSIVE_GRAMMAR = MathGrammar(strict=False)