# math_grammar.py  --  ASCII only, UTF-8 no BOM
"""
MathGrammar -- grammar constraints for OUROBOROS expression-tree search.

All lookups go through _RULES[node.name] -- works identically for
ExtNodeType members and original NodeType shims (both have .name).

_RULES layout per name:  (arg0_cats, arg1_cats, arg2_cats)
  frozenset[NodeCategory]  -- allowed categories for that argument position
  None                     -- argument position does not exist for this node
"""

from __future__ import annotations
import warnings
from typing import Dict, FrozenSet, List, Tuple
from ouroboros.nodes.extended_nodes import ExtNodeType, NodeCategory, NODE_SPECS

ANY_TYPES = frozenset(NodeCategory)

# -- Category shorthands -------------------------------------------------------
_T  = frozenset([NodeCategory.TERMINAL])
_A  = frozenset([NodeCategory.ARITHMETIC])
_C  = frozenset([NodeCategory.CALCULUS])
_S  = frozenset([NodeCategory.STATISTICAL])
_L  = frozenset([NodeCategory.LOGICAL])
_TR = frozenset([NodeCategory.TRANSFORM])
_N  = frozenset([NodeCategory.NUMBER])
_M  = frozenset([NodeCategory.MEMORY])
_TX = frozenset([NodeCategory.TRANSCEND])

# -- Semantic groups -----------------------------------------------------------
_NUMERIC = _T | _A | _C | _S | _N | _M | _TX   # any real-valued scalar
_SMOOTH  = _T | _A | _C | _S | _TX              # continuous / differentiable
_INTEGER = _T | _A | _N | _M                     # guaranteed integer-valued
_BOOLEAN = _L                                    # guaranteed 0/1 output
_CONST   = _T                                    # constant terminals only
_SCALAR  = _T | _A | _N          # terminals + arithmetic + numbers only
_EXPONENT = _T | _N             # only terminals/numbers as exponents (CONST, TIME, STATE_VAR)

# =============================================================================
# _RULES -- keyed by node .name string
# Works for both ExtNodeType and original NodeType shims.
# =============================================================================
_RULES: Dict[str, Tuple] = {

    # -- Arithmetic (tightened to _SCALAR) ------------------------------------
    "ADD":   (_SCALAR, _SCALAR, None),
    "SUB":   (_SCALAR, _SCALAR, None),
    "MUL":   (_SCALAR, _SCALAR, None),
    "DIV":   (_SCALAR, _SCALAR, None),
    "MOD":   (_SCALAR, _SCALAR, None),
    "POW":   (_SMOOTH, _EXPONENT, None),

    # -- Transcendentals (tightened to _SMOOTH) --------------------------------
    "SIN":   (_SMOOTH, None, None),
    "COS":   (_SMOOTH, None, None),
    "EXP":   (_SMOOTH, None, None),
    "LOG":   (_SMOOTH, None, None),
    "SQRT":  (_SMOOTH, None, None),
    "ABS":   (_SCALAR, None, None),
    "PRIME": (_INTEGER, None, None),

    # -- Logical / control -----------------------------------------------------
    "EQ":    (_SCALAR, _SCALAR, None),
    "LT":    (_SCALAR, _SCALAR, None),
    "IF":    (_BOOLEAN, _SCALAR, _SCALAR),

    # -- Terminals (arity 0) ---------------------------------------------------
    "CONST":     (frozenset(), None, None),
    "TIME":      (frozenset(), None, None),
    "PREV":      (frozenset(), None, None),
    "STATE_VAR": (frozenset(), None, None),

    # -- CALCULUS (accept _SMOOTH inputs, window args must be _CONST) ----------
    "DERIV":        (_SMOOTH,  None,    None),
    "DERIV2":       (_SMOOTH,  None,    None),
    "DIFF_QUOT":    (_SMOOTH,  _CONST,  None),
    "CUMSUM":       (_SCALAR,  None,    None),
    "CUMSUM_WIN":   (_SCALAR,  _CONST,  None),
    "INTEGRAL":     (_SMOOTH,  None,    None),
    "INTEGRAL_WIN": (_SMOOTH,  _CONST,  None),
    "CONVOLVE":     (_SCALAR,  _SCALAR, None),
    "RUNNING_MAX":  (_SCALAR,  None,    None),
    "RUNNING_MIN":  (_SCALAR,  None,    None),
    "EWMA":         (_SCALAR,  _CONST,  None),

    # -- STATISTICAL (scalar inputs, window = _CONST) -------------------------
    "MEAN_WIN": (_SCALAR, _CONST,   None),
    "VAR_WIN":  (_SCALAR, _CONST,   None),
    "STD_WIN":  (_SCALAR, _CONST,   None),
    "ZSCORE":   (_SCALAR, _CONST,   None),
    "QUANTILE": (_SCALAR, _CONST,   None),
    "CORR":     (_SCALAR, _SCALAR,  _CONST),

    # -- LOGICAL ---------------------------------------------------------------
    "THRESHOLD":   (_SCALAR,  _CONST,   None),
    "SIGN":        (_SCALAR,  None,     None),
    "COMPARE":     (_SCALAR,  _SCALAR,  None),
    "BOOL_AND":    (_BOOLEAN, _BOOLEAN, None),
    "BOOL_OR":     (_BOOLEAN, _BOOLEAN, None),
    "BOOL_NOT":    (_BOOLEAN, None,     None),
    "CLAMP":       (_SCALAR,  _CONST,   _CONST),

    # -- TRANSFORM (smooth inputs only) ----------------------------------------
    "FFT_AMP":     (_SMOOTH, _CONST, _CONST),
    "FFT_PHASE":   (_SMOOTH, _CONST, _CONST),
    "AUTOCORR":    (_SMOOTH, _CONST, None),
    "HILBERT_ENV": (_SMOOTH, None,   None),

    # -- NUMBER-THEORETIC (integer inputs) -------------------------------------
    "GCD_NODE":   (_INTEGER, _INTEGER, None),
    "LCM_NODE":   (_INTEGER, _INTEGER, None),
    "FLOOR_NODE": (_SCALAR,  None,     None),
    "CEIL_NODE":  (_SCALAR,  None,     None),
    "ROUND_NODE": (_SCALAR,  None,     None),
    "FRAC_NODE":  (_SCALAR,  None,     None),
    "TOTIENT":    (_INTEGER, None,     None),
    "ISPRIME":    (_INTEGER, None,     None),

    # -- MEMORY ----------------------------------------------------------------
    "ARGMAX_WIN": (_SCALAR, _CONST,  None),
    "ARGMIN_WIN": (_SCALAR, _CONST,  None),
    "COUNT_WIN":  (_SCALAR, _CONST,  _CONST),
    "STREAK":     (_SCALAR, None,    None),
    "DELTA_ZERO": (_SCALAR, None,    None),
}


# =============================================================================
# MathGrammar
# =============================================================================

class MathGrammar:
    """
    Enforces mathematical grammar constraints during beam search.

    All parent-node lookups use _RULES[node.name], making the class
    immune to import-order issues and dual enum hierarchies.
    """

    def __init__(self, strict=True):
        self.strict = strict
        self._by_cat: Dict[NodeCategory, list] = {}
        for nt, spec in NODE_SPECS.items():
            self._by_cat.setdefault(spec.category, []).append(nt)

    # -- Internal --------------------------------------------------------------

    def _nodes_for_cats(self, cats):
        result = []
        for cat in cats:
            result.extend(self._by_cat.get(cat, []))
        return result

    def _cats_for(self, parent_type, arg_index):
        if not self.strict:
            return ANY_TYPES
        name = parent_type.name if hasattr(parent_type, "name") else str(parent_type)
        rule = _RULES.get(name)
        if rule is None:
            warnings.warn(
                "MathGrammar: no rule for '{}' (arg {}) "
                "-- permissive fallback. Add '{}' to _RULES.".format(
                    name, arg_index, name),
                stacklevel=4,
            )
            return ANY_TYPES
        cats = rule[arg_index] if arg_index < len(rule) else None
        return cats if cats is not None else frozenset()

    # -- Public API ------------------------------------------------------------

    def get_allowed_children(self, parent_type, arg_index=0):
        """Return all node objects valid as the arg_index-th child of parent_type."""
        if not self.strict:
            return list(NODE_SPECS.keys())
        return self._nodes_for_cats(self._cats_for(parent_type, arg_index))

    def allowed_child_categories(self, parent_type, arg_index=0):
        return self._cats_for(parent_type, arg_index)

    def allowed_child_node_types(self, parent_type, arg_index=0,
                                  include_terminals=True):
        return self.get_allowed_children(parent_type, arg_index)

    def category_of(self, node_type):
        spec = NODE_SPECS.get(node_type)
        return spec.category if spec else NodeCategory.ARITHMETIC

    def is_valid_combination(self, parent_type, child_type, arg_index=0):
        if not self.strict:
            return True
        return child_type in self.get_allowed_children(parent_type, arg_index)

    # -- Analytics -------------------------------------------------------------

    def effective_branching_factor(self, arg_index=0):
        total, n = 0, 0
        for nt, spec in NODE_SPECS.items():
            if spec.arity > arg_index:
                total += len(self.get_allowed_children(nt, arg_index))
                n += 1
        return total / max(1, n)

    def print_branching_report(self):
        total = len(NODE_SPECS)
        rows = []
        for nt, spec in NODE_SPECS.items():
            if spec.arity == 0:
                continue
            name = nt.name if hasattr(nt, "name") else str(nt)
            allowed = len(self.get_allowed_children(nt, 0))
            rows.append((name, allowed, spec.arity, spec.category.name))
        rows.sort(key=lambda r: r[1])

        print("\n{:<22} {:<14} {:>5} {:>7} {:>6}".format(
            "Node", "Category", "Arity", "Allowed", "%"))
        print("-" * 58)
        for name, allowed, arity, cat in rows:
            print("  {:<20} {:<14} {:>5}  {:>6}  {:>5.1f}%".format(
                name, cat, arity, allowed, allowed / total * 100))

        avg = self.effective_branching_factor()
        target = 28.0
        print("\n  Average branching factor   : {:.2f}".format(avg))
        print("  Total node types           : {}".format(total))
        print("  Reduction vs unconstrained : {:.1f}x".format(total / avg))
        print("  Target (55-node system)    : < {:.0f}".format(target))
        print("  RESULT: {}  (got {:.2f}, target < {:.0f})".format(
            "PASS" if avg < target else "FAIL", avg, target))


# -- Global instances ----------------------------------------------------------
DEFAULT_GRAMMAR    = MathGrammar(strict=True)
PERMISSIVE_GRAMMAR = MathGrammar(strict=False)