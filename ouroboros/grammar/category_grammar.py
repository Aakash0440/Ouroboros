"""
CategoryConstraintGrammar — Tightly constrained grammar based on node categories.

Design principle:
  Every node type belongs to exactly one category.
  A parent node of category C only accepts children from specific allowed categories.
  This is enforced at the category level, not the individual node level.

Category hierarchy:
  TERMINAL:     CONST, TIME — leaf nodes, no children
  ARITHMETIC:   ADD, SUB, MUL, DIV, POW, ABS — accept ARITHMETIC, TERMINAL, CALCULUS
  LOGICAL:      BOOL_AND, BOOL_OR, BOOL_NOT, COMPARE, THRESHOLD, SIGN — accept TERMINAL, ARITHMETIC only
  CALCULUS:     DERIV, DERIV2, CUMSUM, INTEGRAL, EWMA, RUNNING_MAX, RUNNING_MIN — accept ARITHMETIC, TERMINAL, MEMORY
  STATISTICAL:  MEAN_WIN, VAR_WIN, STD_WIN, CORR, ZSCORE, QUANTILE — accept ARITHMETIC, TERMINAL, CALCULUS
  TRANSFORM:    FFT_AMP, FFT_PHASE, AUTOCORR, HILBERT_ENV, CONVOLVE — accept ARITHMETIC, TERMINAL only
  NUMBER_THEOR: GCD_NODE, LCM_NODE, TOTIENT, ISPRIME, FLOOR_NODE, CEIL_NODE, FRAC_NODE — accept TERMINAL, ARITHMETIC only
  MEMORY:       PREV, STATE_VAR, STREAK, DELTA_ZERO, ARGMAX_WIN, ARGMIN_WIN, COUNT_WIN, EWMA — accept ARITHMETIC, TERMINAL
  MODULAR:      MOD, IF, EQ, LT — accept ARITHMETIC, TERMINAL, LOGICAL

Allowed child categories per parent category:
  TERMINAL     → [] (no children — leaf nodes)
  ARITHMETIC   → [TERMINAL, ARITHMETIC, CALCULUS, MODULAR]
  LOGICAL      → [TERMINAL, ARITHMETIC, MODULAR]
  CALCULUS     → [TERMINAL, ARITHMETIC, MEMORY]
  STATISTICAL  → [TERMINAL, ARITHMETIC, CALCULUS]
  TRANSFORM    → [TERMINAL, ARITHMETIC]
  NUMBER_THEOR → [TERMINAL, ARITHMETIC]
  MEMORY       → [TERMINAL, ARITHMETIC]
  MODULAR      → [TERMINAL, ARITHMETIC, LOGICAL]

This gives effective branching:
  ARITHMETIC: accepts ~4 categories × avg 8 nodes/category = ~32 options (high)
  LOGICAL:    accepts ~3 categories × avg 8 = ~24 options (medium)
  CALCULUS:   accepts ~3 categories × avg 8 = ~24 options (medium)
  TRANSFORM:  accepts ~2 categories × avg 8 = ~16 options (tight)
  NUMBER_THEOR: accepts ~2 categories × avg 8 = ~16 options (tight)
  Average: ~22 options out of 55 ≈ 40% allowed

Target: average branching 10-15 (down from current 28)
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Set, Dict, List, FrozenSet


class NodeCategory(Enum):
    TERMINAL     = auto()
    ARITHMETIC   = auto()
    LOGICAL      = auto()
    CALCULUS     = auto()
    STATISTICAL  = auto()
    TRANSFORM    = auto()
    NUMBER_THEOR = auto()
    MEMORY       = auto()
    MODULAR      = auto()


# Map each NodeType name to its category
# This is the single source of truth for categorization
NODE_CATEGORY_MAP: Dict[str, NodeCategory] = {
    # TERMINAL — leaf nodes
    "CONST":     NodeCategory.TERMINAL,
    "TIME":      NodeCategory.TERMINAL,

    # ARITHMETIC — basic math operations
    "ADD":       NodeCategory.ARITHMETIC,
    "SUB":       NodeCategory.ARITHMETIC,
    "MUL":       NodeCategory.ARITHMETIC,
    "DIV":       NodeCategory.ARITHMETIC,
    "POW":       NodeCategory.ARITHMETIC,
    "ABS":       NodeCategory.ARITHMETIC,
    "SQRT":      NodeCategory.ARITHMETIC,
    "SIN":       NodeCategory.ARITHMETIC,
    "COS":       NodeCategory.ARITHMETIC,
    "EXP":       NodeCategory.ARITHMETIC,
    "LOG":       NodeCategory.ARITHMETIC,
    "DIFF_QUOT": NodeCategory.ARITHMETIC,
    "CLAMP":     NodeCategory.ARITHMETIC,

    # LOGICAL — boolean and comparison operations
    "BOOL_AND":  NodeCategory.LOGICAL,
    "BOOL_OR":   NodeCategory.LOGICAL,
    "BOOL_NOT":  NodeCategory.LOGICAL,
    "COMPARE":   NodeCategory.LOGICAL,
    "THRESHOLD": NodeCategory.LOGICAL,
    "SIGN":      NodeCategory.LOGICAL,

    # CALCULUS — integration and differentiation
    "DERIV":         NodeCategory.CALCULUS,
    "DERIV2":        NodeCategory.CALCULUS,
    "CUMSUM":        NodeCategory.CALCULUS,
    "INTEGRAL":      NodeCategory.CALCULUS,
    "INTEGRAL_WIN":  NodeCategory.CALCULUS,
    "CONVOLVE":      NodeCategory.CALCULUS,

    # STATISTICAL — windowed statistics
    "MEAN_WIN":  NodeCategory.STATISTICAL,
    "VAR_WIN":   NodeCategory.STATISTICAL,
    "STD_WIN":   NodeCategory.STATISTICAL,
    "CORR":      NodeCategory.STATISTICAL,
    "ZSCORE":    NodeCategory.STATISTICAL,
    "QUANTILE":  NodeCategory.STATISTICAL,

    # TRANSFORM — frequency domain
    "FFT_AMP":    NodeCategory.TRANSFORM,
    "FFT_PHASE":  NodeCategory.TRANSFORM,
    "AUTOCORR":   NodeCategory.TRANSFORM,
    "HILBERT_ENV":NodeCategory.TRANSFORM,

    # NUMBER_THEOR — discrete mathematics
    "GCD_NODE":  NodeCategory.NUMBER_THEOR,
    "LCM_NODE":  NodeCategory.NUMBER_THEOR,
    "TOTIENT":   NodeCategory.NUMBER_THEOR,
    "ISPRIME":   NodeCategory.NUMBER_THEOR,
    "FLOOR_NODE":NodeCategory.NUMBER_THEOR,
    "CEIL_NODE": NodeCategory.NUMBER_THEOR,
    "FRAC_NODE": NodeCategory.NUMBER_THEOR,
    "ROUND_NODE":NodeCategory.NUMBER_THEOR,

    # MEMORY — state and history
    "PREV":        NodeCategory.MEMORY,
    "STATE_VAR":   NodeCategory.MEMORY,
    "STREAK":      NodeCategory.MEMORY,
    "DELTA_ZERO":  NodeCategory.MEMORY,
    "ARGMAX_WIN":  NodeCategory.MEMORY,
    "ARGMIN_WIN":  NodeCategory.MEMORY,
    "COUNT_WIN":   NodeCategory.MEMORY,
    "EWMA":        NodeCategory.MEMORY,
    "RUNNING_MAX": NodeCategory.MEMORY,
    "RUNNING_MIN": NodeCategory.MEMORY,

    # MODULAR — control flow and modular arithmetic
    "MOD":  NodeCategory.MODULAR,
    "IF":   NodeCategory.MODULAR,
    "EQ":   NodeCategory.MODULAR,
    "LT":   NodeCategory.MODULAR,
}

# Which child categories each parent category accepts
ALLOWED_CHILD_CATEGORIES: Dict[NodeCategory, FrozenSet[NodeCategory]] = {
    NodeCategory.TERMINAL:     frozenset(),  # leaf — no children
    NodeCategory.ARITHMETIC:   frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
        NodeCategory.CALCULUS,
        NodeCategory.MODULAR,
    }),
    NodeCategory.LOGICAL:      frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
        NodeCategory.MODULAR,
    }),
    NodeCategory.CALCULUS:     frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
        NodeCategory.MEMORY,
    }),
    NodeCategory.STATISTICAL:  frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
        NodeCategory.CALCULUS,
    }),
    NodeCategory.TRANSFORM:    frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
    }),
    NodeCategory.NUMBER_THEOR: frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
    }),
    NodeCategory.MEMORY:       frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
    }),
    NodeCategory.MODULAR:      frozenset({
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
        NodeCategory.LOGICAL,
    }),
}


def get_node_category(node_type_name: str) -> NodeCategory:
    """Get the category of a node type by name."""
    return NODE_CATEGORY_MAP.get(node_type_name, NodeCategory.ARITHMETIC)


def get_allowed_children_for_node(
    parent_name: str,
    all_node_names: List[str],
) -> Set[str]:
    """
    Get allowed child node names for a parent node.
    Returns a set of node type names that can be children.
    """
    parent_cat = get_node_category(parent_name)
    allowed_child_cats = ALLOWED_CHILD_CATEGORIES.get(parent_cat, frozenset())

    return {
        name for name in all_node_names
        if get_node_category(name) in allowed_child_cats
    }


class CategoryConstraintGrammar:
    """
    Grammar that enforces categorical type constraints on expression trees.

    Drop-in replacement for MathGrammar with stricter constraints.
    Reduces average branching from ~28 to ~10-15.
    """

    def __init__(self):
        # Import actual NodeType to get real node names
        try:
            from ouroboros.synthesis.expr_node import NodeType
            self._all_node_names = [nt.name for nt in NodeType]
        except ImportError:
            self._all_node_names = list(NODE_CATEGORY_MAP.keys())

        # Pre-compute allowed children for each node type
        self._cache: Dict[str, Set[str]] = {}
        for name in self._all_node_names:
            self._cache[name] = get_allowed_children_for_node(
                name, self._all_node_names
            )

    def get_allowed_children(self, parent_type) -> Set[str]:
        """
        Get allowed child node names for a parent NodeType.
        Compatible with the MathGrammar interface.
        """
        name = parent_type.name if hasattr(parent_type, 'name') else str(parent_type)
        return self._cache.get(name, set(self._all_node_names))

    def is_valid_child(self, parent_type, child_type) -> bool:
        """Check if child_type is a valid child of parent_type."""
        child_name = child_type.name if hasattr(child_type, 'name') else str(child_type)
        return child_name in self.get_allowed_children(parent_type)

    def get_branching_stats(self) -> dict:
        """Return statistics about the grammar's branching factor."""
        n_total = len(self._all_node_names)
        counts = [len(v) for v in self._cache.values() if v]
        if not counts:
            return {}
        avg = sum(counts) / len(counts)
        return {
            "n_node_types": n_total,
            "avg_branching": round(avg, 2),
            "min_branching": min(counts),
            "max_branching": max(counts),
            "reduction_factor": round(n_total / max(avg, 1), 2),
            "depth5_space": round(avg**5),
        }

    def audit(self, verbose: bool = True) -> dict:
        """Print a detailed grammar audit."""
        n_total = len(self._all_node_names)
        stats = self.get_branching_stats()

        restricted = []
        unrestricted = []
        for name, allowed in self._cache.items():
            ratio = len(allowed) / n_total
            if ratio < 0.25:
                restricted.append((name, len(allowed), ratio))
            elif ratio > 0.60:
                unrestricted.append((name, len(allowed), ratio))

        if verbose:
            print(f"CategoryConstraintGrammar Audit")
            print(f"  Total node types: {n_total}")
            print(f"  Avg branching: {stats['avg_branching']:.2f}")
            print(f"  Reduction factor: {stats['reduction_factor']:.1f}x")
            print(f"  Search space (depth 5): {stats['depth5_space']:.2e}")
            print(f"\n  Tight (< 25% allowed): {len(restricted)}")
            for name, cnt, r in sorted(restricted, key=lambda x: x[1])[:10]:
                print(f"    {name}: {cnt}/{n_total} ({r:.1%})")
            print(f"\n  Loose (> 60% allowed): {len(unrestricted)}")
            for name, cnt, r in sorted(unrestricted, key=lambda x: -x[1])[:5]:
                print(f"    {name}: {cnt}/{n_total} ({r:.1%})")

        return {
            "stats": stats,
            "n_restricted": len(restricted),
            "n_unrestricted": len(unrestricted),
        }