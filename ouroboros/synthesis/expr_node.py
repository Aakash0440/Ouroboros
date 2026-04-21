"""
Shim: re-exports ExprNode, NodeType, build_linear_modular from
ouroboros.compression.program_synthesis and patches the two methods
that search_strategy.py expects (node_count, constant_count).
"""
from ouroboros.compression.program_synthesis import (
    ExprNode, NodeType,
    build_linear_modular,
    LEAF_TYPES, BINARY_TYPES, TERNARY_TYPES,
    C, T, PREV, ADD, SUB, MUL, DIV, MOD, POW, EQ, LT, IF,
)

# Patch node_count → num_nodes alias (search_strategy.py calls node_count())
if not hasattr(ExprNode, 'node_count'):
    ExprNode.node_count = ExprNode.num_nodes

# Patch constant_count — count CONST nodes in the tree
def _constant_count(self) -> int:
    n = 1 if self.node_type == NodeType.CONST else 0
    for child in (self.left, self.right, self.extra):
        if child is not None:
            n += child.constant_count()
    return n

if not hasattr(ExprNode, 'constant_count'):
    ExprNode.constant_count = _constant_count

__all__ = [
    "ExprNode", "NodeType", "build_linear_modular",
    "LEAF_TYPES", "BINARY_TYPES", "TERNARY_TYPES",
    "C", "T", "PREV", "ADD", "SUB", "MUL", "DIV", "MOD", "POW", "EQ", "LT", "IF",
]