"""
Shim: re-exports ExprNode, NodeType, build_linear_modular from
ouroboros.compression.program_synthesis and patches node_count/constant_count.

NodeType is rebuilt as a merged enum containing BOTH the original members
AND every ExtNodeType member, so len(NodeType) == len(ExtNodeType) == 42
and grammar benchmarks use a consistent denominator.
"""
from enum import Enum
from ouroboros.compression.program_synthesis import (
    ExprNode,
    NodeType as _OrigNodeType,
    build_linear_modular,
    LEAF_TYPES, BINARY_TYPES, TERNARY_TYPES,
    C, T, PREV, ADD, SUB, MUL, DIV, MOD, POW, EQ, LT, IF,
)
from ouroboros.nodes.extended_nodes import ExtNodeType as _ExtNodeType

# ── Build merged enum: original members + all ExtNodeType members ─────────────
_merged: dict = {m.name: m.value for m in _OrigNodeType}

# Offset extended values so they don't collide with original (which use 1-based small ints)
_EXT_OFFSET = 200
for _m in _ExtNodeType:
    if _m.name not in _merged:
        _merged[_m.name] = _EXT_OFFSET + _m.value

NodeType = Enum("NodeType", _merged)

# ── Patch ExprNode ────────────────────────────────────────────────────────────
if not hasattr(ExprNode, "node_count"):
    ExprNode.node_count = ExprNode.num_nodes

def _constant_count(self) -> int:
    n = 1 if self.node_type == _OrigNodeType.CONST else 0
    for child in (
        self.left,
        self.right,
        getattr(self, "extra", None),
        getattr(self, "third", None),
    ):
        if child is not None:
            n += child.constant_count()
    return n

if not hasattr(ExprNode, "constant_count"):
    ExprNode.constant_count = _constant_count

__all__ = [
    "ExprNode", "NodeType", "build_linear_modular",
    "LEAF_TYPES", "BINARY_TYPES", "TERNARY_TYPES",
    "C", "T", "PREV", "ADD", "SUB", "MUL", "DIV", "MOD", "POW", "EQ", "LT", "IF",
]