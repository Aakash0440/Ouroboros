"""
ExpressionFragment — Partial expression trees with holes.

An ExpressionFragment is an ExprNode where some leaves are replaced
by HOLE nodes. A HOLE represents an unknown sub-expression that
another agent (or the same agent) must fill in.

Types of holes:
  HOLE_CONST   — must be filled with a CONST node (agent searches only over constants)
  HOLE_UNARY   — must be filled with a unary subtree (SIN, COS, etc.)
  HOLE_BINARY  — must be filled with a binary subtree (ADD, MUL, etc.)
  HOLE_ANY     — can be filled with any subtree

This enables collaborative synthesis:
  1. Agent A finds MUL(CONST(?), MOD(TIME, CONST(7))) — correct structure,
     but doesn't know the constant in position 1
  2. Agent A broadcasts this fragment with HOLE_CONST at position 1
  3. Agent B receives the fragment, searches only over constants 0..N
  4. Agent B quickly finds CONST(3) → completes to (3 * t) % 7
  5. Agent B broadcasts the completed expression
  6. Agent A adopts Agent B's completion

The key efficiency: Agent B's search is O(N) over constants,
vs O(N^depth) for a full beam search from scratch.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from ouroboros.synthesis.expr_node import ExprNode, NodeType


class HoleType(Enum):
    """Type of hole — constrains what can fill it."""
    HOLE_CONST  = auto()   # must be CONST or CONST-like
    HOLE_UNARY  = auto()   # must be unary op applied to TIME or CONST
    HOLE_BINARY = auto()   # must be binary op
    HOLE_ANY    = auto()   # any subtree


@dataclass
class ExpressionFragment:
    """
    A partial expression tree where some leaves are holes.
    
    The fragment is a standard ExprNode tree, but some nodes have
    node_type == NodeType.CONST with a special sentinel value (-9999)
    indicating they are holes.
    
    Better design: we use a parallel tree where each node is either
    a concrete ExprNode or a HoleSpec describing what kind of node
    must fill that position.
    """
    root: ExprNode              # The partial tree (holes represented as CONST(-9999))
    holes: List['HoleSpec']     # Descriptions of each hole
    creator_agent: str          # Who created this fragment
    environment_name: str       # Which env the fragment was found in
    partial_mdl_cost: float     # MDL cost before holes are filled

    @property
    def n_holes(self) -> int:
        return len(self.holes)

    @property
    def is_complete(self) -> bool:
        return len(self.holes) == 0

    def description(self) -> str:
        return (
            f"Fragment by {self.creator_agent} on {self.environment_name}\n"
            f"  Tree: {self.root.to_string()}\n"
            f"  Holes: {[h.description() for h in self.holes]}\n"
            f"  Partial MDL: {self.partial_mdl_cost:.2f}"
        )


HOLE_SENTINEL = -9999   # Sentinel value marking a hole in CONST nodes


@dataclass
class HoleSpec:
    """Specification of a hole in a fragment."""
    hole_type: HoleType
    position_path: List[int]   # Path from root: 0=left, 1=right at each level
    expected_range: Optional[Tuple[int, int]] = None  # For HOLE_CONST: [min, max]
    depth_limit: int = 2   # Max depth of subtree that can fill this hole

    def description(self) -> str:
        return (
            f"Hole({self.hole_type.name} at {self.position_path}, "
            f"range={self.expected_range})"
        )


def create_fragment_from_expr(
    expr: ExprNode,
    n_holes: int = 1,
    hole_type: HoleType = HoleType.HOLE_CONST,
    preferred_positions: List[List[int]] = None,
) -> ExpressionFragment:
    """
    Create a fragment from a complete expression by replacing n_holes
    leaf CONST nodes with holes.
    
    This is used when an agent wants to share its expression structure
    while asking others to search for the best constants.
    """
    fragment_root = copy.deepcopy(expr)
    holes_created = []

    # Find CONST nodes to replace with holes
    const_paths = []
    _find_const_paths(fragment_root, [], const_paths)

    # Replace up to n_holes const nodes with holes
    for i, path in enumerate(const_paths[:n_holes]):
        node = _get_node_at_path(fragment_root, path)
        if node is not None:
            original_value = node.value
            node.value = HOLE_SENTINEL  # mark as hole
            holes_created.append(HoleSpec(
                hole_type=HoleType.HOLE_CONST,
                position_path=path,
                expected_range=(0, 50),   # default const range
                depth_limit=0,   # just a constant
            ))

    return ExpressionFragment(
        root=fragment_root,
        holes=holes_created,
        creator_agent="unknown",
        environment_name="unknown",
        partial_mdl_cost=0.0,
    )


def fill_hole_in_fragment(
    fragment: ExpressionFragment,
    hole_index: int,
    fill_value: int,
) -> ExprNode:
    """
    Fill hole at hole_index with a CONST(fill_value) node.
    Returns the completed expression if all holes are filled,
    otherwise a partially-filled fragment.
    """
    result = copy.deepcopy(fragment.root)
    hole = fragment.holes[hole_index]
    node = _get_node_at_path(result, hole.position_path)
    if node is not None:
        node.value = fill_value
    return result


def _find_const_paths(
    node: ExprNode,
    current_path: List[int],
    result: List[List[int]],
) -> None:
    """Recursively find paths to all CONST nodes."""
    if node.node_type == NodeType.CONST:
        result.append(list(current_path))
        return
    if node.left is not None:
        _find_const_paths(node.left, current_path + [0], result)
    if node.right is not None:
        _find_const_paths(node.right, current_path + [1], result)


def _get_node_at_path(root: ExprNode, path: List[int]) -> Optional[ExprNode]:
    """Navigate to the node at the given path from root."""
    node = root
    for step in path:
        if step == 0:
            if node.left is None:
                return None
            node = node.left
        elif step == 1:
            if node.right is None:
                return None
            node = node.right
        else:
            return None
    return node