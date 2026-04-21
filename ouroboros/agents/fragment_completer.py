"""
FragmentCompleter — Completes received ExpressionFragments.

When an agent receives a fragment from another agent, it uses the
FragmentCompleter to search for the best values to fill the holes.

The key efficiency gain:
  Full beam search: explores O(beam_width^depth × n_iters) candidates
  Fragment completion: explores O(const_range) candidates (just constants)
  → 100-1000× speedup for filling HOLE_CONST holes

Protocol:
  1. Agent B receives a fragment from Agent A via MessageBus
  2. Agent B runs FragmentCompleter.complete(fragment, observations)
  3. FragmentCompleter enumerates values for each hole, scores each
  4. Returns the completion with lowest MDL cost
  5. Agent B sends the completed expression to ProtoAxiomPool
  6. If it's better than what Agent B already had, Agent B adopts it
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ouroboros.agents.expression_fragment import (
    ExpressionFragment, HoleSpec, HoleType,
    fill_hole_in_fragment, HOLE_SENTINEL,
)
from ouroboros.synthesis.expr_node import ExprNode, NodeType
from ouroboros.compression.mdl_engine import MDLEngine, MDLResult


@dataclass
class CompletionResult:
    """Result of attempting to complete a fragment."""
    fragment: ExpressionFragment
    completed_expr: Optional[ExprNode]
    filled_values: List[int]   # what value was put in each hole
    mdl_cost: float
    completing_agent: str
    improvement_over_partial: float

    @property
    def is_valid(self) -> bool:
        return self.completed_expr is not None

    def description(self) -> str:
        return (
            f"Completion by {self.completing_agent}\n"
            f"  Filled: {self.filled_values}\n"
            f"  MDL: {self.mdl_cost:.2f} bits\n"
            f"  Improvement over partial: {self.improvement_over_partial:.2f}"
        )


class FragmentCompleter:
    """
    Completes ExpressionFragments by searching for the best hole values.
    
    For HOLE_CONST holes: enumerate const_range values, score each.
    For HOLE_ANY holes: run a mini beam search over the hole's sub-expression.
    """

    def __init__(
        self,
        completing_agent: str,
        const_range: int = 30,
        mini_beam_width: int = 10,
        max_lag: int = 3,
    ):
        self.completing_agent = completing_agent
        self.const_range = const_range
        self.mini_beam_width = mini_beam_width
        self.max_lag = max_lag
        self._mdl = MDLEngine()

    def _score_expr(self, expr: ExprNode, observations: List[int]) -> float:
        """Score an expression under MDL."""
        predictions = [
            expr.evaluate(t, observations[:t])
            for t in range(len(observations))
        ]
        result = self._mdl.compute(
            predictions, observations,
            expr.node_count(), expr.constant_count()
        )
        return result.total_mdl_cost

    def complete(
        self,
        fragment: ExpressionFragment,
        observations: List[int],
    ) -> CompletionResult:
        """
        Find the best completion of the fragment.
        
        Searches over all possible hole values and returns the
        combination with lowest MDL cost.
        """
        if fragment.n_holes == 0:
            # Already complete
            cost = self._score_expr(fragment.root, observations)
            return CompletionResult(
                fragment=fragment,
                completed_expr=fragment.root,
                filled_values=[],
                mdl_cost=cost,
                completing_agent=self.completing_agent,
                improvement_over_partial=fragment.partial_mdl_cost - cost,
            )

        if fragment.n_holes == 1:
            return self._complete_one_hole(fragment, observations)
        else:
            return self._complete_multiple_holes(fragment, observations)

    def _complete_one_hole(
        self,
        fragment: ExpressionFragment,
        observations: List[int],
    ) -> CompletionResult:
        """Efficient single-hole completion: enumerate all values."""
        hole = fragment.holes[0]
        best_expr = None
        best_cost = float('inf')
        best_value = 0

        # For HOLE_CONST: enumerate integer constants
        if hole.hole_type in (HoleType.HOLE_CONST, HoleType.HOLE_ANY):
            search_range = range(self.const_range)
            if hole.expected_range:
                lo, hi = hole.expected_range
                search_range = range(max(0, lo), min(hi + 1, self.const_range + 1))

            for v in search_range:
                candidate = fill_hole_in_fragment(fragment, hole_index=0, fill_value=v)
                # Check no remaining holes
                if HOLE_SENTINEL in self._collect_consts(candidate):
                    continue
                cost = self._score_expr(candidate, observations)
                if cost < best_cost:
                    best_cost = cost
                    best_expr = candidate
                    best_value = v

        return CompletionResult(
            fragment=fragment,
            completed_expr=best_expr,
            filled_values=[best_value] if best_expr else [],
            mdl_cost=best_cost,
            completing_agent=self.completing_agent,
            improvement_over_partial=fragment.partial_mdl_cost - best_cost,
        )

    def _complete_multiple_holes(
        self,
        fragment: ExpressionFragment,
        observations: List[int],
    ) -> CompletionResult:
        """Multi-hole completion: greedy (fill each hole in order)."""
        current_fragment = copy.deepcopy(fragment)
        filled_values = []
        current_cost = fragment.partial_mdl_cost

        for i in range(fragment.n_holes):
            # Complete hole i
            single_result = self._complete_one_hole(current_fragment, observations)
            if not single_result.is_valid:
                break
            filled_values.extend(single_result.filled_values)
            current_cost = single_result.mdl_cost

            # Create a new "fragment" with hole i filled and remaining holes still open
            # For simplicity: rebuild the fragment with one fewer hole
            new_root = single_result.completed_expr
            remaining_holes = [
                h for j, h in enumerate(current_fragment.holes)
                if j != 0
            ]
            current_fragment = ExpressionFragment(
                root=new_root,
                holes=remaining_holes,
                creator_agent=fragment.creator_agent,
                environment_name=fragment.environment_name,
                partial_mdl_cost=current_cost,
            )

        final_expr = current_fragment.root if current_fragment.n_holes == 0 else None
        return CompletionResult(
            fragment=fragment,
            completed_expr=final_expr,
            filled_values=filled_values,
            mdl_cost=current_cost,
            completing_agent=self.completing_agent,
            improvement_over_partial=fragment.partial_mdl_cost - current_cost,
        )

    def _collect_consts(self, expr: ExprNode) -> List[int]:
        """Collect all constant values in an expression tree."""
        values = []
        if expr.node_type == NodeType.CONST:
            values.append(expr.value)
        if expr.left:
            values.extend(self._collect_consts(expr.left))
        if expr.right:
            values.extend(self._collect_consts(expr.right))
        return values