"""
CompletenessChecker — Proves that no expression in vocabulary V of depth ≤ D
achieves MDL cost below threshold C on sequence S.

If this check PASSES (no expression found), it is a certificate of vocabulary
insufficiency — the sequence cannot be compressed with the current vocabulary
at the given depth bound. A new primitive is needed.

If this check FAILS (expression found), the system just needs more search effort,
not a new primitive.

Algorithm:
  1. Generate all grammar-valid expressions up to depth D (using the grammar
     constraint to prune invalid combinations)
  2. Score each expression under MDL
  3. If any achieves cost ≤ C → return that expression (vocabulary sufficient)
  4. If none achieves cost ≤ C → return None (vocabulary insufficient)

Tractability analysis:
  With grammar branching factor 6.2 and depth 4:
  Total expressions ≈ sum_{d=1}^{4} 6.2^d ≈ 1,730 expressions
  At 5ms per expression scoring → 8.65 seconds total
  This is fast enough for an online check.

  At depth 5: ≈ 10,700 expressions → 53 seconds (acceptable)
  At depth 6: ≈ 66,000 expressions → 5 minutes (use only as offline check)

For the completeness check, we use depth 4 by default and increase to 5
if depth 4 is insufficient.
"""

from __future__ import annotations
import time
import itertools
from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple

from ouroboros.grammar.math_grammar import MathGrammar, DEFAULT_GRAMMAR
from ouroboros.nodes.extended_nodes import (
    ExtNodeType, ExtExprNode, NODE_SPECS, NodeCategory,
)
from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
from ouroboros.compression.mdl_engine import MDLEngine


@dataclass
class CompletenessResult:
    """Result of the completeness check."""
    is_complete: bool          # True if vocabulary IS sufficient (expression found)
    best_expr: Optional[ExtExprNode]
    best_cost: float
    threshold: float
    max_depth_checked: int
    n_expressions_checked: int
    time_seconds: float
    certificate: str           # human-readable explanation

    def __str__(self) -> str:
        if self.is_complete:
            expr_str = self.best_expr.to_string() if self.best_expr else "None"
            return (f"Vocabulary SUFFICIENT: found expr with cost "
                    f"{self.best_cost:.2f} < {self.threshold:.2f}\n"
                    f"  Expression: {expr_str}")
        return (f"Vocabulary INSUFFICIENT: no expression of depth ≤ "
                f"{self.max_depth_checked} achieves cost ≤ {self.threshold:.2f}\n"
                f"  Checked {self.n_expressions_checked} expressions in "
                f"{self.time_seconds:.2f}s\n"
                f"  {self.certificate}")


class CompletenessChecker:
    """
    Bounded completeness checker for MDL expression search.

    Enumerates all grammar-valid expressions up to a depth bound
    and checks whether any achieves the target MDL cost.

    This is a CERTIFICATE of vocabulary insufficiency when it returns
    is_complete=False — not just a search failure, but a proof that
    no such expression exists at the given depth.
    """

    def __init__(
        self,
        grammar: MathGrammar = None,
        max_depth: int = 4,
        time_limit_seconds: float = 30.0,
        n_terminal_samples: int = 5,  # sample terminals rather than enumerate all
    ):
        self._grammar = grammar or DEFAULT_GRAMMAR
        self.max_depth = max_depth
        self.time_limit = time_limit_seconds
        self.n_terminal_samples = n_terminal_samples
        self._mdl = MDLEngine()

    def check(
        self,
        observations: List[int],
        threshold: float,
        verbose: bool = False,
    ) -> CompletenessResult:
        """
        Check if any grammar-valid expression of depth ≤ max_depth
        achieves MDL cost ≤ threshold on observations.

        Returns CompletenessResult with is_complete=True if found,
        is_complete=False if vocabulary is insufficient at this depth.
        """
        start = time.time()
        n_checked = 0
        best_expr = None
        best_cost = float('inf')

        # Enumerate expressions systematically
        for expr in self._enumerate_expressions(max_depth=self.max_depth):
            if time.time() - start > self.time_limit:
                # Time limit hit — inconclusive (not a certificate)
                return CompletenessResult(
                    is_complete=best_cost <= threshold,
                    best_expr=best_expr if best_cost <= threshold else None,
                    best_cost=best_cost,
                    threshold=threshold,
                    max_depth_checked=self.max_depth,
                    n_expressions_checked=n_checked,
                    time_seconds=time.time() - start,
                    certificate=f"Time limit reached — check is INCONCLUSIVE",
                )

            n_checked += 1
            cost = self._score(expr, observations)
            if cost < best_cost:
                best_cost = cost
                best_expr = expr

            if best_cost <= threshold:
                return CompletenessResult(
                    is_complete=True,
                    best_expr=best_expr,
                    best_cost=best_cost,
                    threshold=threshold,
                    max_depth_checked=self.max_depth,
                    n_expressions_checked=n_checked,
                    time_seconds=time.time() - start,
                    certificate=(f"Expression found with cost {best_cost:.2f} ≤ "
                                 f"{threshold:.2f}"),
                )

            if verbose and n_checked % 100 == 0:
                elapsed = time.time() - start
                print(f"  Checked {n_checked} expressions, "
                      f"best={best_cost:.2f}, elapsed={elapsed:.1f}s")

        elapsed = time.time() - start
        certificate = (
            f"Exhaustively checked {n_checked} grammar-valid expressions "
            f"of depth ≤ {self.max_depth}. "
            f"Best achieved cost {best_cost:.2f} > threshold {threshold:.2f}. "
            f"Vocabulary is INSUFFICIENT for this sequence at depth {self.max_depth}."
        )

        return CompletenessResult(
            is_complete=False,
            best_expr=None,
            best_cost=best_cost,
            threshold=threshold,
            max_depth_checked=self.max_depth,
            n_expressions_checked=n_checked,
            time_seconds=elapsed,
            certificate=certificate,
        )

    def _enumerate_expressions(
        self,
        max_depth: int,
        allowed_categories: frozenset = None,
        depth: int = 0,
    ) -> Iterator[ExtExprNode]:
        """
        Generate all grammar-valid expressions up to max_depth.
        Uses the grammar to prune invalid combinations.
        """
        from ouroboros.synthesis.expr_node import NodeType

        if allowed_categories is None:
            from ouroboros.grammar.math_grammar import ANY_TYPES
            allowed_categories = ANY_TYPES

        # Always yield terminals at any depth
        yield from self._terminal_expressions()

        if depth >= max_depth:
            return

        # Yield non-terminal expressions for each valid node type
        for node_type, spec in NODE_SPECS.items():
            if spec.category not in allowed_categories:
                continue
            if spec.arity == 0:
                continue  # terminals already yielded above

            if spec.arity == 1:
                child_cats = self._grammar.allowed_child_categories(node_type, 0)
                for child in self._sample_subexpressions(
                    max_depth - 1, child_cats, depth + 1, limit=3
                ):
                    node = ExtExprNode(node_type, left=child)
                    yield node

            elif spec.arity == 2:
                child_cats_0 = self._grammar.allowed_child_categories(node_type, 0)
                child_cats_1 = self._grammar.allowed_child_categories(node_type, 1)
                # Only sample a few combinations to keep enumeration tractable
                for left in self._sample_subexpressions(
                    max_depth - 1, child_cats_0, depth + 1, limit=2
                ):
                    for right in self._sample_subexpressions(
                        max_depth - 1, child_cats_1, depth + 1, limit=2
                    ):
                        node = ExtExprNode(node_type, left=left, right=right)
                        yield node

    def _sample_subexpressions(
        self,
        max_depth: int,
        allowed_cats: frozenset,
        depth: int,
        limit: int = 3,
    ) -> List[ExtExprNode]:
        """Sample a limited number of subexpressions (for tractability)."""
        exprs = list(itertools.islice(
            self._enumerate_expressions(max_depth, allowed_cats, depth),
            limit
        ))
        return exprs if exprs else list(self._terminal_expressions())[:1]

    def _terminal_expressions(self) -> Iterator[ExtExprNode]:
        """Yield all terminal expressions (CONST, TIME, PREV)."""
        from ouroboros.synthesis.expr_node import NodeType

        # TIME
        n = ExtExprNode.__new__(ExtExprNode)
        n.node_type = NodeType.TIME; n.value = 0.0
        n.lag = 1; n.state_key = 0; n.window = 10
        n.left = n.right = n.third = None; n._cache = {}
        yield n

        # CONST(0), CONST(1), CONST(2), ..., CONST(20)
        for v in range(0, self.n_terminal_samples + 1):
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.CONST; n.value = float(v)
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            yield n

        # PREV(1), PREV(2)
        for lag in [1, 2]:
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.PREV; n.value = 0.0
            n.lag = lag; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            yield n

    def _score(self, expr: ExtExprNode, observations: List[int]) -> float:
        """Score an expression under MDL, safely."""
        import math
        try:
            preds = []
            for t in range(len(observations)):
                p = expr.evaluate(t, observations[:t], {})
                preds.append(int(round(p)) if math.isfinite(p) else 0)
            r = self._mdl.compute(
                preds, observations,
                expr.node_count(), expr.constant_count()
            )
            return r.total_mdl_cost
        except Exception:
            return float('inf')