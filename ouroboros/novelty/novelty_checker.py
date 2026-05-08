"""
NoveltyChecker — scores how novel an OUROBOROS discovery is.

Novelty score in [0, 1]:
  0.0 = sequence is a well-known OEIS core sequence (e.g. Fibonacci)
  0.5 = sequence found in OEIS but not marked 'core'
  0.8 = sequence not found in OEIS (genuinely novel or too noisy to match)
  1.0 = sequence not found AND expression is structurally complex

Usage:
    checker = NoveltyChecker()
    score   = checker.score(expr, observations, predicted)
"""
from __future__ import annotations
from typing import List, Optional, Any
from ouroboros.novelty.oeis_client import OEISClient, OEISResult


class NoveltyChecker:

    def __init__(self, cache_path: str = "results/oeis_cache.db", verbose: bool = False):
        self._client = OEISClient(cache_path=cache_path, verbose=verbose)

    def score(
        self,
        expr: Any,
        observations: List[int],
        predicted: Optional[List[int]] = None,
    ) -> dict:
        """
        Returns dict with:
          novelty_score  : float in [0, 1]
          oeis_id        : str or None
          oeis_name      : str or None
          known          : bool
        """
        # Use predictions if available, else raw observations
        seq = predicted if predicted is not None else observations

        # OEIS needs positive integers — shift if needed
        shifted, offset = self._to_positive(seq[:12])

        result: OEISResult = self._client.search_sequence(shifted)

        if not result.found:
            # Not in OEIS — likely novel
            complexity = self._expr_complexity(expr)
            score = 0.8 + 0.2 * min(complexity / 5.0, 1.0)
            return {
                "novelty_score": round(score, 3),
                "oeis_id": None,
                "oeis_name": None,
                "known": False,
                "message": "Not found in OEIS",
            }

        if result.is_well_known:
            return {
                "novelty_score": 0.0,
                "oeis_id": result.oeis_id,
                "oeis_name": result.name,
                "known": True,
                "message": f"Core OEIS sequence: {result.oeis_id} ({result.name})",
            }

        return {
            "novelty_score": 0.5,
            "oeis_id": result.oeis_id,
            "oeis_name": result.name,
            "known": True,
            "message": f"Known OEIS sequence: {result.oeis_id} ({result.name})",
        }

    @staticmethod
    def _to_positive(seq: List[int]) -> tuple[List[int], int]:
        """Shift sequence so minimum value is 1 (OEIS convention)."""
        if not seq:
            return [], 0
        lo = min(seq)
        offset = 1 - lo if lo < 1 else 0
        return [v + offset for v in seq], offset

    @staticmethod
    def _expr_complexity(expr: Any) -> int:
        """Rough complexity: node count of expression tree."""
        if expr is None:
            return 0
        try:
            return expr.node_count()
        except Exception:
            return 1