"""
LiteratureMatcher — Unified novelty detection against mathematical literature.

Combines:
  1. Behavioral embedding similarity (ExpressionDatabase)
  2. OEIS sequence search (for integer sequences)
  3. A combined novelty score

The matching pipeline:
  Given a discovered expression and its evaluated sequence:
  
  Step 1: OEIS search
    If the expression produces integer outputs, search OEIS for
    the integer sequence. If found with high confidence → not novel.
  
  Step 2: Embedding search
    Compute behavioral embedding, search registry for nearest known.
    If nearest distance < 0.1 → known equivalent.
  
  Step 3: Combine scores
    Novelty = min(oeis_score, embedding_score)
    (both must indicate novelty for the finding to be novel)
  
  Step 4: Generate recommendation
    Route to appropriate action based on combined novelty score.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.novelty.oeis_client import OEISClient, OEISResult
from ouroboros.novelty.embedder import BehavioralEmbedder
from ouroboros.novelty.registry import EmbeddingRegistry, RegistrySearchResult


@dataclass
class LiteratureMatch:
    """One match from the literature search."""
    source: str         # "oeis", "registry", "ouroboros_kb"
    match_id: str       # OEIS A-number or expression hash
    match_name: str
    match_expression: str
    confidence: float   # 0-1, how confident we are this is the same thing
    distance: float     # semantic distance (0=same, 1=different)
    notes: str


@dataclass
class LiteratureSearchResult:
    """Complete result of searching the mathematical literature."""
    query_expression: str
    query_sequence: List[float]

    # OEIS result
    oeis_result: Optional[OEISResult]
    oeis_match_confidence: float   # 0-1

    # Embedding result
    registry_result: Optional[RegistrySearchResult]

    # Combined analysis
    all_matches: List[LiteratureMatch]
    combined_novelty_score: float   # 0-1 (0=known, 1=novel)
    novelty_category: str
    recommendation: str             # what to do with this discovery

    def is_novel(self) -> bool:
        return self.combined_novelty_score > 0.4

    def should_route_to_expert(self) -> bool:
        return self.combined_novelty_score > 0.6

    def summary(self) -> str:
        lines = [
            f"LITERATURE SEARCH: {self.query_expression[:60]}",
            f"  Combined novelty: {self.combined_novelty_score:.3f} ({self.novelty_category})",
            f"  Recommendation: {self.recommendation}",
        ]
        if self.oeis_result and self.oeis_result.found:
            lines.append(f"  OEIS: {self.oeis_result.oeis_id} — {self.oeis_result.name}")
        if self.registry_result and self.registry_result.nearest_known:
            nn = self.registry_result.nearest_known
            lines.append(f"  Registry: {nn.name} (dist={self.registry_result.nearest_distance:.3f})")
        if not self.all_matches:
            lines.append("  No close matches found in literature")
        return "\n".join(lines)


class LiteratureMatcher:
    """
    Unified literature matching for OUROBOROS discoveries.

    Usage:
        matcher = LiteratureMatcher()
        result = matcher.match(expr, observations)
        if result.should_route_to_expert():
            print(f"NOVEL FINDING: {result.query_expression}")
            print(result.summary())
    """

    def __init__(
        self,
        oeis_cache_path: str = "results/oeis_cache.db",
        registry_path: str = "results/novelty_registry.json",
        use_oeis: bool = True,
        verbose: bool = False,
    ):
        self._oeis = OEISClient(cache_path=oeis_cache_path) if use_oeis else None
        self._registry = EmbeddingRegistry(registry_path=registry_path)
        self._embedder = BehavioralEmbedder()
        self._verbose = verbose
        self._n_novel_found = 0
        self._n_total_queries = 0

    def match(
        self,
        expr,
        observations: List[float],
        alphabet_size: int = None,
    ) -> LiteratureSearchResult:
        """
        Search mathematical literature for matches to a discovered expression.
        """
        self._n_total_queries += 1
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)

        # Step 1: Generate predictions from the expression
        predictions = []
        for t in range(min(len(observations), 20)):
            try:
                pred = expr.evaluate(t, list(observations[:t]), {})
                if math.isfinite(pred):
                    predictions.append(pred)
                else:
                    predictions.append(0.0)
            except Exception:
                predictions.append(0.0)

        all_matches = []
        oeis_result = None
        oeis_confidence = 0.0

        # Step 2: OEIS search (for integer-like sequences)
        if self._oeis is not None:
            int_preds = [int(round(p)) for p in predictions[:8]]
            if all(abs(int(round(p)) - p) < 0.5 for p in predictions[:8]):
                oeis_result = self._oeis.search_sequence(int_preds)
                if oeis_result.found:
                    oeis_confidence = self._compute_oeis_confidence(
                        int_preds, oeis_result.example_values
                    )
                    all_matches.append(LiteratureMatch(
                        source="oeis",
                        match_id=oeis_result.oeis_id or "",
                        match_name=oeis_result.name or "",
                        match_expression=oeis_result.formula or "",
                        confidence=oeis_confidence,
                        distance=1.0 - oeis_confidence,
                        notes=f"OEIS match: {oeis_result.oeis_id}",
                    ))
                    if self._verbose:
                        print(f"  OEIS: {oeis_result.description_str()}")

        # Step 3: Registry embedding search
        registry_result = self._registry.query(expr, verbose=self._verbose)
        embedding_distance = registry_result.nearest_distance

        if registry_result.nearest_known and embedding_distance < 0.3:
            nn = registry_result.nearest_known
            all_matches.append(LiteratureMatch(
                source="registry",
                match_id=nn.oeis_id or nn.name,
                match_name=nn.name,
                match_expression=nn.expression_str,
                confidence=1.0 - embedding_distance,
                distance=embedding_distance,
                notes=f"Registry match: {nn.domain}",
            ))

        # Step 4: Combine novelty scores
        oeis_novelty = 1.0 - oeis_confidence  # 1.0 if not in OEIS
        embedding_novelty = registry_result.novelty_score

        # Both must indicate novelty for the combined score to be high
        # Use harmonic mean: penalizes cases where one says novel but other doesn't
        if oeis_result is not None and oeis_result.found:
            combined = 2 * (oeis_novelty * embedding_novelty) / max(oeis_novelty + embedding_novelty, 1e-10)
        else:
            # No OEIS result → rely on embedding novelty
            combined = embedding_novelty * 0.7  # discount due to uncertainty

        novelty_category = self._categorize(combined)
        recommendation = self._recommend(combined, all_matches)

        if combined > 0.5:
            self._n_novel_found += 1

        return LiteratureSearchResult(
            query_expression=expr_str,
            query_sequence=list(predictions[:20]),
            oeis_result=oeis_result,
            oeis_match_confidence=oeis_confidence,
            registry_result=registry_result,
            all_matches=sorted(all_matches, key=lambda m: -m.confidence),
            combined_novelty_score=combined,
            novelty_category=novelty_category,
            recommendation=recommendation,
        )

    def _compute_oeis_confidence(
        self,
        query_terms: List[int],
        oeis_terms: List[int],
    ) -> float:
        """How well do the query terms match the OEIS terms?"""
        if not oeis_terms:
            return 0.0
        n = min(len(query_terms), len(oeis_terms))
        if n == 0:
            return 0.0
        matches = sum(1 for q, o in zip(query_terms[:n], oeis_terms[:n]) if q == o)
        return matches / n

    def _categorize(self, score: float) -> str:
        if score < 0.15:
            return "known"
        if score < 0.35:
            return "variant_of_known"
        if score < 0.55:
            return "potentially_novel"
        if score < 0.75:
            return "likely_novel"
        return "route_to_mathematician"

    def _recommend(self, score: float, matches: List[LiteratureMatch]) -> str:
        if score < 0.15:
            if matches:
                return f"Known result: {matches[0].match_name}"
            return "Routine rediscovery — skip"
        if score < 0.35:
            return "Interesting variant — log and continue"
        if score < 0.55:
            return "Potentially novel — increase search effort, verify on more data"
        if score < 0.75:
            return "Likely novel — generate formal proof, compare with arXiv"
        return "⭐ ROUTE TO MATHEMATICIAN — this may be a new result"

    @property
    def stats(self) -> dict:
        return {
            "total_queries": self._n_total_queries,
            "novel_flagged": self._n_novel_found,
            "novel_rate": self._n_novel_found / max(self._n_total_queries, 1),
            "oeis_stats": self._oeis.stats if self._oeis else {},
        }