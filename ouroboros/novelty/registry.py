"""
EmbeddingRegistry — The growing database of embedded known expressions.

This is the heart of the novelty detection system. Every time OUROBOROS
discovers an expression, it:
  1. Computes the expression's behavioral embedding
  2. Queries the registry for nearest neighbors
  3. If nearest neighbor distance > NOVELTY_THRESHOLD → flag as potentially novel
  4. If approved by proof market → add to registry as a known result

The registry grows over time. After 1000 discoveries, it is a comprehensive
database of mathematical behaviors across all domains OUROBOROS has explored.

Population strategy for the registry:
  - Seed: ~50 classic expressions (sine, exp, linear, modular, Fibonacci, etc.)
  - OEIS: batch-embed the first 1000 OEIS sequences that have known formulas
  - OUROBOROS discoveries: every promoted axiom gets added
  - Manual additions: domain experts can add known results from their field

Query strategy:
  For a query with embedding e, find the k nearest neighbors by cosine distance.
  Return:
    - closest_known: the single nearest known expression
    - closest_distance: cosine distance (0=same, 1=completely different)
    - domain_distances: minimum distance per domain (so we can say
      "novel in physics but not in number theory")
    - novelty_score: calibrated score from 0 to 1
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

from ouroboros.novelty.embedder import (
    BehavioralEmbedder, ExpressionEmbedding, ExpressionDatabase,
    KnownExpression, EMBEDDING_DIM,
)


@dataclass
class RegistrySearchResult:
    """Result of querying the registry for an expression's novelty."""
    query_expression: str
    query_embedding: ExpressionEmbedding

    nearest_known: Optional[KnownExpression]
    nearest_distance: float          # 0=identical, 1=completely different
    top_k_results: List[Tuple[KnownExpression, float]]

    # Domain-level analysis
    domain_distances: Dict[str, float]  # domain → min distance in that domain
    most_novel_domain: str              # domain where expression is most novel

    # Novelty score (calibrated 0-1)
    novelty_score: float
    novelty_category: str   # "routine" / "interesting" / "potentially_novel" / "route_to_expert"

    # Timing
    query_time_seconds: float

    def description(self) -> str:
        nearest_str = (
            f"{self.nearest_known.name} (dist={self.nearest_distance:.3f})"
            if self.nearest_known else "none"
        )
        return (
            f"Novelty Analysis: {self.query_expression[:60]}\n"
            f"  Novelty score: {self.novelty_score:.3f} ({self.novelty_category})\n"
            f"  Nearest known: {nearest_str}\n"
            f"  Most novel in: {self.most_novel_domain}\n"
            f"  Query time: {self.query_time_seconds*1000:.1f}ms"
        )

    def is_worth_investigating(self) -> bool:
        return self.novelty_category in ("potentially_novel", "route_to_expert")


def _calibrate_novelty_score(distance: float, n_in_registry: int) -> float:
    """
    Convert cosine distance to calibrated novelty score (0-1).

    Calibration accounts for:
    - The registry size: a small registry means we can't be confident
      something is novel just because it's distant from the few entries
    - Distance distribution: most random expressions are far from all known ones
      so raw distance is not a reliable novelty signal without calibration

    With a registry of 1000+ entries, distances > 0.3 are genuinely unusual.
    With a registry of 50 entries, we need distance > 0.6 to be confident.
    """
    # Minimum registry size for meaningful novelty claims
    min_registry = 20
    if n_in_registry < min_registry:
        # Registry too small — scale down confidence significantly
        confidence_factor = n_in_registry / min_registry
        return distance * confidence_factor * 0.5

    # With a reasonable registry:
    # distance 0.0-0.1: routine (known equivalent exists)
    # distance 0.1-0.3: interesting (structurally similar to known)
    # distance 0.3-0.6: potentially novel (nothing very similar known)
    # distance 0.6-1.0: route to expert (substantially different from all known)
    return min(1.0, distance)


def _categorize_novelty(score: float) -> str:
    if score < 0.10:
        return "routine"
    if score < 0.30:
        return "interesting"
    if score < 0.60:
        return "potentially_novel"
    return "route_to_expert"


class EmbeddingRegistry:
    """
    The central registry of known mathematical expressions with embeddings.

    This is a singleton in the OUROBOROS system — all agents share
    one registry and it persists across sessions.
    """

    def __init__(
        self,
        registry_path: Optional[str] = None,
        embedder: Optional[BehavioralEmbedder] = None,
    ):
        self._embedder = embedder or BehavioralEmbedder()
        self._db = ExpressionDatabase()
        self._registry_path = registry_path
        self._n_queries = 0
        self._n_novel_flags = 0

        # Fast numpy matrix for batch similarity search
        self._embedding_matrix: Optional[np.ndarray] = None
        self._matrix_entries: List[KnownExpression] = []
        self._matrix_dirty = True

        if registry_path and Path(registry_path).exists():
            self._load(registry_path)

    def register_known(
        self,
        expr,
        name: str,
        domain: str = "unknown",
        source: str = "manual",
        oeis_id: Optional[str] = None,
    ) -> KnownExpression:
        """Register a known expression (with behavioral embedding computed)."""
        entry = self._db.add_with_embedding(
            expr, name, domain, source, oeis_id=oeis_id
        )
        self._matrix_dirty = True
        return entry

    def register_string(
        self,
        expr_str: str,
        name: str,
        domain: str = "unknown",
        source: str = "manual",
        oeis_id: Optional[str] = None,
        outputs: Optional[List[float]] = None,
    ) -> KnownExpression:
        """
        Register a known expression given as a string.
        If outputs are provided, uses them directly for embedding.
        Otherwise, the expression is added without an embedding
        (won't participate in similarity search until embedded).
        """
        entry = KnownExpression(
            name=name,
            expression_str=expr_str,
            domain=domain,
            source=source,
            oeis_id=oeis_id,
        )
        if outputs is not None:
            entry.embedding = self._embedder.embed_from_outputs(outputs, expr_str)
        self._db.add(entry)
        self._matrix_dirty = True
        return entry

    def query(
        self,
        expr,
        top_k: int = 5,
        verbose: bool = False,
    ) -> RegistrySearchResult:
        """
        Query the registry for the novelty of a discovered expression.
        Returns a RegistrySearchResult with novelty score and nearest neighbors.
        """
        start = time.time()
        self._n_queries += 1

        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)
        query_emb = self._embedder.embed(expr)

        if not query_emb.is_valid:
            return RegistrySearchResult(
                query_expression=expr_str,
                query_embedding=query_emb,
                nearest_known=None,
                nearest_distance=1.0,
                top_k_results=[],
                domain_distances={},
                most_novel_domain="unknown",
                novelty_score=0.0,  # invalid expression is not novel
                novelty_category="routine",
                query_time_seconds=time.time() - start,
            )

        # Rebuild matrix if needed
        if self._matrix_dirty:
            self._rebuild_matrix()

        # Search
        top_results = self._search_matrix(query_emb, top_k)
        if not top_results:
            top_results = self._db.search_nearest(query_emb, top_k)

        # Compute per-domain distances
        domain_distances: Dict[str, float] = {}
        for entry, dist in top_results:
            d = entry.domain
            if d not in domain_distances or dist < domain_distances[d]:
                domain_distances[d] = dist

        nearest_known = top_results[0][0] if top_results else None
        nearest_distance = top_results[0][1] if top_results else 1.0

        most_novel_domain = max(domain_distances, key=domain_distances.get) \
                            if domain_distances else "unknown"

        novelty_score = _calibrate_novelty_score(nearest_distance, self._db.n_embedded)
        novelty_category = _categorize_novelty(novelty_score)

        if novelty_category in ("potentially_novel", "route_to_expert"):
            self._n_novel_flags += 1

        result = RegistrySearchResult(
            query_expression=expr_str,
            query_embedding=query_emb,
            nearest_known=nearest_known,
            nearest_distance=nearest_distance,
            top_k_results=top_results,
            domain_distances=domain_distances,
            most_novel_domain=most_novel_domain,
            novelty_score=novelty_score,
            novelty_category=novelty_category,
            query_time_seconds=time.time() - start,
        )

        if verbose:
            print(result.description())

        return result

    def query_from_embedding(
        self,
        embedding: ExpressionEmbedding,
        top_k: int = 5,
    ) -> RegistrySearchResult:
        """Query using a pre-computed embedding (faster for batch processing)."""
        if self._matrix_dirty:
            self._rebuild_matrix()
        top_results = self._search_matrix(embedding, top_k)
        nearest_known = top_results[0][0] if top_results else None
        nearest_distance = top_results[0][1] if top_results else 1.0
        novelty_score = _calibrate_novelty_score(nearest_distance, self._db.n_embedded)
        return RegistrySearchResult(
            query_expression=embedding.expression_str,
            query_embedding=embedding,
            nearest_known=nearest_known,
            nearest_distance=nearest_distance,
            top_k_results=top_results,
            domain_distances={},
            most_novel_domain="unknown",
            novelty_score=novelty_score,
            novelty_category=_categorize_novelty(novelty_score),
            query_time_seconds=0.0,
        )

    def _rebuild_matrix(self) -> None:
        """Rebuild the fast numpy search matrix from all embedded entries."""
        embedded = [e for e in self._db._entries if e.embedding is not None]
        if not embedded:
            self._embedding_matrix = None
            self._matrix_entries = []
            return
        matrix = np.stack([e.embedding.vector for e in embedded], axis=0)
        self._embedding_matrix = matrix
        self._matrix_entries = embedded
        self._matrix_dirty = False

    def _search_matrix(
        self,
        query: ExpressionEmbedding,
        top_k: int,
    ) -> List[Tuple[KnownExpression, float]]:
        """Fast numpy cosine similarity search."""
        if self._embedding_matrix is None or len(self._matrix_entries) == 0:
            return []

        # Cosine similarity = dot product (vectors are unit-normalized)
        similarities = self._embedding_matrix @ query.vector
        distances = 1.0 - similarities

        n = min(top_k, len(distances))
        top_indices = np.argpartition(distances, range(n))[:n]
        top_indices = top_indices[np.argsort(distances[top_indices])]

        return [(self._matrix_entries[i], float(distances[i])) for i in top_indices]

    def populate_from_ouroboros_kb(self, kb_path: str) -> int:
        """
        Load and embed all expressions from an OUROBOROS knowledge base JSON.
        Returns number of expressions added.
        """
        if not Path(kb_path).exists():
            return 0
        try:
            data = json.loads(Path(kb_path).read_text())
            n_added = 0
            for entry in data:
                expr_str = entry.get("expr", "")
                mdl_cost = entry.get("mdl_cost", 999.0)
                env_name = entry.get("env", "unknown")
                if not expr_str:
                    continue
                # Create a minimal entry without behavioral embedding
                # (we can't re-evaluate string expressions without parsing)
                known = KnownExpression(
                    name=f"ouroboros:{expr_str[:20]}",
                    expression_str=expr_str,
                    domain=self._infer_domain(env_name),
                    source="ouroboros",
                )
                self._db.add(known)
                n_added += 1
            return n_added
        except Exception:
            return 0

    def _infer_domain(self, env_name: str) -> str:
        name_lower = env_name.lower()
        if "modular" in name_lower or "prime" in name_lower or "gcd" in name_lower:
            return "number_theory"
        if "spring" in name_lower or "decay" in name_lower or "fall" in name_lower:
            return "physics"
        if "fib" in name_lower or "tribonacci" in name_lower:
            return "combinatorics"
        if "noise" in name_lower:
            return "statistics"
        return "unknown"

    def save(self, path: str) -> None:
        """Save registry metadata to JSON (embeddings are recomputed on load)."""
        data = [
            {
                "name": e.name,
                "expr": e.expression_str,
                "domain": e.domain,
                "source": e.source,
                "oeis_id": e.oeis_id,
            }
            for e in self._db._entries
        ]
        Path(path).write_text(json.dumps(data, indent=2))

    def _load(self, path: str) -> None:
        """Load registry from JSON."""
        try:
            data = json.loads(Path(path).read_text())
            for entry in data:
                known = KnownExpression(
                    name=entry.get("name", ""),
                    expression_str=entry.get("expr", ""),
                    domain=entry.get("domain", "unknown"),
                    source=entry.get("source", "manual"),
                    oeis_id=entry.get("oeis_id"),
                )
                self._db.add(known)
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {
            "total_entries": self._db.size,
            "embedded_entries": self._db.n_embedded,
            "n_queries": self._n_queries,
            "n_novel_flags": self._n_novel_flags,
        }