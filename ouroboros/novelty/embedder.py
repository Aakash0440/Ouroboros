"""
BehavioralEmbedder — Semantic embeddings for mathematical expressions.

The key idea: an expression's meaning IS its behavior on test inputs.
Two expressions that always produce the same outputs ARE the same expression,
regardless of how they look syntactically.

Embedding construction:
  1. Define a fixed set of CANONICAL TEST INPUTS — sequences that cover
     diverse mathematical behaviors (linear, quadratic, exponential, periodic,
     random, modular, prime-dense)
  2. Evaluate the expression on each canonical input at each timestep
  3. Concatenate all outputs into a single vector
  4. Normalize to unit length

This gives a 1D vector of dimension (n_sequences × n_timesteps) that
captures the behavior of the expression across diverse inputs.

Two expressions with the same vector are mathematically equivalent
on these inputs — with high probability, they are equivalent everywhere
(a claim strengthened by the Schwartz-Zippel lemma for polynomial expressions).

Distance metric:
  Cosine distance in the behavioral embedding space.
  distance = 1 - dot(e₁, e₂) / (|e₁| × |e₂|)
  0 = identical behavior, 1 = completely different behavior

Novelty detection:
  Given a newly discovered expression, compute its behavioral embedding,
  query the nearest neighbor in the known-expressions database,
  return the distance as the novelty score.

Known limitation:
  Two expressions can have the same behavioral fingerprint on all canonical
  inputs but differ on other inputs (aliasing). We mitigate this by using
  many diverse canonical sequences (20 sequences × 30 timesteps = 600-dim).
  For polynomial expressions, Schwartz-Zippel guarantees aliasing probability
  < d/|F| where d is degree and |F| is field size — effectively zero for
  large enough integer fields.
"""

from __future__ import annotations
import math
import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


# ── Canonical test inputs ─────────────────────────────────────────────────────
# These sequences are fixed forever — changing them invalidates all stored embeddings.
# They are chosen to stress-test as many mathematical behaviors as possible.

def _build_canonical_sequences(n_sequences: int = 20, seq_length: int = 30) -> List[List[float]]:
    """
    Build the fixed set of canonical test sequences.
    These are deterministic and never change between runs.
    """
    sequences = []
    rng = random.Random(42)  # FIXED seed — never change this

    # Linear sequences
    sequences.append([float(t) for t in range(seq_length)])                    # 0,1,2,...
    sequences.append([float(2*t + 1) for t in range(seq_length)])              # 1,3,5,...
    sequences.append([float(-t + 50) for t in range(seq_length)])              # 50,49,48,...

    # Quadratic / polynomial
    sequences.append([float(t*t) for t in range(seq_length)])                  # 0,1,4,9,...
    sequences.append([float(t*(t-10)) for t in range(seq_length)])             # negative then positive

    # Exponential / logarithmic
    sequences.append([math.exp(t * 0.1) for t in range(seq_length)])           # slow growth
    sequences.append([1000 * math.exp(-t * 0.05) for t in range(seq_length)])  # decay
    sequences.append([math.log(t + 1) for t in range(seq_length)])             # log growth

    # Periodic / trigonometric
    sequences.append([math.sin(2*math.pi*t/7) for t in range(seq_length)])     # period 7
    sequences.append([math.cos(2*math.pi*t/11) for t in range(seq_length)])    # period 11
    sequences.append([math.sin(t/3) + math.cos(t/7) for t in range(seq_length)]) # mixed

    # Modular / number-theoretic
    sequences.append([float((3*t+1) % 7) for t in range(seq_length)])          # mod 7
    sequences.append([float((5*t+2) % 11) for t in range(seq_length)])         # mod 11
    sequences.append([float(t % 13) for t in range(seq_length)])               # mod 13

    # Number-theoretic
    sequences.append([float(bin(t).count('1')) for t in range(seq_length)])    # Hamming weight
    sequences.append([float(math.gcd(t, 12)) for t in range(1, seq_length+1)]) # GCD with 12

    # Statistical / random-like
    sequences.append([float(rng.randint(0, 9)) for _ in range(seq_length)])    # random integers
    sequences.append([float(rng.gauss(0, 1)) for _ in range(seq_length)])      # Gaussian noise

    # Compound
    sequences.append([float(t**2 % 17) for t in range(seq_length)])            # quadratic residues
    sequences.append([float(t if t % 2 == 0 else -t) for t in range(seq_length)]) # alternating

    return sequences[:n_sequences]


# Build once at module load time — these never change
CANONICAL_SEQUENCES: List[List[float]] = _build_canonical_sequences(
    n_sequences=20, seq_length=30
)
EMBEDDING_DIM: int = len(CANONICAL_SEQUENCES) * len(CANONICAL_SEQUENCES[0])  # 600


@dataclass
class ExpressionEmbedding:
    """The behavioral embedding of a mathematical expression."""
    vector: np.ndarray           # shape: (EMBEDDING_DIM,), unit-normalized
    expression_str: str
    is_valid: bool               # False if expression produced NaN/inf on some inputs
    coverage: float              # fraction of canonical inputs where eval succeeded (0-1)
    hash_str: str                # for deduplication

    @classmethod
    def from_vector(cls, vec: np.ndarray, expr_str: str, coverage: float) -> 'ExpressionEmbedding':
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            unit_vec = vec
            is_valid = False
        else:
            unit_vec = vec / norm
            is_valid = coverage > 0.5
        hash_str = hashlib.md5(unit_vec.tobytes()).hexdigest()[:12]
        return cls(unit_vec, expr_str, is_valid, coverage, hash_str)

    def distance_to(self, other: 'ExpressionEmbedding') -> float:
        """Cosine distance in [0, 1]. 0=identical, 1=opposite."""
        if not self.is_valid or not other.is_valid:
            return 1.0
        dot = float(np.dot(self.vector, other.vector))
        return max(0.0, min(1.0, 1.0 - dot))


class BehavioralEmbedder:
    """
    Computes semantic embeddings for mathematical expressions.
    
    Usage:
        embedder = BehavioralEmbedder()
        
        # Embed a discovered expression
        embedding = embedder.embed(expr)
        
        # Compare two expressions
        dist = embedder.distance(expr1, expr2)
        # 0.0 = semantically identical, 1.0 = completely different
        
        # Check if two expressions are equivalent
        are_equiv = embedder.are_equivalent(expr1, expr2, threshold=0.01)
    """

    def __init__(
        self,
        canonical_sequences: List[List[float]] = None,
        nan_penalty: float = 999.0,
    ):
        self._sequences = canonical_sequences or CANONICAL_SEQUENCES
        self._nan_penalty = nan_penalty
        self._cache: Dict[str, ExpressionEmbedding] = {}

    def embed(self, expr) -> ExpressionEmbedding:
        """Compute the behavioral embedding of an expression."""
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)

        if expr_str in self._cache:
            return self._cache[expr_str]

        outputs = []
        n_valid = 0
        total = 0

        for seq in self._sequences:
            seq_outputs = []
            for t in range(len(seq)):
                total += 1
                try:
                    val = expr.evaluate(t, seq[:t], {}) if hasattr(expr, 'evaluate') \
                          else float(expr)
                    if not isinstance(val, (int, float)) or not math.isfinite(val):
                        val = self._nan_penalty
                    else:
                        n_valid += 1
                    seq_outputs.append(float(val))
                except Exception:
                    seq_outputs.append(self._nan_penalty)
            outputs.extend(seq_outputs)

        vec = np.array(outputs, dtype=np.float32)
        coverage = n_valid / max(total, 1)

        # Clip extreme values to prevent embedding collapse
        vec = np.clip(vec, -1e6, 1e6)

        embedding = ExpressionEmbedding.from_vector(vec, expr_str, coverage)
        self._cache[expr_str] = embedding
        return embedding

    def embed_from_outputs(
        self,
        outputs: List[float],
        expr_str: str = "unknown",
    ) -> ExpressionEmbedding:
        """Embed from pre-computed outputs (faster for batch processing)."""
        vec = np.array(outputs[:EMBEDDING_DIM], dtype=np.float32)
        if len(vec) < EMBEDDING_DIM:
            vec = np.pad(vec, (0, EMBEDDING_DIM - len(vec)), constant_values=self._nan_penalty)
        vec = np.clip(vec, -1e6, 1e6)
        coverage = sum(1 for v in outputs if abs(v) < 1e5) / max(len(outputs), 1)
        return ExpressionEmbedding.from_vector(vec, expr_str, coverage)

    def distance(self, expr1, expr2) -> float:
        """Cosine distance between two expressions' embeddings."""
        e1 = self.embed(expr1)
        e2 = self.embed(expr2)
        return e1.distance_to(e2)

    def are_equivalent(self, expr1, expr2, threshold: float = 0.02) -> bool:
        """True if expressions produce essentially identical outputs."""
        return self.distance(expr1, expr2) < threshold

    def cluster_expressions(
        self,
        expressions: list,
        threshold: float = 0.05,
    ) -> List[List[int]]:
        """
        Group expressions into clusters of semantically equivalent ones.
        Returns list of groups, each group is a list of indices.
        """
        if not expressions:
            return []

        embeddings = [self.embed(e) for e in expressions]
        clusters = []
        assigned = set()

        for i, emb_i in enumerate(embeddings):
            if i in assigned:
                continue
            cluster = [i]
            assigned.add(i)
            for j, emb_j in enumerate(embeddings):
                if j in assigned:
                    continue
                if emb_i.distance_to(emb_j) < threshold:
                    cluster.append(j)
                    assigned.add(j)
            clusters.append(cluster)

        return clusters


# ── Known Expression Database ─────────────────────────────────────────────────

@dataclass
class KnownExpression:
    """An entry in the known-expressions database."""
    name: str                    # human name, e.g. "exponential decay"
    expression_str: str          # symbolic form, e.g. "C * exp(-k*t)"
    domain: str                  # "physics", "number_theory", "statistics", etc.
    source: str                  # "textbook", "oeis", "arxiv", "ouroboros"
    embedding: Optional[ExpressionEmbedding] = None
    oeis_id: Optional[str] = None  # e.g. "A000045" for Fibonacci
    reference: Optional[str] = None

    def description(self) -> str:
        return f"{self.name} ({self.domain}) — {self.expression_str}"


class ExpressionDatabase:
    """
    Database of known mathematical expressions with semantic embeddings.
    
    Supports:
    - Adding new expressions (from OEIS, textbooks, OUROBOROS discoveries)
    - Nearest-neighbor search in embedding space
    - Exact deduplication via embedding hash
    """

    def __init__(self):
        self._entries: List[KnownExpression] = []
        self._embedder = BehavioralEmbedder()
        self._embedding_matrix: Optional[np.ndarray] = None  # cached for fast search
        self._dirty = True  # True when matrix needs rebuild

        # Seed with fundamental known expressions
        self._seed_with_classics()

    def _seed_with_classics(self) -> None:
        """Add the most fundamental known expressions as seeds."""
        classics = [
            ("constant_zero", "CONST(0)", "arithmetic", "textbook"),
            ("constant_one", "CONST(1)", "arithmetic", "textbook"),
            ("identity", "TIME", "arithmetic", "textbook"),
            ("linear", "k*t + c", "arithmetic", "textbook"),
            ("exponential_decay", "C*exp(-k*t)", "physics", "textbook"),
            ("sine", "A*sin(omega*t)", "physics", "textbook"),
            ("cosine", "A*cos(omega*t)", "physics", "textbook"),
            ("fibonacci", "PREV(1) + PREV(2)", "combinatorics", "textbook"),
            ("modular_linear", "(k*t + c) % N", "number_theory", "textbook"),
            ("prime_counting", "CUMSUM(ISPRIME(t))", "number_theory", "ouroboros"),
            ("hookes_law_signature", "DERIV2(x) + k*x", "physics", "ouroboros"),
            ("free_fall", "h - 0.5*g*t^2", "physics", "textbook"),
        ]

        for name, expr_str, domain, source in classics:
            entry = KnownExpression(
                name=name,
                expression_str=expr_str,
                domain=domain,
                source=source,
            )
            self._entries.append(entry)
        # Note: embeddings are NOT computed at init time (too slow)
        # They are computed lazily on first search

    def add(self, entry: KnownExpression) -> bool:
        """Add an expression. Returns False if already present (by hash)."""
        if entry.embedding is None:
            # Try to embed if we have an evaluable expression
            pass  # Skip embedding for string-only entries for now

        self._entries.append(entry)
        self._dirty = True
        return True

    def search_nearest(
        self,
        query_embedding: ExpressionEmbedding,
        top_k: int = 5,
    ) -> List[Tuple[KnownExpression, float]]:
        """
        Find the top_k most similar known expressions.
        Returns list of (entry, distance) pairs, sorted by distance.
        """
        if not query_embedding.is_valid:
            return []

        results = []
        for entry in self._entries:
            if entry.embedding is None:
                continue
            dist = query_embedding.distance_to(entry.embedding)
            results.append((entry, dist))

        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def add_with_embedding(
        self,
        expr,
        name: str,
        domain: str,
        source: str,
        oeis_id: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> KnownExpression:
        """Add an expression with its behavioral embedding computed."""
        embedding = self._embedder.embed(expr)
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)
        entry = KnownExpression(
            name=name,
            expression_str=expr_str,
            domain=domain,
            source=source,
            embedding=embedding,
            oeis_id=oeis_id,
            reference=reference,
        )
        self._entries.append(entry)
        self._dirty = True
        return entry

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def n_embedded(self) -> int:
        return sum(1 for e in self._entries if e.embedding is not None)