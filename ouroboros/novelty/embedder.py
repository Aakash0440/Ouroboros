"""
BehavioralEmbedder — Semantic embeddings for mathematical expressions.

The key idea: an expression's meaning IS its behavior on test inputs.
Two expressions that always produce the same outputs ARE the same expression,
regardless of how they look syntactically.

Embedding construction (v2 — structural fingerprint):
  The v1 embedding evaluated expr(t) on canonical sequences and concatenated
  the raw outputs. This made it:
    - Scale-blind:  3*t and t produce parallel vectors → after L2 norm they
                    are identical (cos distance = 0), so 3*t looks "known".
    - Phase-blind:  sin(t+φ) and sin(t) differ only by a rigid rotation of
                    the output vector → after norm they are also identical.
    - Frequency-conflating: sin(t) and sin(2t) differ in frequency but both
                    produce unit-norm sinusoidal vectors → small cosine distance.

  v2 fixes this with a STRUCTURAL fingerprint:
    Feature 1 — Shape descriptors (scale-invariant by construction):
      Zero-mean + unit-std normalise each output sequence before stacking.
      This makes 3*t and t produce identical normalised shapes → correctly
      treated as the same law. Scale is NOT part of the law identity.

    Feature 2 — Autocorrelation spectrum (phase-invariant, frequency-sensitive):
      Compute the autocorrelation of the normalised output at lags 1..L.
      ACF is shift-invariant: sin(t+φ) and sin(t) have the same ACF.
      But sin(2t) has ACF that decays twice as fast → different fingerprint.
      This is what distinguishes frequency while ignoring phase.

    Feature 3 — Frequency power spectrum (via DFT magnitudes):
      |FFT(output)|² is phase-invariant and frequency-discriminating.
      sin(t) peaks at frequency 1/(2π), sin(2t) peaks at frequency 1/π.
      Captures periodicity structure that ACF alone may miss.

    Feature 4 — Structural moments (skewness, kurtosis, monotonicity):
      Higher-order statistics that distinguish polynomial from exponential,
      modular from smooth, etc.

  All four feature blocks are concatenated and L2-normalised to produce
  the final embedding vector.

Distance metric:
  Cosine distance in the structural embedding space.
  distance = 1 - dot(e₁, e₂) / (|e₁| × |e₂|)
  0 = identical structure, 1 = completely different structure

Novelty detection:
  Given a newly discovered expression, compute its structural embedding,
  query the nearest neighbor in the known-expressions database,
  return the distance as the novelty score.
"""

from __future__ import annotations
import math
import hashlib
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


# ── Canonical evaluation inputs ───────────────────────────────────────────────
# These are fixed t-values at which we evaluate every expression.
# We use a single timeline (not sequences as context) because ExprNode.evaluate
# only depends on t, not on the surrounding sequence.

def _build_canonical_sequences(n_sequences: int = 20, seq_length: int = 30) -> List[List[float]]:
    """
    Build the fixed set of canonical test sequences.
    These are deterministic and never change between runs.
    """
    sequences = []
    rng = random.Random(42)  # FIXED seed — never change this

    # Linear sequences
    sequences.append([float(t) for t in range(seq_length)])
    sequences.append([float(2*t + 1) for t in range(seq_length)])
    sequences.append([float(-t + 50) for t in range(seq_length)])

    # Quadratic / polynomial
    sequences.append([float(t*t) for t in range(seq_length)])
    sequences.append([float(t*(t-10)) for t in range(seq_length)])

    # Exponential / logarithmic
    sequences.append([math.exp(t * 0.1) for t in range(seq_length)])
    sequences.append([1000 * math.exp(-t * 0.05) for t in range(seq_length)])
    sequences.append([math.log(t + 1) for t in range(seq_length)])

    # Periodic / trigonometric
    sequences.append([math.sin(2*math.pi*t/7) for t in range(seq_length)])
    sequences.append([math.cos(2*math.pi*t/11) for t in range(seq_length)])
    sequences.append([math.sin(t/3) + math.cos(t/7) for t in range(seq_length)])

    # Modular / number-theoretic
    sequences.append([float((3*t+1) % 7) for t in range(seq_length)])
    sequences.append([float((5*t+2) % 11) for t in range(seq_length)])
    sequences.append([float(t % 13) for t in range(seq_length)])

    # Number-theoretic
    sequences.append([float(bin(t).count('1')) for t in range(seq_length)])
    sequences.append([float(math.gcd(t, 12)) for t in range(1, seq_length+1)])

    # Statistical / random-like
    sequences.append([float(rng.randint(0, 9)) for _ in range(seq_length)])
    sequences.append([float(rng.gauss(0, 1)) for _ in range(seq_length)])

    # Compound
    sequences.append([float(t**2 % 17) for t in range(seq_length)])
    sequences.append([float(t if t % 2 == 0 else -t) for t in range(seq_length)])

    return sequences[:n_sequences]


CANONICAL_SEQUENCES: List[List[float]] = _build_canonical_sequences(
    n_sequences=20, seq_length=30
)

# v2: embedding dim is determined by structural features, not raw output size
# We compute it dynamically in _structural_fingerprint but export a constant
# for compatibility. Actual dim = n_seqs * (acf_lags + fft_bins + moments)
# = 20 * (15 + 15 + 4) = 20 * 34 = 680
_ACF_LAGS   = 15
_FFT_BINS   = 15
_N_MOMENTS  = 4
EMBEDDING_DIM: int = len(CANONICAL_SEQUENCES) * (_ACF_LAGS + _FFT_BINS * 3 + _N_MOMENTS)

# ── Structural fingerprint helpers ────────────────────────────────────────────

def _zscore_normalize(seq: List[float]) -> List[float]:
    """Zero-mean, unit-std normalise. Returns zeros if std < 1e-10 (constant)."""
    n = len(seq)
    if n == 0:
        return seq
    mean = sum(seq) / n
    var  = sum((x - mean)**2 for x in seq) / n
    std  = math.sqrt(var)
    if std < 1e-10:
        return [0.0] * n
    return [(x - mean) / std for x in seq]


def _acf(seq: List[float], max_lag: int) -> List[float]:
    """
    Autocorrelation function at lags 1..max_lag (lag-0 is always 1, skip it).
    Phase-invariant: sin(t+φ) and sin(t) have identical ACF.
    Frequency-sensitive: sin(t) and sin(2t) have different ACF decay rates.
    """
    n = len(seq)
    if n < 3:
        return [0.0] * max_lag
    mean = sum(seq) / n
    var  = sum((x - mean)**2 for x in seq) / n
    if var < 1e-10:
        return [0.0] * max_lag
    result = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            result.append(0.0)
            continue
        cov = sum((seq[i] - mean) * (seq[i - lag] - mean) for i in range(lag, n)) / n
        result.append(cov / var)
    return result


def _fft_magnitude(seq: List[float], n_bins: int) -> List[float]:
    """
    DFT magnitude spectrum (first n_bins positive frequencies).
    Phase-invariant: |FFT(sin(t+φ))| = |FFT(sin(t))|.
    Frequency-sensitive: sin(t) and sin(2t) have peaks at different bins.
    """
    n = len(seq)
    if n < 2:
        return [0.0] * n_bins
    # Compute DFT manually (no numpy.fft dependency issue)
    mags = []
    for k in range(min(n_bins, n // 2 + 1)):
        re = sum(seq[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
        im = sum(seq[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
        mags.append(math.sqrt(re**2 + im**2) / n)
    # Pad if needed
    while len(mags) < n_bins:
        mags.append(0.0)
    return mags[:n_bins]


def _structural_moments(seq: List[float]) -> List[float]:
    """
    4 scale-invariant shape moments on the z-scored sequence:
      [0] skewness            — asymmetry (linear vs exponential)
      [1] excess kurtosis     — tail weight (modular vs smooth)
      [2] monotonicity index  — fraction of steps in same direction
      [3] zero-crossing rate  — oscillation frequency proxy
    """
    n = len(seq)
    if n < 4:
        return [0.0, 0.0, 0.0, 0.0]

    # Skewness
    mean = sum(seq) / n
    std  = math.sqrt(sum((x - mean)**2 for x in seq) / n)
    if std < 1e-10:
        return [0.0, 0.0, 0.0, 0.0]
    skew = sum(((x - mean) / std)**3 for x in seq) / n

    # Excess kurtosis
    kurt = sum(((x - mean) / std)**4 for x in seq) / n - 3.0

    # Monotonicity: fraction of consecutive pairs going in the same direction
    # as the overall trend (sign of last - first)
    diffs = [seq[i+1] - seq[i] for i in range(n-1)]
    overall_sign = 1.0 if seq[-1] >= seq[0] else -1.0
    mono = sum(1 for d in diffs if d * overall_sign > 0) / max(len(diffs), 1)

    # Zero-crossing rate of the z-scored sequence
    zcr = sum(1 for i in range(1, n) if seq[i-1] * seq[i] < 0) / max(n - 1, 1)

    # Clamp to reasonable ranges
    skew = max(-10.0, min(10.0, skew))
    kurt = max(-10.0, min(10.0, kurt))

    return [skew, kurt, mono, zcr]

def _structural_fingerprint(outputs: List[float]) -> np.ndarray:
    normed = _zscore_normalize(outputs)

    acf_features = _acf(normed, _ACF_LAGS)
    fft_features = _fft_magnitude(normed, _FFT_BINS)
    mom_features = _structural_moments(normed)

    # Peak frequency feature: one-hot-ish vector marking the dominant FFT bin.
    # This is the single most discriminating feature for sin(t) vs sin(2t):
    # sin(t) peaks at bin ~2, sin(2t) peaks at bin ~4 (for seq_length=30).
    # Weight it 3x to ensure frequency differences dominate cosine distance.
    if max(fft_features) > 1e-10:
        peak_bin = fft_features.index(max(fft_features))
    else:
        peak_bin = 0
    peak_onehot = [0.0] * _FFT_BINS
    peak_onehot[peak_bin] = 3.0   # strong weight on dominant frequency

    combined = acf_features + fft_features + fft_features + peak_onehot + mom_features
    return np.array(combined, dtype=np.float32)

# ── Embedding classes ─────────────────────────────────────────────────────────

@dataclass
class ExpressionEmbedding:
    """The behavioral embedding of a mathematical expression."""
    vector: np.ndarray           # shape: (EMBEDDING_DIM,), unit-normalized
    expression_str: str
    is_valid: bool
    coverage: float
    hash_str: str

    @classmethod
    def from_vector(cls, vec: np.ndarray, expr_str: str, coverage: float) -> 'ExpressionEmbedding':
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            unit_vec = np.zeros_like(vec)
            is_valid = False
        else:
            unit_vec = vec / norm
            is_valid = coverage > 0.5
        hash_str = hashlib.md5(unit_vec.tobytes()).hexdigest()[:12]
        return cls(unit_vec, expr_str, is_valid, coverage, hash_str)

    def distance_to(self, other: 'ExpressionEmbedding') -> float:
        """Cosine distance in [0, 2]. 0=identical, 1=orthogonal, 2=opposite."""
        if not self.is_valid or not other.is_valid:
            return 1.0
        dot = float(np.dot(self.vector, other.vector))
        dot = max(-1.0, min(1.0, dot))
        return max(0.0, 1.0 - dot)


class BehavioralEmbedder:
    """
    Computes structural embeddings for mathematical expressions.

    v2: uses ACF + FFT magnitude + shape moments instead of raw outputs.
    This makes embeddings:
      - Scale-invariant:  3*t ≡ t (same law, different scale)
      - Phase-invariant:  sin(t+φ) ≡ sin(t) (same law, different phase)
      - Frequency-sensitive: sin(t) ≠ sin(2t) (different frequency = different law)
      - Composition-aware: fib%7 ≠ fib and ≠ mod7 (ambiguous novelty)
    """

    def __init__(
        self,
        canonical_sequences: List[List[float]] = None,
        nan_penalty: float = 999.0,
    ):
        self._sequences = canonical_sequences or CANONICAL_SEQUENCES
        self._nan_penalty = nan_penalty
        self._cache: Dict[str, ExpressionEmbedding] = {}

    def _evaluate_on_sequence(self, expr, seq: List[float]) -> List[float]:
        """Evaluate expr at each timestep, returning outputs."""
        outputs = []
        for t in range(len(seq)):
            try:
                # Try with full signature first, fall back to t-only
                if hasattr(expr, 'evaluate'):
                    try:
                        val = expr.evaluate(t, seq[:t], {})
                    except TypeError:
                        try:
                            val = expr.evaluate(t)
                        except TypeError:
                            val = float(expr)
                else:
                    val = float(expr)

                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    val = self._nan_penalty
            except Exception:
                val = self._nan_penalty
            outputs.append(float(val))
        return outputs

    def embed(self, expr) -> ExpressionEmbedding:
        """Compute the structural embedding of an expression."""
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)

        if expr_str in self._cache:
            return self._cache[expr_str]

        feature_blocks = []
        n_valid_seqs = 0

        for seq in self._sequences:
            outputs = self._evaluate_on_sequence(expr, seq)

            # Check validity
            n_valid = sum(1 for v in outputs if abs(v) < 1e5)
            if n_valid < len(outputs) // 2:
                # Too many nan/inf — use zero block for this sequence
                block = np.zeros(_ACF_LAGS + _FFT_BINS + _N_MOMENTS, dtype=np.float32)
            else:
                # Replace nan_penalty values with sequence mean for fingerprinting
                clean = [v if abs(v) < 1e5 else 0.0 for v in outputs]
                block = _structural_fingerprint(clean)
                n_valid_seqs += 1

            feature_blocks.append(block)

        vec = np.concatenate(feature_blocks)
        coverage = n_valid_seqs / max(len(self._sequences), 1)

        embedding = ExpressionEmbedding.from_vector(vec, expr_str, coverage)
        self._cache[expr_str] = embedding
        return embedding

    def embed_from_outputs(
        self,
        outputs: List[float],
        expr_str: str = "unknown",
    ) -> ExpressionEmbedding:
        """
        Embed from pre-computed raw outputs.
        Splits outputs into per-sequence blocks and applies structural fingerprint.
        """
        seq_len = len(self._sequences[0]) if self._sequences else 30
        n_seqs  = len(self._sequences)

        feature_blocks = []
        for i in range(n_seqs):
            start = i * seq_len
            end   = start + seq_len
            chunk = outputs[start:end] if start < len(outputs) else []
            if len(chunk) < seq_len:
                chunk = chunk + [0.0] * (seq_len - len(chunk))
            clean = [v if abs(v) < 1e5 else 0.0 for v in chunk]
            feature_blocks.append(_structural_fingerprint(clean))

        vec = np.concatenate(feature_blocks)
        coverage = sum(1 for v in outputs if abs(v) < 1e5) / max(len(outputs), 1)
        return ExpressionEmbedding.from_vector(vec, expr_str, coverage)

    def distance(self, expr1, expr2) -> float:
        """Cosine distance between two expressions' embeddings."""
        e1 = self.embed(expr1)
        e2 = self.embed(expr2)
        return e1.distance_to(e2)

    def are_equivalent(self, expr1, expr2, threshold: float = 0.05) -> bool:
        """True if expressions have essentially identical structural fingerprints."""
        return self.distance(expr1, expr2) < threshold

    def cluster_expressions(
        self,
        expressions: list,
        threshold: float = 0.10,
    ) -> List[List[int]]:
        """Group expressions into clusters of structurally equivalent ones."""
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
    name: str
    expression_str: str
    domain: str
    source: str
    embedding: Optional[ExpressionEmbedding] = None
    oeis_id: Optional[str] = None
    reference: Optional[str] = None

    def description(self) -> str:
        return f"{self.name} ({self.domain}) — {self.expression_str}"


class ExpressionDatabase:
    """
    Database of known mathematical expressions with structural embeddings.
    """

    def __init__(self):
        self._entries: List[KnownExpression] = []
        self._embedder = BehavioralEmbedder()
        self._embedding_matrix: Optional[np.ndarray] = None
        self._dirty = True
        self._seed_with_classics()

    def _seed_with_classics(self) -> None:
        classics = [
            ("constant_zero",        "CONST(0)",              "arithmetic",     "textbook"),
            ("constant_one",         "CONST(1)",              "arithmetic",     "textbook"),
            ("identity",             "TIME",                  "arithmetic",     "textbook"),
            ("linear",               "k*t + c",               "arithmetic",     "textbook"),
            ("exponential_decay",    "C*exp(-k*t)",           "physics",        "textbook"),
            ("sine",                 "A*sin(omega*t)",        "physics",        "textbook"),
            ("cosine",               "A*cos(omega*t)",        "physics",        "textbook"),
            ("fibonacci",            "PREV(1) + PREV(2)",     "combinatorics",  "textbook"),
            ("modular_linear",       "(k*t + c) % N",         "number_theory",  "textbook"),
            ("prime_counting",       "CUMSUM(ISPRIME(t))",    "number_theory",  "ouroboros"),
            ("hookes_law_signature", "DERIV2(x) + k*x",      "physics",        "ouroboros"),
            ("free_fall",            "h - 0.5*g*t^2",         "physics",        "textbook"),
        ]
        for name, expr_str, domain, source in classics:
            entry = KnownExpression(
                name=name, expression_str=expr_str,
                domain=domain, source=source,
            )
            self._entries.append(entry)

    def add(self, entry: KnownExpression) -> bool:
        self._entries.append(entry)
        self._dirty = True
        return True

    def search_nearest(
        self,
        query_embedding: ExpressionEmbedding,
        top_k: int = 5,
    ) -> List[Tuple[KnownExpression, float]]:
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
        embedding = self._embedder.embed(expr)
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)
        entry = KnownExpression(
            name=name, expression_str=expr_str, domain=domain, source=source,
            embedding=embedding, oeis_id=oeis_id, reference=reference,
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