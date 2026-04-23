"""
EnvironmentClassifier — Identify the mathematical family of an observation sequence.

Instead of searching all 60 node types blindly, first classify the sequence:
  PERIODIC     → use Calculus + Transcendental nodes (SIN, COS, FFT_AMP)
  EXPONENTIAL  → use Calculus + EWMA + EXP nodes
  RECURRENT    → use PREV nodes, Berlekamp-Massey, CUMSUM
  STATISTICAL  → use MEAN_WIN, VAR_WIN, STD_WIN, CORR
  NUMBER_THEOR → use GCD, ISPRIME, TOTIENT, FLOOR, MOD
  MONOTONE     → use CUMSUM, RUNNING_MAX, DERIV
  MIXED        → use all categories (no classification possible)

Classification uses simple statistics computed directly from the sequence:
  - Autocorrelation at lag 1, 7, 14 (periodicity signal)
  - Derivative variance (exponential vs polynomial vs constant)
  - Run-length statistics (recurrence signal)
  - Entropy (structure vs noise)
  - Monotonicity fraction (trend signal)

No training required — pure statistical heuristics.
These are fast (O(n)) and reliable enough to narrow the search.
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

from ouroboros.nodes.extended_nodes import NodeCategory


class MathFamily(Enum):
    """Mathematical family of an observation sequence."""
    PERIODIC     = auto()   # sin/cos patterns, FFT has clear peaks
    EXPONENTIAL  = auto()   # exp growth/decay, log-linear derivative
    RECURRENT    = auto()   # linear recurrence, PREV-expressible
    STATISTICAL  = auto()   # rolling statistics structure
    NUMBER_THEOR = auto()   # modular, primality, GCD patterns
    MONOTONE     = auto()   # consistently increasing or decreasing
    RANDOM       = auto()   # high entropy, no clear structure
    MIXED        = auto()   # multiple families, use all


@dataclass
class ClassificationResult:
    """Result of classifying an observation sequence."""
    primary_family: MathFamily
    family_scores: Dict[MathFamily, float]   # score for each family (0-1)
    recommended_categories: List[NodeCategory]
    confidence: float   # 0-1, how confident the classifier is
    
    # Sequence statistics used for classification
    entropy: float
    autocorr_lag1: float
    autocorr_lag7: float
    deriv_variance: float
    monotonicity: float
    unique_ratio: float   # unique values / total — low = discrete/modular

    def description(self) -> str:
        scores_str = ", ".join(
            f"{f.name}={s:.2f}" for f, s in
            sorted(self.family_scores.items(), key=lambda x: -x[1])
        )
        return (
            f"Classification: {self.primary_family.name} "
            f"(confidence={self.confidence:.2f})\n"
            f"  Scores: {scores_str}\n"
            f"  Entropy: {self.entropy:.3f}, "
            f"  AutoCorr(1): {self.autocorr_lag1:.3f}, "
            f"  AutoCorr(7): {self.autocorr_lag7:.3f}\n"
            f"  Recommended: {[c.name for c in self.recommended_categories]}"
        )


# ── Category recommendations per family ───────────────────────────────────────

FAMILY_TO_CATEGORIES: Dict[MathFamily, List[NodeCategory]] = {
    MathFamily.PERIODIC: [
        NodeCategory.TRANSCEND,    # SIN, COS — primary
        NodeCategory.TRANSFORM,    # FFT_AMP, AUTOCORR — for period finding
        NodeCategory.CALCULUS,     # DERIV (phase), CUMSUM (integral)
        NodeCategory.ARITHMETIC,   # ADD, MUL for combining
        NodeCategory.TERMINAL,     # always needed
    ],
    MathFamily.EXPONENTIAL: [
        NodeCategory.TRANSCEND,    # EXP, LOG — primary
        NodeCategory.CALCULUS,     # DERIV (growth rate), EWMA (smoothing)
        NodeCategory.ARITHMETIC,
        NodeCategory.TERMINAL,
    ],
    MathFamily.RECURRENT: [
        NodeCategory.TERMINAL,     # PREV nodes — primary
        NodeCategory.ARITHMETIC,   # ADD, MUL for recurrence coefficients
        NodeCategory.NUMBER,       # MOD for modular recurrences
        NodeCategory.CALCULUS,     # CUMSUM for prefix sums
    ],
    MathFamily.STATISTICAL: [
        NodeCategory.STATISTICAL,  # MEAN_WIN, VAR_WIN, CORR — primary
        NodeCategory.LOGICAL,      # THRESHOLD, ZSCORE conditions
        NodeCategory.CALCULUS,     # EWMA, rolling
        NodeCategory.ARITHMETIC,
        NodeCategory.TERMINAL,
    ],
    MathFamily.NUMBER_THEOR: [
        NodeCategory.NUMBER,       # GCD, ISPRIME, TOTIENT, FLOOR — primary
        NodeCategory.ARITHMETIC,   # MOD already in original set
        NodeCategory.LOGICAL,      # THRESHOLD, COMPARE for conditionals
        NodeCategory.TERMINAL,
    ],
    MathFamily.MONOTONE: [
        NodeCategory.CALCULUS,     # RUNNING_MAX, CUMSUM, DERIV — primary
        NodeCategory.ARITHMETIC,
        NodeCategory.STATISTICAL,  # MEAN_WIN for trends
        NodeCategory.TERMINAL,
    ],
    MathFamily.RANDOM: [
        NodeCategory.STATISTICAL,  # Can't compress, but try statistics
        NodeCategory.LOGICAL,
        NodeCategory.TERMINAL,
    ],
    MathFamily.MIXED: [
        # All categories — no restriction
        NodeCategory.TERMINAL,
        NodeCategory.ARITHMETIC,
        NodeCategory.CALCULUS,
        NodeCategory.STATISTICAL,
        NodeCategory.LOGICAL,
        NodeCategory.TRANSFORM,
        NodeCategory.NUMBER,
        NodeCategory.MEMORY,
        NodeCategory.TRANSCEND,
    ],
}


class EnvironmentClassifier:
    """
    Classifies observation sequences into mathematical families.
    
    All methods are O(n) and require no training.
    Classification takes < 1ms for sequences of length 1000.
    """

    def classify(
        self,
        observations: List[float],
        verbose: bool = False,
    ) -> ClassificationResult:
        """Classify the observation sequence into a mathematical family."""
        if len(observations) < 10:
            return self._make_result(MathFamily.MIXED, {}, observations)

        # Compute sequence statistics
        stats = self._compute_stats(observations)
        
        # Score each family
        scores = {
            MathFamily.PERIODIC:     self._score_periodic(stats, observations),
            MathFamily.EXPONENTIAL:  self._score_exponential(stats, observations),
            MathFamily.RECURRENT:    self._score_recurrent(stats, observations),
            MathFamily.STATISTICAL:  self._score_statistical(stats, observations),
            MathFamily.NUMBER_THEOR: self._score_number_theoretic(stats, observations),
            MathFamily.MONOTONE:     self._score_monotone(stats, observations),
            MathFamily.RANDOM:       self._score_random(stats, observations),
        }

        # Find primary family
        primary = max(scores, key=scores.get)
        max_score = scores[primary]

        # Confidence: gap between top and second family
        sorted_scores = sorted(scores.values(), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
        confidence = min(1.0, gap * 2.0)

        # If random or low confidence → MIXED
        if primary == MathFamily.RANDOM or confidence < 0.2:
            primary = MathFamily.MIXED

        recommended = FAMILY_TO_CATEGORIES.get(primary, FAMILY_TO_CATEGORIES[MathFamily.MIXED])

        if verbose:
            result = ClassificationResult(
                primary_family=primary,
                family_scores=scores,
                recommended_categories=recommended,
                confidence=confidence,
                **stats
            )
            print(result.description())
            return result

        return ClassificationResult(
            primary_family=primary,
            family_scores=scores,
            recommended_categories=recommended,
            confidence=confidence,
            **stats
        )

    def _compute_stats(self, obs: List[float]) -> dict:
        """Compute O(n) statistics from the observation sequence."""
        n = len(obs)
        
        # Entropy (normalized)
        from collections import Counter
        int_obs = [int(round(v)) for v in obs]
        counts = Counter(int_obs)
        entropy = -sum((c/n) * math.log2(c/n) for c in counts.values() if c > 0)
        max_entropy = math.log2(max(len(counts), 2))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

        # Autocorrelation at lag 1 and 7
        def autocorr(lag: int) -> float:
            if len(obs) <= lag: return 0.0
            m = statistics.mean(obs)
            num = sum((obs[i] - m) * (obs[i - lag] - m) for i in range(lag, n))
            denom = sum((v - m)**2 for v in obs)
            return num / max(denom, 1e-10)

        ac1 = autocorr(1)
        ac7 = autocorr(min(7, n//3))

        # Derivative variance (rate of change)
        derivs = [obs[i] - obs[i-1] for i in range(1, n)]
        deriv_var = statistics.variance(derivs) if len(derivs) > 1 else 0.0

        # Monotonicity (fraction of steps going in same direction)
        if derivs:
            pos = sum(1 for d in derivs if d > 0)
            monotone_frac = max(pos / len(derivs), 1 - pos / len(derivs))
        else:
            monotone_frac = 0.5

        # Unique ratio
        unique_ratio = len(set(int_obs)) / n

        return {
            "entropy": norm_entropy,
            "autocorr_lag1": ac1,
            "autocorr_lag7": ac7,
            "deriv_variance": deriv_var,
            "monotonicity": monotone_frac,
            "unique_ratio": unique_ratio,
        }

    def _score_periodic(self, stats: dict, obs: List[float]) -> float:
        """Higher score if sequence shows periodic patterns."""
        score = 0.0
        # High autocorrelation at any lag is a periodicity signal
        score += min(1.0, abs(stats["autocorr_lag7"]) * 2)
        score += min(0.5, abs(stats["autocorr_lag1"]) * 1)
        # Low entropy → repeated patterns
        score += (1.0 - stats["entropy"]) * 0.3
        # Small unique ratio for discrete periodic sequences
        if stats["unique_ratio"] < 0.2:
            score += 0.3
        return min(1.0, score)

    def _score_exponential(self, stats: dict, obs: List[float]) -> float:
        """Higher score if sequence grows/decays exponentially."""
        score = 0.0
        # High monotonicity + high derivative variance → exponential
        if stats["monotonicity"] > 0.8:
            score += 0.4
        # High derivative variance relative to values
        if obs and max(abs(v) for v in obs) > 0:
            rel_var = math.sqrt(stats["deriv_variance"]) / max(abs(v) for v in obs)
            if rel_var > 0.1:
                score += 0.3
        score += stats["monotonicity"] * 0.3
        return min(1.0, score)

    def _score_recurrent(self, stats: dict, obs: List[float]) -> float:
        """Higher score if sequence satisfies a linear recurrence."""
        from ouroboros.emergence.recurrence_detector import berlekamp_massey_mod
        # Run BM as a quick check
        int_obs = [int(round(v)) for v in obs[:100]]
        mod_candidates = [v for v in range(2, 20) if all(0 <= x < v for x in int_obs)]
        if mod_candidates:
            m = max(mod_candidates)
            result = berlekamp_massey_mod(int_obs, m)
            if result is not None and len(result) <= 5:
                return 0.9
        # Fall back to autocorrelation check
        return abs(stats["autocorr_lag1"]) * 0.5

    def _score_statistical(self, stats: dict, obs: List[float]) -> float:
        """Higher score if rolling statistics show structure."""
        # Medium autocorrelation + medium entropy → statistical structure
        score = 0.0
        if 0.2 < abs(stats["autocorr_lag1"]) < 0.9:
            score += 0.4
        if 0.3 < stats["entropy"] < 0.8:
            score += 0.3
        if stats["unique_ratio"] > 0.3:
            score += 0.2
        return min(1.0, score)

    def _score_number_theoretic(self, stats: dict, obs: List[float]) -> float:
        """Higher score if sequence has modular/number-theoretic structure."""
        int_obs = [int(round(v)) for v in obs]
        score = 0.0
        # Low unique ratio with bounded values → modular
        max_val = max(int_obs) if int_obs else 1
        if max_val > 0 and stats["unique_ratio"] < 0.3 and max_val < 100:
            score += 0.5
        # Very low entropy with small alphabet
        if stats["entropy"] < 0.5 and max_val < 20:
            score += 0.3
        return min(1.0, score)

    def _score_monotone(self, stats: dict, obs: List[float]) -> float:
        return min(1.0, (stats["monotonicity"] - 0.5) * 4)

    def _score_random(self, stats: dict, obs: List[float]) -> float:
        return stats["entropy"] * 0.8 + (1 - abs(stats["autocorr_lag1"])) * 0.2

    def _make_result(self, family, scores, obs):
        stats = self._compute_stats(obs) if obs else {
            "entropy": 1.0, "autocorr_lag1": 0.0, "autocorr_lag7": 0.0,
            "deriv_variance": 0.0, "monotonicity": 0.5, "unique_ratio": 1.0,
        }
        return ClassificationResult(
            primary_family=family,
            family_scores=scores or {family: 1.0},
            recommended_categories=FAMILY_TO_CATEGORIES.get(family, [NodeCategory.TERMINAL]),
            confidence=1.0,
            **stats
        )