"""
DerivativeAnalyzer — Analyzes sequences by their derivative structure.

Given an observation sequence (from any source — real data, agent prediction,
or environment generation), computes:
  1. First discrete derivative DERIV(seq)
  2. Second discrete derivative DERIV2(seq)
  3. Correlations between derivatives and the original
  4. Phase space plots (seq[t] vs DERIV(seq)[t]) — shape reveals the law
  5. Best-fit physics law from the correlation structure

This is the "law identification" step that runs AFTER an agent finds
an expression. The verifier asks: does this expression satisfy any
known physics law signature?

Independent of OUROBOROS agents — can analyze any time series.
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from ouroboros.physics.law_signature import (
    PhysicsLaw, SignatureTestResult, LawSignature,
    ALL_SIGNATURES, _deriv, _deriv2, _safe_corr,
)


@dataclass
class DerivativeProfile:
    """Complete derivative analysis of a sequence."""
    original: List[float]
    deriv1: List[float]
    deriv2: List[float]

    # Correlations
    corr_d1_orig: float    # CORR(DERIV, orig) — decay signature
    corr_d2_orig: float    # CORR(DERIV2, orig) — spring signature
    corr_d1_d1:   float    # CORR(DERIV, DERIV) — self-correlation

    # Statistics
    mean_d2: float         # mean second derivative — free fall signature
    cv_d2: float           # coefficient of variation of DERIV2
    d1_range: float        # range of first derivative — volatility
    monotonicity: float    # fraction of positive steps

    # Phase space analysis
    phase_radius_mean: float   # mean of sqrt(x² + ẋ²) — energy-like
    phase_radius_std: float    # std of energy — should be small for conservative systems

    def is_oscillatory(self) -> bool:
        """Returns True if sequence has strong oscillatory signature."""
        return self.corr_d2_orig < -0.7

    def is_exponential(self) -> bool:
        """Returns True if sequence has exponential decay/growth signature."""
        return self.corr_d1_orig < -0.7 and self.monotonicity > 0.7

    def is_constant_acceleration(self) -> bool:
        """Returns True if sequence has constant second derivative."""
        return self.cv_d2 < 0.1

    def summary(self) -> str:
        laws = []
        if self.is_oscillatory(): laws.append("OSCILLATORY (Hooke/SHM)")
        if self.is_exponential(): laws.append("EXPONENTIAL DECAY")
        if self.is_constant_acceleration(): laws.append("CONSTANT ACCEL (Free Fall)")
        if not laws: laws.append("UNCLASSIFIED")
        return (
            f"DerivativeProfile:\n"
            f"  CORR(DERIV2, orig): {self.corr_d2_orig:.4f}  "
            f"{'← Hooke/SHM!' if self.corr_d2_orig < -0.7 else ''}\n"
            f"  CORR(DERIV, orig):  {self.corr_d1_orig:.4f}  "
            f"{'← Exponential decay!' if self.corr_d1_orig < -0.7 else ''}\n"
            f"  CV(DERIV2):         {self.cv_d2:.4f}  "
            f"{'← Constant accel!' if self.cv_d2 < 0.1 else ''}\n"
            f"  Detected: {', '.join(laws)}"
        )


class DerivativeAnalyzer:
    """
    Analyzes the derivative structure of observation sequences to identify
    the underlying physical law.
    """

    def analyze(self, seq: List[float]) -> DerivativeProfile:
        """Compute the complete derivative profile of a sequence."""
        if len(seq) < 5:
            return self._empty_profile(seq)

        d1 = _deriv(seq)
        d2 = _deriv(d1)

        # Align lengths
        n_d2 = len(d2)
        orig_aligned = seq[2:2 + n_d2]

        corr_d2_orig = _safe_corr(d2, orig_aligned)
        corr_d1_orig = _safe_corr(d1, seq[1:1 + len(d1)])
        corr_d1_d1 = _safe_corr(d1[:-1], d1[1:]) if len(d1) > 2 else 0.0

        mean_d2 = statistics.mean(d2) if d2 else 0.0
        std_d2 = statistics.stdev(d2) if len(d2) > 1 else 0.0
        cv_d2 = std_d2 / max(abs(mean_d2), 1e-10)

        d1_range = max(d1) - min(d1) if d1 else 0.0
        pos_steps = sum(1 for v in d1 if v > 0)
        monotonicity = pos_steps / max(len(d1), 1)

        # Phase space: (x, ẋ) radius = sqrt(x² + ẋ²)
        phase_radii = [
            math.sqrt(x**2 + xdot**2)
            for x, xdot in zip(seq[1:], d1)
        ]
        phase_radius_mean = statistics.mean(phase_radii) if phase_radii else 0.0
        phase_radius_std = statistics.stdev(phase_radii) if len(phase_radii) > 1 else 0.0

        return DerivativeProfile(
            original=seq,
            deriv1=d1,
            deriv2=d2,
            corr_d1_orig=corr_d1_orig,
            corr_d2_orig=corr_d2_orig,
            corr_d1_d1=corr_d1_d1,
            mean_d2=mean_d2,
            cv_d2=cv_d2,
            d1_range=d1_range,
            monotonicity=monotonicity if monotonicity > 0.5 else 1.0 - monotonicity,
            phase_radius_mean=phase_radius_mean,
            phase_radius_std=phase_radius_std,
        )

    def _empty_profile(self, seq: List[float]) -> DerivativeProfile:
        return DerivativeProfile(
            original=seq, deriv1=[], deriv2=[],
            corr_d1_orig=0.0, corr_d2_orig=0.0, corr_d1_d1=0.0,
            mean_d2=0.0, cv_d2=1.0, d1_range=0.0, monotonicity=0.5,
            phase_radius_mean=0.0, phase_radius_std=0.0,
        )

    def identify_law(
        self,
        seq: List[float],
        verbose: bool = False,
    ) -> Tuple[PhysicsLaw, List[SignatureTestResult]]:
        """
        Test all known law signatures and return the best match.
        """
        results = []
        for sig in ALL_SIGNATURES:
            result = sig.test_function(seq)
            results.append(result)
            if verbose:
                print(f"  {result.description()}")

        # Find best passing signature
        passing = [r for r in results if r.passed]
        if passing:
            best = max(passing, key=lambda r: r.confidence)
            return best.law, results

        return PhysicsLaw.UNKNOWN, results


class PhysicsLawVerifier:
    """
    Verifies whether an OUROBOROS-discovered expression satisfies
    a physical law signature.
    
    Usage:
        verifier = PhysicsLawVerifier()
        law, results = verifier.verify(expr, observations)
        # law: PhysicsLaw enum value
        # results: list of SignatureTestResult for each law tested
    """

    def __init__(self):
        self._analyzer = DerivativeAnalyzer()

    def verify(
        self,
        expr,  # ExtExprNode
        observations: List[float],
        verbose: bool = False,
    ) -> Tuple[PhysicsLaw, List[SignatureTestResult]]:
        """
        Verify what physical law (if any) the expression satisfies.
        
        Steps:
        1. Evaluate expression to get predicted sequence
        2. Run derivative analysis on predictions
        3. Test all law signatures
        4. Return best match
        """
        # Evaluate expression
        try:
            predictions = []
            history = list(observations)
            for t in range(len(observations)):
                pred = expr.evaluate(t, observations[:t])
                if not isinstance(pred, (int, float)) or not math.isfinite(pred):
                    pred = 0.0
                predictions.append(float(pred))
        except Exception:
            predictions = [0.0] * len(observations)

        if verbose:
            print(f"\nPhysics verification on {len(predictions)} predictions")
            profile = self._analyzer.analyze(predictions)
            print(profile.summary())

        law, results = self._analyzer.identify_law(predictions, verbose=verbose)
        return law, results

    def verify_raw_sequence(
        self,
        observations: List[float],
        verbose: bool = False,
    ) -> Tuple[PhysicsLaw, List[SignatureTestResult]]:
        """
        Verify what physical law the raw observation sequence satisfies.
        Used to identify the environment type before search.
        """
        if verbose:
            profile = self._analyzer.analyze(observations)
            print(profile.summary())
        return self._analyzer.identify_law(observations, verbose=verbose)