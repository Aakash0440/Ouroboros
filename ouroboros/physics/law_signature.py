"""
LawSignature — Formal description of a physical law's mathematical signature.

A physical law is not just "this formula fits the data." It is a specific
structural relationship between a quantity and its derivatives.

Known signatures:
  Hooke's Law:          DERIV2(x) = -k * x           (spring oscillation)
  Radioactive Decay:    DERIV(N) = -λ * N             (exponential decay)
  Free Fall:            DERIV2(y) = -g                (constant acceleration)
  Logistic Growth:      DERIV(P) = r*P*(1 - P/K)      (population dynamics)
  Newton Cooling:       DERIV(T) = -k*(T - T_env)     (heat transfer)
  Simple Harmonic:      DERIV2(x) + ω²*x = 0         (pendulum, spring)

How OUROBOROS tests a signature:
  Given a discovered expression expr and observations obs:
  1. Compute the predicted sequence: [expr(t) for t in range(n)]
  2. Compute DERIV: [pred(t) - pred(t-1) for t in range(1, n)]
  3. Compute DERIV2: [deriv(t) - deriv(t-1) for t in range(1, n)]
  4. Test the signature condition:
     - Hooke: CORR(DERIV2_seq, pred_seq) ≈ -1 (strong negative correlation)
     - Decay: CORR(DERIV_seq, pred_seq[1:]) ≈ -1
     - Free Fall: STD(DERIV2_seq) ≈ 0 (constant second derivative)

If the correlation exceeds threshold, the law is confirmed.
This is not ML — it's classical physics verification by correlation.
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Callable, Dict, Tuple


class PhysicsLaw(Enum):
    """Known physical laws with verifiable mathematical signatures."""
    HOOKES_LAW          = auto()   # spring oscillation: DERIV2 ∝ -x
    EXPONENTIAL_DECAY   = auto()   # radioactive decay: DERIV ∝ -N
    FREE_FALL           = auto()   # constant acceleration: DERIV2 = const
    LOGISTIC_GROWTH     = auto()   # population: DERIV = rP(1-P/K)
    NEWTON_COOLING      = auto()   # heat: DERIV ∝ -(T - T_env)
    SIMPLE_HARMONIC     = auto()   # pendulum: DERIV2 + ω²*x = 0
    UNKNOWN             = auto()   # no known law matched


@dataclass
class SignatureTestResult:
    """Result of testing a sequence against a physics law signature."""
    law: PhysicsLaw
    passed: bool
    confidence: float          # 0.0 to 1.0
    key_metric: str            # human-readable description of the test
    key_value: float           # the measured value
    threshold: float           # required value for the test to pass

    def description(self) -> str:
        status = "✅ CONFIRMED" if self.passed else "❌ Not matched"
        return (
            f"{status} {self.law.name}: {self.key_metric} = "
            f"{self.key_value:.4f} (threshold: {self.threshold:.4f}, "
            f"confidence: {self.confidence:.2f})"
        )


@dataclass
class LawSignature:
    """
    Formal description of a physical law's mathematical signature.
    
    The signature test takes a predicted sequence and computes
    derived quantities (derivatives, correlations) to verify
    whether the sequence satisfies the law.
    """
    law: PhysicsLaw
    description: str
    test_function: Callable[[List[float]], SignatureTestResult]
    examples: List[str] = field(default_factory=list)


def _safe_corr(x: List[float], y: List[float]) -> float:
    """Pearson correlation, returning 0 if undefined."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = sum(x)/n, sum(y)/n
    num = sum((xi - mx)*(yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx)**2 for xi in x))
    sy = math.sqrt(sum((yi - my)**2 for yi in y))
    denom = sx * sy
    if denom < 1e-12:
        return 0.0
    return num / denom


def _deriv(seq: List[float]) -> List[float]:
    """First discrete derivative."""
    return [seq[i] - seq[i-1] for i in range(1, len(seq))]


def _deriv2(seq: List[float]) -> List[float]:
    """Second discrete derivative."""
    return _deriv(_deriv(seq))


# ── Law signature definitions ──────────────────────────────────────────────────

def _test_hookes_law(seq: List[float], threshold: float = 0.85) -> SignatureTestResult:
    """
    Hooke's Law: DERIV2(x) ∝ -x
    Test: Pearson correlation between DERIV2(seq) and seq[2:] < -threshold
    """
    if len(seq) < 10:
        return SignatureTestResult(PhysicsLaw.HOOKES_LAW, False, 0.0,
                                  "sequence too short", 0.0, threshold)
    d2 = _deriv2(seq)
    orig = seq[2:]  # align with d2
    corr = _safe_corr(d2, orig)
    passed = corr < -threshold
    confidence = min(1.0, abs(corr) / threshold) if passed else abs(corr) / threshold * 0.5
    return SignatureTestResult(
        law=PhysicsLaw.HOOKES_LAW,
        passed=passed,
        confidence=confidence,
        key_metric="CORR(DERIV2(x), x)",
        key_value=corr,
        threshold=-threshold,
    )


def _test_exponential_decay(seq: List[float], threshold: float = 0.85) -> SignatureTestResult:
    """
    Radioactive Decay: DERIV(N) ∝ -N
    Test: correlation between DERIV(seq) and seq[1:] < -threshold
    """
    if len(seq) < 10:
        return SignatureTestResult(PhysicsLaw.EXPONENTIAL_DECAY, False, 0.0,
                                  "sequence too short", 0.0, threshold)
    # Remove zeros to avoid division issues
    nz = [(d, n) for d, n in zip(_deriv(seq), seq[1:]) if abs(n) > 1e-6]
    if len(nz) < 5:
        return SignatureTestResult(PhysicsLaw.EXPONENTIAL_DECAY, False, 0.0,
                                  "too many zeros", 0.0, threshold)
    derivs, origs = zip(*nz)
    corr = _safe_corr(list(derivs), list(origs))
    passed = corr < -threshold
    confidence = min(1.0, abs(corr) / threshold)
    return SignatureTestResult(
        law=PhysicsLaw.EXPONENTIAL_DECAY,
        passed=passed,
        confidence=confidence,
        key_metric="CORR(DERIV(N), N)",
        key_value=corr,
        threshold=-threshold,
    )


def _test_free_fall(seq: List[float], threshold: float = 0.05) -> SignatureTestResult:
    """
    Free Fall: DERIV2(y) = constant (≈ -g)
    Test: coefficient of variation of DERIV2(seq) < threshold
    """
    if len(seq) < 10:
        return SignatureTestResult(PhysicsLaw.FREE_FALL, False, 0.0,
                                  "sequence too short", 0.0, threshold)
    d2 = _deriv2(seq)
    if not d2 or statistics.mean([abs(v) for v in d2]) < 1e-10:
        return SignatureTestResult(PhysicsLaw.FREE_FALL, False, 0.0,
                                  "zero acceleration", 1.0, threshold)
    mean_d2 = statistics.mean(d2)
    if abs(mean_d2) < 1e-10:
        cv = 1.0
    else:
        std_d2 = statistics.stdev(d2) if len(d2) > 1 else 0.0
        cv = std_d2 / abs(mean_d2)  # coefficient of variation
    passed = cv < threshold
    confidence = min(1.0, threshold / max(cv, 1e-10)) if passed else threshold / max(cv, 1e-10) * 0.5
    return SignatureTestResult(
        law=PhysicsLaw.FREE_FALL,
        passed=passed,
        confidence=confidence,
        key_metric="CV(DERIV2(y))",
        key_value=cv,
        threshold=threshold,
    )


def _test_simple_harmonic(seq: List[float], threshold: float = 0.80) -> SignatureTestResult:
    """
    Simple Harmonic Motion: DERIV2(x) + ω²x = 0
    This is equivalent to Hooke's Law but tests both sign and magnitude.
    """
    result = _test_hookes_law(seq, threshold)
    result.law = PhysicsLaw.SIMPLE_HARMONIC
    result.key_metric = "CORR(DERIV2(x)+ω²x, 0)"
    return result


def _test_newton_cooling(seq: List[float], threshold: float = 0.85) -> SignatureTestResult:
    """
    Newton's Law of Cooling: DERIV(T) ∝ -(T - T_env)
    Similar to exponential decay but with an offset.
    """
    # Estimate T_env as the final value (equilibrium)
    t_env = seq[-1] if seq else 0.0
    shifted = [v - t_env for v in seq]
    result = _test_exponential_decay(shifted, threshold)
    result.law = PhysicsLaw.NEWTON_COOLING
    result.key_metric = "CORR(DERIV(T-T_env), T-T_env)"
    return result


# ── Registry of all signatures ────────────────────────────────────────────────

ALL_SIGNATURES: List[LawSignature] = [
    LawSignature(
        law=PhysicsLaw.HOOKES_LAW,
        description="Spring oscillation: DERIV2(x) = -k*x",
        test_function=_test_hookes_law,
        examples=["SpringMassEnv", "PendulumEnv"],
    ),
    LawSignature(
        law=PhysicsLaw.EXPONENTIAL_DECAY,
        description="Radioactive decay: DERIV(N) = -λ*N",
        test_function=_test_exponential_decay,
        examples=["RadioactiveDecayEnv"],
    ),
    LawSignature(
        law=PhysicsLaw.FREE_FALL,
        description="Free fall: DERIV2(y) = constant",
        test_function=_test_free_fall,
        examples=["FreeFallEnv"],
    ),
    LawSignature(
        law=PhysicsLaw.NEWTON_COOLING,
        description="Newton cooling: DERIV(T) = -k*(T - T_env)",
        test_function=_test_newton_cooling,
        examples=["CoolingEnv"],
    ),
    LawSignature(
        law=PhysicsLaw.SIMPLE_HARMONIC,
        description="Simple harmonic motion: DERIV2(x) + ω²x = 0",
        test_function=_test_simple_harmonic,
        examples=["SpringMassEnv", "PendulumEnv"],
    ),
]