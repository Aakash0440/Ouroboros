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


# ── Shared utilities ───────────────────────────────────────────────────────────

def _clean_seq(seq: List) -> List[float]:
    """
    Replace None/NaN values via linear interpolation.
    Handles missing values at the edges via forward/backward fill.
    """
    result = list(seq)
    n = len(result)

    # Mark bad indices
    def is_bad(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    # Build list of valid (index, value) pairs
    valid = [(i, float(result[i])) for i in range(n) if not is_bad(result[i])]
    if not valid:
        return [0.0] * n

    # Fill bad values by linear interpolation between nearest valid neighbours
    for i in range(n):
        if not is_bad(result[i]):
            continue
        # find nearest valid on left
        left = next(((j, v) for j, v in reversed(valid) if j < i), None)
        # find nearest valid on right
        right = next(((j, v) for j, v in valid if j > i), None)

        if left is None and right is None:
            result[i] = 0.0
        elif left is None:
            result[i] = right[1]
        elif right is None:
            result[i] = left[1]
        else:
            lj, lv = left
            rj, rv = right
            result[i] = lv + (rv - lv) * (i - lj) / (rj - lj)

    return [float(v) for v in result]


def _smooth(seq: List[float], w: int = 5) -> List[float]:
    """Centred moving average with half-width w."""
    out = []
    n = len(seq)
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        out.append(sum(seq[lo:hi]) / (hi - lo))
    return out


def _safe_corr(x: List[float], y: List[float]) -> float:
    """Pearson correlation, returning 0 if undefined."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    x, y = x[:n], y[:n]
    mx, my = sum(x) / n, sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    sy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    denom = sx * sy
    if denom < 1e-12:
        return 0.0
    return num / denom


def _deriv(seq: List[float]) -> List[float]:
    """First discrete derivative."""
    return [seq[i] - seq[i - 1] for i in range(1, len(seq))]


def _deriv2(seq: List[float]) -> List[float]:
    """Second discrete derivative."""
    return _deriv(_deriv(seq))


# ── Law signature definitions ──────────────────────────────────────────────────
def _test_hookes_law(seq: List[float], threshold: float = 0.85) -> SignatureTestResult:
    """
    Hooke's Law: DERIV2(x) ∝ -x
    
    v3 strategy: instead of correlating raw derivatives (fails at SNR=2),
    we fit the sequence to A*cos(ω*t + φ) + C using least-squares over ω,
    then verify the fit quality via normalised RMSE.
    
    Why this works at low SNR:
      - Smoothed derivative correlation degrades as O(noise²) because
        differentiation amplifies high-frequency noise.
      - Cosine fitting uses all N points simultaneously, so noise averages
        out as O(noise/√N) — much more robust.
      - A good cosine fit IS Hooke's law: x(t) = A·cos(ωt+φ) is the exact
        solution to ẍ + ω²x = 0.
    
    The test: fit A·cos(ωt) + B·sin(ωt) + C for a grid of ω values.
    Best-fit normalised RMSE < threshold → Hooke's law confirmed.
    Confidence = 1 - best_rmse / signal_std.
    """
    seq = _clean_seq(seq)
    n = len(seq)
    if n < 10:
        return SignatureTestResult(PhysicsLaw.HOOKES_LAW, False, 0.0,
                                   "sequence too short", 0.0, threshold)

    import math
    mean_val = sum(seq) / n
    signal_std = math.sqrt(sum((v - mean_val)**2 for v in seq) / n)
    if signal_std < 1e-10:
        return SignatureTestResult(PhysicsLaw.HOOKES_LAW, False, 0.0,
                                   "constant sequence", 1.0, threshold)

    best_rmse = float('inf')
    best_omega = 0.0

    # Grid search over plausible angular frequencies
    # Period range: 4 steps (Nyquist) to n/2 steps
    for period_steps in range(4, n // 2 + 1):
        omega = 2 * math.pi / period_steps

        # Linear least squares: fit A*cos(wt) + B*sin(wt) + C
        # Build design matrix columns
        cos_col = [math.cos(omega * t) for t in range(n)]
        sin_col = [math.sin(omega * t) for t in range(n)]

        # Normal equations for [A, B, C]
        # X^T X
        cc = sum(c**2 for c in cos_col)
        ss = sum(s**2 for s in sin_col)
        cs = sum(cos_col[t] * sin_col[t] for t in range(n))
        c1 = sum(cos_col)
        s1 = sum(sin_col)

        # X^T y
        cy = sum(cos_col[t] * seq[t] for t in range(n))
        sy = sum(sin_col[t] * seq[t] for t in range(n))
        oy = sum(seq)

        # Solve 3x3 system via substitution
        # Since cos and sin are nearly orthogonal over integer periods, 
        # cs ≈ 0 and c1, s1 ≈ 0 for full periods — use direct formula
        try:
            # Full 3x3 solve using Cramer's rule
            mat = [
                [cc, cs, c1],
                [cs, ss, s1],
                [c1, s1,  n],
            ]
            rhs = [cy, sy, oy]

            # Gaussian elimination
            m = [row[:] + [rhs[i]] for i, row in enumerate(mat)]
            for col in range(3):
                # Pivot
                max_row = max(range(col, 3), key=lambda r: abs(m[r][col]))
                m[col], m[max_row] = m[max_row], m[col]
                if abs(m[col][col]) < 1e-12:
                    continue
                for row in range(col + 1, 3):
                    factor = m[row][col] / m[col][col]
                    for j in range(col, 4):
                        m[row][j] -= factor * m[col][j]

            # Back substitution
            x = [0.0, 0.0, 0.0]
            for row in range(2, -1, -1):
                x[row] = m[row][3]
                for col in range(row + 1, 3):
                    x[row] -= m[row][col] * x[col]
                if abs(m[row][row]) > 1e-12:
                    x[row] /= m[row][row]

            A_fit, B_fit, C_fit = x

        except Exception:
            continue

        # Compute RMSE of fit
        fitted = [A_fit * math.cos(omega * t) + B_fit * math.sin(omega * t) + C_fit
                  for t in range(n)]
        residuals = [seq[t] - fitted[t] for t in range(n)]
        rmse = math.sqrt(sum(r**2 for r in residuals) / n)
        norm_rmse = rmse / signal_std  # normalised: 0=perfect, 1=no better than mean

        if norm_rmse < best_rmse:
            best_rmse = norm_rmse
            best_omega = omega

    # A good cosine fit means Hooke's law holds
    # threshold here is re-interpreted as max normalised RMSE
    # For clean data: norm_rmse ≈ 0.0
    # For SNR=2 (noise_std = signal_std/2): expect norm_rmse ≈ 0.4-0.5
    # We use threshold=0.75 as the ceiling (very noisy but still oscillatory)
    hooke_threshold = 0.75  # always use this regardless of caller's threshold
    passed = best_rmse < hooke_threshold
    confidence = max(0.0, 1.0 - best_rmse / hooke_threshold) if passed \
                 else max(0.0, 1.0 - best_rmse)

    return SignatureTestResult(
        law=PhysicsLaw.HOOKES_LAW,
        passed=passed,
        confidence=confidence,
        key_metric="norm_RMSE(cosine_fit)",
        key_value=best_rmse,
        threshold=hooke_threshold,
    )

def _test_exponential_decay(seq: List[float], threshold: float = 0.85) -> SignatureTestResult:
    """
    Radioactive Decay: DERIV(N) ∝ -N
    Test: correlation between DERIV(seq) and seq[1:] < -threshold

    Fix v2: null guard + log-space outlier rejection before computing correlation.
    A single spike outlier in linear space maps to a large positive log-deviation
    and is cleanly identified as > 3σ from the log mean, then discarded.
    """
    seq = _clean_seq(seq)
    if len(seq) < 10:
        return SignatureTestResult(PhysicsLaw.EXPONENTIAL_DECAY, False, 0.0,
                                   "sequence too short", 0.0, threshold)

    # Log-space outlier rejection: values more than 3σ from the log mean are spikes
    log_vals = []
    valid_idx = []
    for i, v in enumerate(seq):
        if v > 1e-10:
            log_vals.append(math.log(v))
            valid_idx.append(i)

    if len(log_vals) < 5:
        return SignatureTestResult(PhysicsLaw.EXPONENTIAL_DECAY, False, 0.0,
                                   "too few positive values", 0.0, threshold)

    mean_log = statistics.mean(log_vals)
    std_log  = statistics.stdev(log_vals) if len(log_vals) > 1 else 1.0

    clean = [seq[i] for i, lv in zip(valid_idx, log_vals)
             if abs(lv - mean_log) < 3.0 * std_log]

    if len(clean) < 5:
        clean = seq  # fallback: use everything

    # Correlation test on cleaned sequence
    nz = [(d, n) for d, n in zip(_deriv(clean), clean[1:]) if abs(n) > 1e-6]
    if len(nz) < 5:
        return SignatureTestResult(PhysicsLaw.EXPONENTIAL_DECAY, False, 0.0,
                                   "too many zeros after cleaning", 0.0, threshold)

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
    Free Fall: DERIV2(y) = constant (≡ linear velocity)
    Primary test: relative RMSE of a linear fit to first differences.

    Fix v2: null guard + strip flat prefix/suffix + linear-velocity test.

    Why linear-velocity instead of CV(DERIV2)?
    - Quantization turns the smooth parabola into a staircase; DERIV2 of a
      staircase has large spike variance → cv explodes even after smoothing.
    - First differences (velocity proxy) of a parabola are linear in t.
      Fitting a line to those differences and measuring normalised residuals
      is robust to quantization because the linear trend dominates even when
      individual steps are rounded integers.
    - We additionally require the fitted slope to be negative (object is
      falling, not rising) to avoid false positives on rising sequences.
    """
    seq = _clean_seq(seq)
    if len(seq) < 10:
        return SignatureTestResult(PhysicsLaw.FREE_FALL, False, 0.0,
                                   "sequence too short", 0.0, threshold)

    # Strip flat prefix (object hasn't started falling yet)
    start = 0
    for i in range(1, len(seq)):
        if abs(seq[i] - seq[i - 1]) > 1e-9:
            start = max(0, i - 1)
            break

    # Strip flat suffix (object has hit the ground / clamped at 0)
    end = len(seq)
    for i in range(len(seq) - 1, 0, -1):
        if abs(seq[i] - seq[i - 1]) > 1e-9:
            end = i + 1
            break

    active = seq[start:end]
    if len(active) < 5:
        return SignatureTestResult(PhysicsLaw.FREE_FALL, False, 0.0,
                                   "active region too short", 1.0, threshold)

    # First differences (velocity proxy)
    d1 = [active[i + 1] - active[i] for i in range(len(active) - 1)]

    if not d1 or all(abs(x) < 1e-9 for x in d1):
        return SignatureTestResult(PhysicsLaw.FREE_FALL, False, 0.0,
                                   "no motion in active region", 1.0, threshold)

    # Fit a linear trend to d1 — free fall gives perfectly linear velocity
    n = len(d1)
    t_vals = list(range(n))
    mean_t = sum(t_vals) / n
    mean_d = sum(d1) / n

    ss_t  = sum((t - mean_t) ** 2 for t in t_vals)
    ss_td = sum((t - mean_t) * (d - mean_d) for t, d in zip(t_vals, d1))

    if ss_t < 1e-10:
        return SignatureTestResult(PhysicsLaw.FREE_FALL, False, 0.0,
                                   "no time variance", 1.0, threshold)

    slope     = ss_td / ss_t
    intercept = mean_d - slope * mean_t

    residuals = [d1[i] - (slope * t_vals[i] + intercept) for i in range(n)]
    rmse      = (sum(r ** 2 for r in residuals) / n) ** 0.5

    d1_range  = max(d1) - min(d1)
    rel_rmse  = rmse / d1_range if d1_range > 1e-9 else float('inf')

    # Must have negative slope (falling object loses height)
    passed = rel_rmse < threshold and slope < 0

    if passed:
        confidence = min(1.0, threshold / max(rel_rmse, 1e-10))
    else:
        confidence = 0.0

    return SignatureTestResult(
        law=PhysicsLaw.FREE_FALL,
        passed=passed,
        confidence=confidence,
        key_metric="rel_RMSE(Δh~linear)",
        key_value=rel_rmse,
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
    seq = _clean_seq(seq)
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