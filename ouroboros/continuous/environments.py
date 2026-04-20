"""
Continuous observation environments for OUROBOROS.

These emit List[float] instead of List[int]. The hidden mathematical
structure includes sinusoidal functions, polynomials, exponentials,
and differential equations — none of which are reachable from integer
modular arithmetic.

Design: Each environment has a noise_sigma parameter. At sigma=0.0,
the sequence is deterministic and MDL compression should reach near-zero
residual cost. At sigma>0 the Gaussian MDL penalizes noisy predictions
appropriately.
"""

from __future__ import annotations
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ContinuousObservation:
    """A single float observation with optional ground-truth label."""
    value: float
    timestep: int
    env_name: str
    ground_truth_expr: Optional[str] = None  # e.g. "sin(t/7)"


class ContinuousEnvironment(ABC):
    """
    Base class for all continuous observation environments.
    
    Agents receive List[float] and must find a symbolic expression
    that predicts the sequence under Gaussian MDL.
    """

    def __init__(self, name: str, noise_sigma: float = 0.0, seed: int = 42):
        self.name = name
        self.noise_sigma = noise_sigma
        self.seed = seed
        self._rng = random.Random(seed)

    @abstractmethod
    def true_value(self, t: int) -> float:
        """The noiseless value at timestep t."""
        ...

    def observe(self, t: int) -> float:
        """Sample value at timestep t, adding Gaussian noise if configured."""
        v = self.true_value(t)
        if self.noise_sigma > 0.0:
            v += self._rng.gauss(0.0, self.noise_sigma)
        return v

    def generate(self, length: int, start: int = 0) -> List[float]:
        """Generate a sequence of length observations starting at timestep start."""
        return [self.observe(t) for t in range(start, start + length)]

    @abstractmethod
    def ground_truth_expr(self) -> str:
        """Human-readable description of the true generating function."""
        ...

    def difficulty_score(self) -> str:
        """Rough difficulty for human reference."""
        return "medium"


# ─── ENVIRONMENT 1: Pure Sinusoid ────────────────────────────────────────────

class SineEnv(ContinuousEnvironment):
    """
    obs[t] = amplitude * sin(frequency * t + phase) + offset
    
    Simplest continuous structure. Agents should discover the 4 constants.
    At noise_sigma=0, Gaussian MDL residual → 0.
    Ground truth: amplitude=1.0, frequency=1/7, phase=0.0, offset=0.0
    """

    def __init__(
        self,
        amplitude: float = 1.0,
        frequency: float = 1.0 / 7.0,
        phase: float = 0.0,
        offset: float = 0.0,
        noise_sigma: float = 0.0,
        seed: int = 42
    ):
        super().__init__("SineEnv", noise_sigma, seed)
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.offset = offset

    def true_value(self, t: int) -> float:
        return self.amplitude * math.sin(self.frequency * t + self.phase) + self.offset

    def ground_truth_expr(self) -> str:
        return (
            f"{self.amplitude:.3f} * sin({self.frequency:.4f} * t"
            f" + {self.phase:.3f}) + {self.offset:.3f}"
        )

    def difficulty_score(self) -> str:
        return "easy"


# ─── ENVIRONMENT 2: Polynomial ────────────────────────────────────────────────

class PolynomialEnv(ContinuousEnvironment):
    """
    obs[t] = sum(coeffs[i] * t^i for i in range(degree+1))
    
    Default: 0.5*t^2 - 2.0*t + 1.0 (degree-2 polynomial)
    Agents must discover both the degree and the coefficients.
    """

    def __init__(
        self,
        coefficients: List[float] = None,
        noise_sigma: float = 0.0,
        seed: int = 42
    ):
        super().__init__("PolynomialEnv", noise_sigma, seed)
        self.coefficients = coefficients if coefficients is not None else [1.0, -2.0, 0.5]

    def true_value(self, t: int) -> float:
        return sum(c * (t ** i) for i, c in enumerate(self.coefficients))

    def ground_truth_expr(self) -> str:
        terms = []
        for i, c in enumerate(self.coefficients):
            if i == 0:
                terms.append(f"{c:.3f}")
            elif i == 1:
                terms.append(f"{c:+.3f}*t")
            else:
                terms.append(f"{c:+.3f}*t^{i}")
        return " ".join(terms)

    def difficulty_score(self) -> str:
        return "easy" if len(self.coefficients) <= 3 else "medium"


# ─── ENVIRONMENT 3: Exponential ───────────────────────────────────────────────

class ExponentialEnv(ContinuousEnvironment):
    """
    obs[t] = scale * exp(rate * t) + offset
    
    Default: 1.0 * exp(0.05 * t)  — slow exponential growth
    Warning: at large t, values explode. Generate short sequences (length ≤ 100).
    """

    def __init__(
        self,
        scale: float = 1.0,
        rate: float = 0.05,
        offset: float = 0.0,
        noise_sigma: float = 0.0,
        seed: int = 42
    ):
        super().__init__("ExponentialEnv", noise_sigma, seed)
        self.scale = scale
        self.rate = rate
        self.offset = offset

    def true_value(self, t: int) -> float:
        return self.scale * math.exp(self.rate * t) + self.offset

    def ground_truth_expr(self) -> str:
        return f"{self.scale:.3f} * exp({self.rate:.4f} * t) + {self.offset:.3f}"

    def difficulty_score(self) -> str:
        return "medium"


# ─── ENVIRONMENT 4: Damped Oscillator ────────────────────────────────────────

class DampedOscillatorEnv(ContinuousEnvironment):
    """
    obs[t] = amplitude * exp(-decay * t) * sin(omega * t + phase)
    
    Classic physics: spring-mass system with friction.
    Default: amplitude=2.0, decay=0.1, omega=0.8, phase=0.0
    
    This is the HARD environment for continuous agents — requires
    discovering both the exponential envelope AND the oscillation simultaneously.
    No simple 1-parameter family covers it.
    """

    def __init__(
        self,
        amplitude: float = 2.0,
        decay: float = 0.1,
        omega: float = 0.8,
        phase: float = 0.0,
        noise_sigma: float = 0.0,
        seed: int = 42
    ):
        super().__init__("DampedOscillatorEnv", noise_sigma, seed)
        self.amplitude = amplitude
        self.decay = decay
        self.omega = omega
        self.phase = phase

    def true_value(self, t: int) -> float:
        return (
            self.amplitude
            * math.exp(-self.decay * t)
            * math.sin(self.omega * t + self.phase)
        )

    def ground_truth_expr(self) -> str:
        return (
            f"{self.amplitude:.2f} * exp(-{self.decay:.3f}*t)"
            f" * sin({self.omega:.3f}*t + {self.phase:.3f})"
        )

    def difficulty_score(self) -> str:
        return "hard"


# ─── ENVIRONMENT 5: Logistic / Chaos ─────────────────────────────────────────

class LogisticMapEnv(ContinuousEnvironment):
    """
    obs[t] = logistic map: x_{t+1} = r * x_t * (1 - x_t)
    
    At r=3.5 → period-4 attractor (structured, discoverable)
    At r=3.9 → chaotic (no simple closed form, high MDL cost)
    At r=2.8 → converges to fixed point (trivially discoverable)
    
    This is OUROBOROS's noise-equivalent for continuous agents:
    the r=3.9 case should produce near-zero compression and zero axioms.
    """

    def __init__(
        self,
        r: float = 3.5,
        x0: float = 0.5,
        noise_sigma: float = 0.0,
        seed: int = 42
    ):
        super().__init__("LogisticMapEnv", noise_sigma, seed)
        self.r = r
        self.x0 = x0
        self._cache: List[float] = []
        self._build_cache(2000)

    def _build_cache(self, length: int) -> None:
        x = self.x0
        self._cache = []
        for _ in range(length):
            self._cache.append(x)
            x = self.r * x * (1.0 - x)

    def true_value(self, t: int) -> float:
        if t >= len(self._cache):
            self._build_cache(t + 500)
        return self._cache[t]

    def ground_truth_expr(self) -> str:
        if self.r < 3.0:
            lim = 1.0 - 1.0 / self.r
            return f"fixed point: {lim:.4f}"
        if self.r < 3.45:
            return f"2-cycle (r={self.r})"
        if self.r < 3.54:
            return f"4-cycle (r={self.r})"
        return f"chaotic (r={self.r}) — no closed form"

    def difficulty_score(self) -> str:
        if self.r < 3.0:
            return "trivial"
        if self.r < 3.54:
            return "medium"
        return "impossible (chaotic)"


# ─── ENVIRONMENT 6: Gaussian Noise baseline ──────────────────────────────────

class ContinuousNoiseEnv(ContinuousEnvironment):
    """
    Pure Gaussian white noise — no structure.
    
    Like discrete NoiseEnv: agents should find ZERO axioms here.
    MDL cost should remain high (residual ≈ variance of noise).
    """

    def __init__(self, sigma: float = 1.0, seed: int = 42):
        super().__init__("ContinuousNoiseEnv", noise_sigma=sigma, seed=seed)
        self.sigma = sigma

    def true_value(self, t: int) -> float:
        return 0.0  # The "true" function is zero; all signal is noise

    def ground_truth_expr(self) -> str:
        return f"Gaussian(0, {self.sigma}²) — no structure"

    def difficulty_score(self) -> str:
        return "impossible (noise)"


# ─── Factory ──────────────────────────────────────────────────────────────────

def make_continuous_environment_suite() -> List[ContinuousEnvironment]:
    """Return all 6 environments for the continuous benchmark suite."""
    return [
        SineEnv(amplitude=1.0, frequency=1/7, noise_sigma=0.0),
        SineEnv(amplitude=1.5, frequency=1/11, noise_sigma=0.05),   # noisy version
        PolynomialEnv(coefficients=[1.0, -2.0, 0.5], noise_sigma=0.0),
        ExponentialEnv(scale=1.0, rate=0.05, noise_sigma=0.0),
        DampedOscillatorEnv(amplitude=2.0, decay=0.1, omega=0.8),
        LogisticMapEnv(r=3.5, x0=0.5),    # period-4 — discoverable
        LogisticMapEnv(r=3.9, x0=0.5),    # chaotic — control
        ContinuousNoiseEnv(sigma=1.0),
    ]
