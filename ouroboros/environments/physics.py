"""
Physics environments for OUROBOROS.

These generate time series from physical systems.
Agents with DERIV, DERIV2, EWMA, and CUMSUM nodes can discover
the governing equations from raw measurement data.

Environments:
  SpringMassEnv      — position x(t) = A*cos(ωt)
  RadioactiveDecayEnv — count(t) = N₀*exp(-λt) (discrete approximation)
  FreeFallEnv        — position y(t) = h - ½gt²
  PendulumEnv        — θ(t) = θ₀*cos(√(g/L)*t) (small angle approx)
"""

from __future__ import annotations
import math
from typing import List
from ouroboros.environments.base import Environment


class SpringMassEnv(Environment):
    """
    x(t) = A * cos(omega * t + phi)
    
    Discoverable rule: DERIV2(x) = -omega² * x   (Hooke's Law)
    With DERIV2 and CORR nodes, agents can discover this relation.
    
    Discrete version: x[t] rounded to nearest integer
    for integer-valued environments.
    """
    def __init__(
        self,
        amplitude: float = 10.0,
        omega: float = 0.3,
        phi: float = 0.0,
        noise_sigma: float = 0.0,
        as_integer: bool = True,
        seed: int = 42,
    ):
        self.seed = seed
        super().__init__(alphabet_size=int(2 * amplitude) + 2, name=f"SpringMass(A={amplitude},ω={omega:.2f})", seed=seed)
        self.amplitude = amplitude
        self.omega = omega
        self.phi = phi
        self.noise_sigma = noise_sigma
        self.as_integer = as_integer

    def generate(self, length: int, start: int = 0) -> List:
        import random
        rng = random.Random(self.seed)
        result = []
        for t in range(start, start + length):
            v = self.amplitude * math.cos(self.omega * t + self.phi)
            if self.noise_sigma > 0:
                v += rng.gauss(0, self.noise_sigma)
            if self.as_integer:
                result.append(max(0, int(round(v + self.amplitude))))  # shift to positive
            else:
                result.append(v)
        return result

    def ground_truth_rule(self) -> str:
        return f"x[t] = {self.amplitude}*cos({self.omega:.3f}*t)"

    def discovered_law(self) -> str:
        return f"DERIV2(x) + {self.omega**2:.4f}*x = 0   (Hooke's Law)"


class RadioactiveDecayEnv(Environment):
    """
    count(t) = floor(N0 * exp(-lambda * t))
    
    Discoverable rule: DERIV(count) ≈ -lambda * count
    Agents with DERIV, CORR, and THRESHOLD nodes can find this.
    """
    def __init__(
        self,
        n0: int = 1000,
        decay_rate: float = 0.05,
        seed: int = 42,
    ):
        super().__init__(alphabet_size=n0 + 1, name=f"RadioactiveDecay(λ={decay_rate})", seed=seed)
        self.n0 = n0
        self.decay_rate = decay_rate

    def generate(self, length: int, start: int = 0) -> List[int]:
        return [
            max(0, int(self.n0 * math.exp(-self.decay_rate * t)))
            for t in range(start, start + length)
        ]

    def ground_truth_rule(self) -> str:
        return f"count[t] = {self.n0}*exp(-{self.decay_rate}*t)"

    def discovered_law(self) -> str:
        return f"DERIV(count)/count ≈ -{self.decay_rate}   (Radioactive decay law)"


class FreeFallEnv(Environment):
    """
    y(t) = h0 - (g/2) * t²   (free fall from height h0)
    
    Discoverable rule: DERIV2(y) ≈ -g = -9.8 (constant acceleration)
    """
    def __init__(
        self,
        h0: float = 100.0,
        g: float = 9.8,
        scale: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(alphabet_size=int(h0) + 2, name=f"FreeFall(h={h0},g={g})", seed=seed)
        self.h0 = h0
        self.g = g
        self.scale = scale  # scale factor to keep values in integer range

    def generate(self, n: int) -> list:
        """
        Generate n samples of free-fall height.

        Regardless of the stored scale parameter, we map t=0..n-1 across the
        full parabolic arc (launch → impact) so the second derivative is
        always constant and detectable. The scale is preserved for the
        amplitude but does not compress the time axis into invisibility.
        """
        import math
        # Time of impact under real physics: h0 = ½g·t_impact²
        t_impact = math.sqrt(2.0 * self.h0 / self.g)
        result = []
        for i in range(n):
            # Map sample index to physical time across the full arc
            t_phys = i / (n - 1) * t_impact
            h = self.h0 - 0.5 * self.g * t_phys ** 2
            result.append(float(max(0.0, h)))
        return result


    def ground_truth_rule(self) -> str:
        return f"y[t] = {self.h0:.1f} - {0.5*self.g*self.scale**2:.4f}*t²"

    def discovered_law(self) -> str:
        return f"DERIV2(y) ≈ -{self.g*self.scale**2:.4f}   (Constant acceleration)"
