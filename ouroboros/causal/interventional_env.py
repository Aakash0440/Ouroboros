"""
InterventionalEnvironment — Environments supporting do(X=x) queries.

Standard OUROBOROS environments are observational: they generate sequences
from a fixed mechanism. An interventional environment additionally supports
do-queries: "what would happen to Y if I forced X=x?"

This is the key capability that separates causal models from predictive models.
The system can test whether its discovered expression predicts intervention
outcomes correctly — if it does, the relationship is likely causal.

Environments implemented:
  InterventionalSpringMass  — do(position=x) stops the spring, tests prediction
  InterventionalDecay       — do(count=N) resets the count, tests decay rate
  InterventionalClimate     — synthetic CO2→Temperature with forced CO2 values

Usage:
  env = InterventionalSpringMass()
  # Observational data
  obs_seq = env.generate(200)
  # Interventional data: what if we force position=5 at t=50?
  int_seq = env.intervene("position", value=5.0, at_time=50, n_steps=100)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from ouroboros.environments.base import Environment


@dataclass
class InterventionResult:
    """Result of a do(X=x) intervention."""
    intervened_var: str
    intervention_value: float
    at_time: int
    pre_intervention: List[float]
    post_intervention: List[float]
    counterfactual: List[float]    # what would have happened without intervention

    @property
    def causal_effect(self) -> float:
        """Average difference between intervention and counterfactual."""
        n = min(len(self.post_intervention), len(self.counterfactual))
        if n == 0:
            return 0.0
        return sum(self.post_intervention[i] - self.counterfactual[i]
                   for i in range(n)) / n


class InterventionalSpringMassEnv(Environment):
    """
    Spring-mass system supporting position interventions.

    Observational: x(t) = A*cos(ω*t + φ)
    Causal structure: POSITION → FORCE (Hooke's), FORCE → ACCELERATION,
                      ACCELERATION → VELOCITY, VELOCITY → POSITION

    Intervention: do(POSITION=x₀) at time t₀
      - Cuts the Hooke's law feedback at t₀
      - Position is fixed to x₀ at t₀
      - System evolves from (x₀, v(t₀)) after t₀

    If OUROBOROS has learned that DERIV2(position) = -k*position (Hooke's Law),
    it can predict the post-intervention sequence correctly.
    """

    def __init__(
        self,
        amplitude: float = 10.0,
        omega: float = 0.3,
        phi: float = 0.0,
        seed: int = 42,
    ):
        super().__init__(name="InterventionalSpringMass", seed=seed)
        self.A = amplitude
        self.omega = omega
        self.phi = phi

    def generate(self, length: int, start: int = 0) -> List[float]:
        return [self.A * math.cos(self.omega * (t + start) + self.phi)
                for t in range(length)]

    def intervene(
        self,
        var_name: str,
        value: float,
        at_time: int,
        n_steps: int = 50,
    ) -> InterventionResult:
        """Apply do(var_name = value) at at_time, observe for n_steps."""
        # Pre-intervention: observational trajectory
        pre = self.generate(at_time)

        # At intervention time: position and velocity
        pos_at_t = self.A * math.cos(self.omega * at_time + self.phi)
        vel_at_t = -self.A * self.omega * math.sin(self.omega * at_time + self.phi)

        if var_name == "position":
            # Force position to value, velocity unchanged
            new_pos = value
            new_vel = vel_at_t  # velocity not changed by position intervention
        elif var_name == "velocity":
            new_pos = pos_at_t
            new_vel = value
        else:
            new_pos = pos_at_t
            new_vel = vel_at_t

        # Post-intervention: system evolves from (new_pos, new_vel) with same spring
        # Using simple harmonic motion from new initial conditions
        # x(t) = C*cos(ω*t) + D*sin(ω*t) where C, D from (new_pos, new_vel)
        C = new_pos
        D = new_vel / max(self.omega, 1e-10)

        post = []
        for t in range(n_steps):
            x = C * math.cos(self.omega * t) + D * math.sin(self.omega * t)
            post.append(x)

        # Counterfactual: what would have happened without intervention
        counterfactual = [
            self.A * math.cos(self.omega * (at_time + t) + self.phi)
            for t in range(n_steps)
        ]

        return InterventionResult(
            intervened_var=var_name,
            intervention_value=value,
            at_time=at_time,
            pre_intervention=pre,
            post_intervention=post,
            counterfactual=counterfactual,
        )

    @property
    def alphabet_size(self) -> int:
        return int(2 * self.A) + 2

    def get_causal_graph(self) -> 'CausalGraph':
        """Return the ground-truth causal graph for this system."""
        from ouroboros.causal.causal_graph import CausalGraph, CausalEdge, CausalVariable
        g = CausalGraph()
        pos = CausalVariable("position", "observed")
        vel = CausalVariable("velocity", "derived")
        acc = CausalVariable("acceleration", "derived")
        force = CausalVariable("force", "derived")
        for v in [pos, vel, acc, force]:
            g.add_variable(v)
        g.add_edge(CausalEdge(pos, force, lag=0))    # Hooke's: F = -k*x
        g.add_edge(CausalEdge(force, acc, lag=0))    # Newton: a = F/m
        g.add_edge(CausalEdge(acc, vel, lag=1))      # v(t) = v(t-1) + a(t-1)
        g.add_edge(CausalEdge(vel, pos, lag=1))      # x(t) = x(t-1) + v(t-1)
        return g


class SyntheticClimateEnv(Environment):
    """
    Synthetic CO2 → Temperature system with interventional support.

    Ground truth causal structure:
      CO2(t) → RadiativeForcing(t) → Temperature(t)
      Temperature(t) → CO2(t+100)  [feedback loop, long lag]

    Observational generation:
      CO2(t) = CO2_baseline * (1 + 0.005*t)  [gradual increase]
      RF(t) = 3.7 * log2(CO2(t)/CO2_baseline)  [radiative forcing formula]
      Temp(t) = Temp_baseline + 0.8 * EWMA(RF, 30yr)  [slow response]

    Intervention: do(CO2=2*baseline)
      Counterfactual: what would temperature be if CO2 had stayed at baseline?
      Causal effect: temperature difference between intervention and counterfactual

    This is the structure of climate sensitivity calculation.
    """

    def __init__(
        self,
        co2_baseline: float = 280.0,  # ppm, pre-industrial
        temp_baseline: float = 14.0,  # °C
        climate_sensitivity: float = 3.0,  # °C per CO2 doubling (IPCC central estimate)
        seed: int = 42,
    ):
        super().__init__(name="SyntheticClimate", seed=seed)
        self.co2_baseline = co2_baseline
        self.temp_baseline = temp_baseline
        self.sensitivity = climate_sensitivity

    def generate(self, length: int, start: int = 0) -> List[float]:
        """Returns temperature sequence."""
        return [self._temperature(t + start) for t in range(length)]

    def generate_multivariate(self, length: int) -> Dict[str, List[float]]:
        """Returns all variables: CO2, RF, Temperature."""
        return {
            "co2": [self._co2(t) for t in range(length)],
            "radiative_forcing": [self._rf(t) for t in range(length)],
            "temperature": [self._temperature(t) for t in range(length)],
        }

    def _co2(self, t: int) -> float:
        return self.co2_baseline * (1 + 0.005 * t)

    def _rf(self, t: int) -> float:
        co2 = self._co2(t)
        return 3.7 * math.log2(max(co2, 1.0) / self.co2_baseline)

    def _temperature(self, t: int) -> float:
        # Temperature responds to RF with a lag (EWMA)
        alpha = 0.05  # slow response
        ewma_rf = 0.0
        for i in range(t + 1):
            ewma_rf = alpha * self._rf(i) + (1 - alpha) * ewma_rf
        return self.temp_baseline + (self.sensitivity / 3.7) * ewma_rf

    def intervene_co2(
        self,
        intervention_value: float,
        at_time: int,
        n_steps: int = 100,
    ) -> InterventionResult:
        """Compute do(CO2 = intervention_value) counterfactual."""
        pre = self.generate(at_time)

        # Post-intervention: CO2 fixed to intervention_value
        post = []
        for t_offset in range(n_steps):
            t = at_time + t_offset
            # Forced CO2 level
            co2_forced = intervention_value
            rf_forced = 3.7 * math.log2(max(co2_forced, 1.0) / self.co2_baseline)

            # Temperature response with EWMA (simplified)
            alpha = 0.05
            temp = self.temp_baseline + (self.sensitivity / 3.7) * rf_forced * alpha
            post.append(temp)

        counterfactual = [self._temperature(at_time + t) for t in range(n_steps)]

        return InterventionResult(
            intervened_var="co2",
            intervention_value=intervention_value,
            at_time=at_time,
            pre_intervention=pre,
            post_intervention=post,
            counterfactual=counterfactual,
        )

    @property
    def alphabet_size(self) -> int:
        return 50


class CausalDiscoveryRunner:
    """
    Runs the complete causal discovery pipeline on a set of time series.

    Pipeline:
    1. Collect multiple time series (observed + derived quantities)
    2. Run DoCalculusEngine to discover causal graph
    3. Estimate causal effect sizes
    4. Generate intervention predictions
    5. Verify predictions against actual interventional data
    """

    def __init__(self, verbose: bool = True):
        from ouroboros.causal.do_calculus import DoCalculusEngine
        self._engine = DoCalculusEngine(
            granger_threshold=4.0,
            partial_corr_threshold=0.25,
            max_lag=5,
        )
        self.verbose = verbose

    def discover_from_environment(self, env, n_obs: int = 300) -> Dict:
        """Run causal discovery on an environment."""
        if hasattr(env, 'generate_multivariate'):
            sequences = env.generate_multivariate(n_obs)
        else:
            obs = env.generate(n_obs)
            # Generate derived quantities
            sequences = {
                "obs": [float(v) for v in obs],
                "deriv": [float(obs[t] - obs[t-1]) if t > 0 else 0.0
                          for t in range(len(obs))],
                "deriv2": [float(obs[t] - 2*obs[t-1] + obs[t-2])
                           if t >= 2 else 0.0
                           for t in range(len(obs))],
            }

        if self.verbose:
            print(f"\nCausal discovery: {env.name}")
            print(f"Variables: {list(sequences.keys())}")

        graph = self._engine.discover(sequences, verbose=self.verbose)

        result = {
            "env_name": env.name,
            "n_variables": graph.n_variables,
            "n_edges": graph.n_edges,
            "graph": graph.to_string(),
        }

        if self.verbose:
            print(f"\n{graph.to_string()}")
            print(f"Discovered {graph.n_edges} causal edges")

        return result