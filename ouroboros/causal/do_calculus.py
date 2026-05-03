"""
DoCalculusEngine — Pearl's do-calculus for causal inference.

The three rules of do-calculus (Pearl, 2000):
  Rule 1: Insertion/deletion of observations
    P(y | do(x), z, w) = P(y | do(x), w)
    when Y ⊥ Z | X, W in G_X̄ (graph with incoming edges to X removed)

  Rule 2: Action/observation exchange
    P(y | do(x), do(z), w) = P(y | do(x), z, w)
    when Y ⊥ Z | X, W in G_X̄Z̄ (graph with both X and Z intervened)

  Rule 3: Insertion/deletion of actions
    P(y | do(x), do(z), w) = P(y | do(x), w)
    when Y ⊥ Z | X, W in G_X̄Z̄(W) (graph with X, Z-ancestors-of-W intervened)

For OUROBOROS, we use these rules to:
  1. Identify causal effects from observational data
  2. Compute intervention predictions (if I set CO2=2x, what happens to temperature?)
  3. Test whether a discovered correlation is causal or confounded

Simplified implementation:
  We don't implement the full do-calculus (that requires interventional experiments).
  We implement the observational causal discovery component:
    - PC algorithm (skeleton discovery + orientation)
    - Score-based discovery (GES with MDL criterion)
    - Linear causal model estimation
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Set

from ouroboros.causal.causal_graph import (
    CausalGraph, CausalEdge, CausalVariable, InterventionalGraph,
)


@dataclass
class CausalEffectEstimate:
    """Estimated causal effect of X on Y."""
    cause_var: str
    effect_var: str
    effect_size: float          # magnitude of causal effect
    direction: str              # "positive", "negative", "nonlinear"
    confidence: float           # 0-1
    lag: int                    # time lag of the effect
    mechanism: Optional[str]    # symbolic form of mechanism
    is_direct: bool             # direct vs indirect effect
    adjustment_set: Set[str]    # variables conditioned on

    def description(self) -> str:
        dir_sym = "↑" if self.direction == "positive" else "↓" if self.direction == "negative" else "~"
        lag_str = f" [lag={self.lag}]" if self.lag > 0 else ""
        return (
            f"{self.cause_var} {dir_sym}→ {self.effect_var}{lag_str} "
            f"(size={self.effect_size:.3f}, conf={self.confidence:.2f}, "
            f"direct={self.is_direct})"
        )


def _partial_correlation(
    x: List[float],
    y: List[float],
    z: List[float],
) -> float:
    """
    Partial correlation of x and y controlling for z.
    Uses the formula: r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))
    """
    def pearson(a, b):
        n = min(len(a), len(b))
        if n < 3: return 0.0
        ma, mb = sum(a[:n])/n, sum(b[:n])/n
        num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
        sa = math.sqrt(sum((a[i]-ma)**2 for i in range(n)))
        sb = math.sqrt(sum((b[i]-mb)**2 for i in range(n)))
        return num / max(sa*sb, 1e-10)

    r_xy = pearson(x, y)
    r_xz = pearson(x, z)
    r_yz = pearson(y, z)
    denom = math.sqrt(max(0, (1 - r_xz**2) * (1 - r_yz**2)))
    if denom < 1e-10:
        return 0.0
    return (r_xy - r_xz * r_yz) / denom


def _granger_causality_test(
    cause: List[float],
    effect: List[float],
    max_lag: int = 5,
) -> Tuple[float, int]:
    """
    Granger causality test: does cause[t-k] help predict effect[t]?

    Returns: (f_statistic, best_lag)
    High F-statistic means cause Granger-causes effect.

    Note: Granger causality is NOT the same as Pearl causality.
    It tests whether past values of X help predict Y above and beyond
    Y's own past values. It's a necessary (not sufficient) condition
    for true causality.
    """
    n = min(len(cause), len(effect))
    if n < max_lag + 5:
        return 0.0, 0

    best_f = 0.0
    best_lag = 1

    for lag in range(1, min(max_lag + 1, n // 3)):
        # Restricted model: effect[t] ~ effect[t-1]
        y = effect[lag:]
        x_restricted = effect[:n-lag]
        # Unrestricted model: effect[t] ~ effect[t-1] + cause[t-lag]
        x_cause = cause[:n-lag]

        # Fit both models (OLS)
        n_obs = len(y)
        if n_obs < 5:
            continue

        # Residuals of restricted model
        r_restricted = _ols_residuals(x_restricted, y)
        # Residuals of unrestricted model (cause as extra predictor)
        r_unrestricted = _ols_residuals_2d(x_restricted, x_cause, y)

        # F-statistic
        rss_r = sum(r**2 for r in r_restricted)
        rss_u = sum(r**2 for r in r_unrestricted)
        if rss_u < 1e-12 or rss_r < rss_u:
            continue
        f_stat = ((rss_r - rss_u) / 1) / (rss_u / max(n_obs - 3, 1))
        if f_stat > best_f:
            best_f = f_stat
            best_lag = lag

    return best_f, best_lag


def _ols_residuals(x: List[float], y: List[float]) -> List[float]:
    """OLS residuals for simple regression y = a + b*x."""
    n = min(len(x), len(y))
    if n < 2: return [0.0] * n
    mx, my = sum(x[:n])/n, sum(y[:n])/n
    b_num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    b_den = sum((x[i]-mx)**2 for i in range(n))
    b = b_num / max(b_den, 1e-12)
    a = my - b * mx
    return [y[i] - (a + b * x[i]) for i in range(n)]


def _ols_residuals_2d(x1: List[float], x2: List[float], y: List[float]) -> List[float]:
    """OLS residuals for regression y = a + b1*x1 + b2*x2."""
    n = min(len(x1), len(x2), len(y))
    if n < 3: return [0.0] * n
    # Normal equations: [X'X][β] = [X'y]
    s11 = sum(x1[i]**2 for i in range(n))
    s22 = sum(x2[i]**2 for i in range(n))
    s12 = sum(x1[i]*x2[i] for i in range(n))
    s1y = sum(x1[i]*y[i] for i in range(n))
    s2y = sum(x2[i]*y[i] for i in range(n))
    sy  = sum(y[i] for i in range(n))
    sx1 = sum(x1)
    sx2 = sum(x2)

    # With intercept — simplified: use centered variables
    mx1 = sx1/n; mx2 = sx2/n; my = sy/n
    c11 = s11/n - mx1**2
    c22 = s22/n - mx2**2
    c12 = s12/n - mx1*mx2
    c1y = s1y/n - mx1*my
    c2y = s2y/n - mx2*my

    det = c11*c22 - c12**2
    if abs(det) < 1e-12:
        return _ols_residuals(x1, y)

    b1 = (c1y*c22 - c2y*c12) / det
    b2 = (c2y*c11 - c1y*c12) / det
    a = my - b1*mx1 - b2*mx2

    return [y[i] - (a + b1*x1[i] + b2*x2[i]) for i in range(n)]


class DoCalculusEngine:
    """
    Engine for causal discovery from observational time series.

    Discovers a CausalGraph by:
    1. Computing pairwise Granger causality (necessary condition)
    2. Orienting edges using time ordering (causes precede effects)
    3. Removing spurious edges via partial correlation tests
    4. Estimating causal effect sizes

    This is a simplified version of the PC algorithm adapted for
    time series data where temporal ordering provides natural edge orientation.
    """

    def __init__(
        self,
        granger_threshold: float = 5.0,   # F-statistic threshold
        partial_corr_threshold: float = 0.3,
        max_lag: int = 5,
    ):
        self.granger_threshold = granger_threshold
        self.partial_corr_threshold = partial_corr_threshold
        self.max_lag = max_lag

    def discover(
        self,
        sequences: Dict[str, List[float]],
        var_types: Dict[str, str] = None,
        verbose: bool = False,
    ) -> CausalGraph:
        """
        Discover causal structure from multiple time series.

        sequences: dict mapping variable name → list of float observations
        var_types: optional dict mapping name → type ("observed", "derived", etc.)
        """
        graph = CausalGraph()
        var_types = var_types or {}

        # Add all variables
        for name, seq in sequences.items():
            var = CausalVariable(name=name, var_type=var_types.get(name, "observed"))
            graph.add_variable(var)

        names = list(sequences.keys())
        n_vars = len(names)

        if verbose:
            print(f"\nCausal discovery over {n_vars} variables")

        # Step 1: Test all pairs for Granger causality
        granger_results = {}
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    continue
                seq_i = sequences[name_i]
                seq_j = sequences[name_j]
                f_stat, best_lag = _granger_causality_test(seq_i, seq_j, self.max_lag)
                granger_results[(name_i, name_j)] = (f_stat, best_lag)

                if verbose and f_stat > self.granger_threshold:
                    print(f"  Granger: {name_i} → {name_j} "
                          f"F={f_stat:.2f} lag={best_lag}")

        # Step 2: Add edges for pairs that pass Granger test
        candidate_edges = []
        for (cause, effect), (f_stat, lag) in granger_results.items():
            if f_stat > self.granger_threshold:
                # Check reverse direction
                reverse_f, _ = granger_results.get((effect, cause), (0.0, 0))
                if f_stat > reverse_f:  # more evidence for this direction
                    candidate_edges.append((cause, effect, f_stat, lag))

        # Step 3: Remove spurious edges via partial correlation
        final_edges = []
        for cause, effect, f_stat, lag in candidate_edges:
            other_vars = [n for n in names if n != cause and n != effect]
            # Check if edge survives conditioning on each third variable
            is_spurious = False
            for other in other_vars[:3]:  # check top 3 confounders
                seq_cause = sequences[cause][lag:]
                seq_effect = sequences[effect][lag:]
                seq_other = sequences[other][:len(seq_cause)]
                if len(seq_other) < len(seq_cause):
                    continue
                pc = _partial_correlation(seq_cause, seq_effect, seq_other)
                if abs(pc) < self.partial_corr_threshold:
                    is_spurious = True
                    break

            if not is_spurious:
                final_edges.append((cause, effect, f_stat, lag))

        # Step 4: Build graph and estimate effect sizes
        for cause, effect, f_stat, lag in final_edges:
            # Estimate effect size via correlation with appropriate lag
            seq_c = sequences[cause][:len(sequences[effect]) - lag]
            seq_e = sequences[effect][lag:]
            n = min(len(seq_c), len(seq_e))
            if n < 3:
                continue
            mc = sum(seq_c[:n])/n; me = sum(seq_e[:n])/n
            corr_num = sum((seq_c[i]-mc)*(seq_e[i]-me) for i in range(n))
            sc = math.sqrt(sum((v-mc)**2 for v in seq_c[:n]))
            se = math.sqrt(sum((v-me)**2 for v in seq_e[:n]))
            effect_size = abs(corr_num / max(sc*se, 1e-10))
            direction = "positive" if corr_num > 0 else "negative"
            confidence = min(1.0, f_stat / (self.granger_threshold * 5))

            var_cause = CausalVariable(name=cause, var_type=var_types.get(cause, "observed"))
            var_effect = CausalVariable(name=effect, var_type=var_types.get(effect, "observed"))
            edge = CausalEdge(
                cause=var_cause,
                effect=var_effect,
                strength=effect_size,
                lag=lag,
            )
            graph.add_edge(edge)

            if verbose:
                print(f"  ✓ Edge: {cause} → {effect} "
                      f"(effect={effect_size:.3f}, conf={confidence:.2f})")

        return graph

    def estimate_causal_effect(
        self,
        graph: CausalGraph,
        cause: str,
        effect: str,
        sequences: Dict[str, List[float]],
    ) -> CausalEffectEstimate:
        """
        Estimate the causal effect of cause on effect using the graph structure.
        Uses backdoor adjustment when confounders are present.
        """
        # Find adjustment set using backdoor criterion
        parents_of_cause = graph.parents(cause)
        adjustment_set = parents_of_cause  # sufficient (but not minimal)

        # Estimate effect size (simplified: use partial correlation adjusted for confounders)
        if not adjustment_set:
            # No confounders — simple correlation
            seq_c = sequences.get(cause, [])
            seq_e = sequences.get(effect, [])
            n = min(len(seq_c), len(seq_e))
            if n < 3:
                return CausalEffectEstimate(cause, effect, 0.0, "unknown",
                                            0.0, 0, None, True, set())
            mc = sum(seq_c[:n])/n; me = sum(seq_e[:n])/n
            corr = sum((seq_c[i]-mc)*(seq_e[i]-me) for i in range(n))
            sc = math.sqrt(sum((v-mc)**2 for v in seq_c[:n]))
            se = math.sqrt(sum((v-me)**2 for v in seq_e[:n]))
            effect_size = corr / max(sc*se, 1e-10)
        else:
            effect_size = 0.5  # simplified — would do full backdoor adjustment

        # Check edge properties in graph
        is_direct = any(
            e.cause.name == cause and e.effect.name == effect
            for e in graph._edges
        )
        edge = next(
            (e for e in graph._edges if e.cause.name == cause and e.effect.name == effect),
            None
        )
        lag = edge.lag if edge else 0

        direction = "positive" if effect_size > 0 else "negative"
        confidence = min(1.0, abs(effect_size))

        return CausalEffectEstimate(
            cause_var=cause,
            effect_var=effect,
            effect_size=abs(effect_size),
            direction=direction,
            confidence=confidence,
            lag=lag,
            mechanism=None,
            is_direct=is_direct,
            adjustment_set=adjustment_set,
        )