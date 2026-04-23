"""
StatisticalTester — Rigorous statistical analysis of communication experiments.

Tests implemented:
  1. Mann-Whitney U: nonparametric comparison of two conditions
     (used because rounds_to_consensus is often not normally distributed)
  2. Cohen's d: standardized effect size
  3. Bootstrap 95% CI: confidence interval for the difference in means
  4. Herding analysis: is herding index significantly higher in COMM condition?
"""

from __future__ import annotations
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MannWhitneyResult:
    """Result of a Mann-Whitney U test."""
    u_statistic: float
    p_value: float      # approximate, two-sided
    n1: int
    n2: int
    significant: bool   # p < 0.05

    def description(self) -> str:
        sig_str = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"Mann-Whitney U={self.u_statistic:.1f}, "
            f"p={self.p_value:.4f} ({sig_str}), "
            f"n1={self.n1}, n2={self.n2}"
        )


@dataclass
class EffectSizeResult:
    """Cohen's d effect size."""
    cohens_d: float
    pooled_std: float
    interpretation: str   # "negligible", "small", "medium", "large"

    @classmethod
    def from_groups(cls, group1: List[float], group2: List[float]) -> 'EffectSizeResult':
        if len(group1) < 2 or len(group2) < 2:
            return cls(0.0, 0.0, "insufficient data")
        m1 = statistics.mean(group1)
        m2 = statistics.mean(group2)
        s1 = statistics.stdev(group1)
        s2 = statistics.stdev(group2)
        n1, n2 = len(group1), len(group2)
        pooled = math.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
        d = abs(m1 - m2) / max(pooled, 1e-10)
        if d < 0.5:
            interp = "negligible"
        elif d < 2.0:
            interp = "small"
        elif d < 4.0:
            interp = "medium"
        else:
            interp = "large"
        return cls(d, pooled, interp)


@dataclass
class BootstrapCI:
    """Bootstrap 95% confidence interval for the difference in means."""
    lower: float
    upper: float
    observed_diff: float
    n_bootstrap: int
    includes_zero: bool = field(init=False)

    def __post_init__(self):
        self.includes_zero = self.lower <= 0 <= self.upper


def mann_whitney_u(x: List[float], y: List[float]) -> MannWhitneyResult:
    """
    Compute Mann-Whitney U statistic and approximate p-value.
    
    The p-value is approximated using the normal approximation to the
    U statistic. This is valid for n > 8.
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return MannWhitneyResult(0, 1.0, n1, n2, False)

    # Count how many pairs (xi, yj) where xi > yj
    u1 = sum(1 for xi in x for yj in y if xi > yj) + \
         0.5 * sum(1 for xi in x for yj in y if xi == yj)
    u2 = n1 * n2 - u1
    u_stat = min(u1, u2)

    # Normal approximation
    mean_u = n1 * n2 / 2.0
    std_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_u == 0:
        return MannWhitneyResult(u_stat, 1.0, n1, n2, False)

    z = (u_stat - mean_u) / std_u
    # Two-sided p-value using normal CDF approximation
    p_value = 2.0 * _norm_cdf(-abs(z))

    return MannWhitneyResult(
        u_statistic=u_stat,
        p_value=p_value,
        n1=n1,
        n2=n2,
        significant=p_value < 0.05,
    )


def _norm_cdf(z: float) -> float:
    """Approximate normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bootstrap_ci(
    x: List[float],
    y: List[float],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> BootstrapCI:
    """
    Bootstrap 95% CI for the difference in means (mean(x) - mean(y)).
    """
    rng = random.Random(seed)
    observed_diff = statistics.mean(x) - statistics.mean(y)

    boot_diffs = []
    for _ in range(n_bootstrap):
        sample_x = [rng.choice(x) for _ in x]
        sample_y = [rng.choice(y) for _ in y]
        boot_diffs.append(statistics.mean(sample_x) - statistics.mean(sample_y))

    boot_diffs.sort()
    lower_idx = int(0.025 * n_bootstrap)
    upper_idx = int(0.975 * n_bootstrap)
    lower = boot_diffs[lower_idx]
    upper = boot_diffs[min(upper_idx, len(boot_diffs)-1)]

    return BootstrapCI(lower=lower, upper=upper, observed_diff=observed_diff, n_bootstrap=n_bootstrap)


@dataclass
class CommunicationAnalysisResult:
    """Full statistical analysis comparing SOLO vs COMM conditions."""
    n_solo_runs: int
    n_comm_runs: int

    # Rounds to consensus
    solo_consensus_rounds: List[float]
    comm_consensus_rounds: List[float]
    consensus_mwu: MannWhitneyResult
    consensus_effect_size: EffectSizeResult
    consensus_ci: BootstrapCI

    # Best MDL cost
    solo_costs: List[float]
    comm_costs: List[float]
    cost_mwu: MannWhitneyResult
    cost_effect_size: EffectSizeResult

    # Herding
    solo_herding: List[float]
    comm_herding: List[float]
    herding_mwu: MannWhitneyResult

    # Unique expressions
    solo_unique: List[int]
    comm_unique: List[int]

    def summary(self) -> str:
        def fmt(lst): return f"{statistics.mean(lst):.3f} ± {statistics.stdev(lst):.3f}" if len(lst) > 1 else str(lst[0])
        lines = [
            "COMMUNICATION EXPERIMENT — STATISTICAL ANALYSIS",
            "=" * 60,
            f"Runs: SOLO={self.n_solo_runs}, COMM={self.n_comm_runs}",
            "",
            "Rounds to Consensus:",
            f"  SOLO: {fmt(self.solo_consensus_rounds)}",
            f"  COMM: {fmt(self.comm_consensus_rounds)}",
            f"  Mann-Whitney: {self.consensus_mwu.description()}",
            f"  Effect size: Cohen's d={self.consensus_effect_size.cohens_d:.3f} ({self.consensus_effect_size.interpretation})",
            f"  Bootstrap CI for diff: [{self.consensus_ci.lower:.3f}, {self.consensus_ci.upper:.3f}]",
            f"  {'Includes zero → communication has NO significant effect on convergence' if self.consensus_ci.includes_zero else 'Excludes zero → communication HAS significant effect'}",
            "",
            "Herding Index (0=diverse, 1=identical):",
            f"  SOLO: {fmt(self.solo_herding)}",
            f"  COMM: {fmt(self.comm_herding)}",
            f"  Mann-Whitney: {self.herding_mwu.description()}",
            "",
            "Unique Expressions Found:",
            f"  SOLO: {statistics.mean(self.solo_unique):.1f} ± {statistics.stdev(self.solo_unique):.1f}" if len(self.solo_unique) > 1 else f"  SOLO: {self.solo_unique[0]}",
            f"  COMM: {statistics.mean(self.comm_unique):.1f} ± {statistics.stdev(self.comm_unique):.1f}" if len(self.comm_unique) > 1 else f"  COMM: {self.comm_unique[0]}",
            "",
            "CONCLUSION:",
        ]
        # Interpret results
        comm_faster = (statistics.mean(self.comm_consensus_rounds) <
                       statistics.mean(self.solo_consensus_rounds)
                       if self.comm_consensus_rounds and self.solo_consensus_rounds else False)
        herding_higher = (statistics.mean(self.comm_herding) >
                         statistics.mean(self.solo_herding)
                         if self.comm_herding and self.solo_herding else False)

        if comm_faster and self.consensus_mwu.significant:
            lines.append("  Communication significantly ACCELERATES convergence")
        elif not comm_faster and self.consensus_mwu.significant:
            lines.append("  Communication significantly SLOWS convergence (herding)")
        else:
            lines.append("  Communication has NO statistically significant effect on convergence")

        if herding_higher and self.herding_mwu.significant:
            lines.append("  Communication INCREASES herding (agents explore same region)")
        else:
            lines.append("  Herding index is not significantly different between conditions")

        return "\n".join(lines)


class StatisticalTester:
    """Runs the full statistical analysis of communication experiments."""

    def analyze(
        self,
        solo_runs: list,
        comm_runs: list,
        n_rounds: int,
    ) -> CommunicationAnalysisResult:
        """
        Run full statistical analysis comparing SOLO vs COMM.
        
        Parameters are lists of ExperimentRun objects.
        """
        # Extract rounds_to_consensus (use n_rounds+1 if not achieved)
        def get_consensus(runs):
            return [
                float(r.rounds_to_consensus) if r.consensus_achieved
                else float(n_rounds + 1)
                for r in runs
            ]

        solo_consensus = get_consensus(solo_runs)
        comm_consensus = get_consensus(comm_runs)

        solo_costs = [r.final_best_cost for r in solo_runs]
        comm_costs = [r.final_best_cost for r in comm_runs]

        solo_herding = [r.herding_index for r in solo_runs]
        comm_herding = [r.herding_index for r in comm_runs]

        solo_unique = [r.unique_expressions for r in solo_runs]
        comm_unique = [r.unique_expressions for r in comm_runs]

        return CommunicationAnalysisResult(
            n_solo_runs=len(solo_runs),
            n_comm_runs=len(comm_runs),
            solo_consensus_rounds=solo_consensus,
            comm_consensus_rounds=comm_consensus,
            consensus_mwu=mann_whitney_u(solo_consensus, comm_consensus),
            consensus_effect_size=EffectSizeResult.from_groups(solo_consensus, comm_consensus),
            consensus_ci=bootstrap_ci(solo_consensus, comm_consensus),
            solo_costs=solo_costs,
            comm_costs=comm_costs,
            cost_mwu=mann_whitney_u(solo_costs, comm_costs),
            cost_effect_size=EffectSizeResult.from_groups(solo_costs, comm_costs),
            solo_herding=solo_herding,
            comm_herding=comm_herding,
            herding_mwu=mann_whitney_u(solo_herding, comm_herding),
            solo_unique=solo_unique,
            comm_unique=comm_unique,
        )