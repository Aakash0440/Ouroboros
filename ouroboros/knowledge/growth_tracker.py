"""
KnowledgeGrowthTracker — Analyzes how the axiom library grows across sessions.

Computes:
  - Axiom discovery rate: new axioms per session
  - MDL improvement rate: how much better does the system get?
  - Transfer benefit curve: benefit of KB priors vs session number
  - Diminishing returns: when does knowledge accumulation plateau?
  - Diversity index: how different are accumulated axioms from each other?
"""

from __future__ import annotations
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class GrowthAnalysis:
    """Analysis of knowledge growth across sessions."""
    n_sessions: int
    n_axioms_final: int

    # Per-session metrics
    mdl_costs: List[float]
    prior_benefits: List[float]
    axiom_counts: List[int]

    # Trend analysis
    early_mean_mdl: float      # sessions 1-10
    late_mean_mdl: float       # sessions -10:
    mdl_improvement: float     # early - late (positive = improved)
    improvement_fraction: float # mdl_improvement / early_mean_mdl

    # Discovery curve
    discovery_rate_early: float   # new axioms per session (sessions 1-20)
    discovery_rate_late: float    # new axioms per session (sessions -20:)
    plateau_session: Optional[int]   # session where discovery rate drops to <0.1

    # Transfer benefit
    mean_prior_benefit: float
    max_prior_benefit: float

    def summary(self) -> str:
        lines = [
            f"KNOWLEDGE GROWTH ANALYSIS ({self.n_sessions} sessions)",
            f"  Final KB size: {self.n_axioms_final} axioms",
            f"  Early MDL (s1-10): {self.early_mean_mdl:.2f}",
            f"  Late MDL (s-10:):  {self.late_mean_mdl:.2f}",
            f"  MDL improvement: {self.mdl_improvement:.2f} bits "
            f"({self.improvement_fraction*100:.1f}%)",
            f"  Discovery rate (early): {self.discovery_rate_early:.3f} axioms/session",
            f"  Discovery rate (late):  {self.discovery_rate_late:.3f} axioms/session",
        ]
        if self.plateau_session:
            lines.append(f"  Plateau at: session {self.plateau_session}")
        lines.append(f"  Mean prior benefit: {self.mean_prior_benefit:.2f} bits/session")
        lines.append(f"  Max prior benefit: {self.max_prior_benefit:.2f} bits")

        if self.improvement_fraction > 0.1:
            lines.append("\n  ✅ KNOWLEDGE ACCUMULATION CONFIRMED: system improves with experience")
        elif self.improvement_fraction > 0.0:
            lines.append("\n  ⚠️  MARGINAL IMPROVEMENT: slight trend toward better performance")
        else:
            lines.append("\n  ❌ NO IMPROVEMENT: KB priors not helping (try more sessions)")
        return "\n".join(lines)


class KnowledgeGrowthTracker:
    """Analyzes knowledge growth from accumulation records."""

    def analyze(
        self,
        sessions: List,   # List[SessionResult]
        verbose: bool = True,
    ) -> GrowthAnalysis:
        n = len(sessions)
        if n == 0:
            return self._empty_analysis()

        mdl_costs = [s.best_mdl_cost for s in sessions]
        prior_benefits = [s.prior_benefit for s in sessions]
        axiom_counts = [s.n_axioms_at_end for s in sessions]

        # Early vs late MDL
        early_window = min(10, n // 4)
        late_window = min(10, n // 4)
        early_costs = mdl_costs[:early_window]
        late_costs = mdl_costs[-late_window:] if n > late_window else mdl_costs

        early_mean = statistics.mean(early_costs)
        late_mean = statistics.mean(late_costs)
        improvement = early_mean - late_mean  # positive = improved
        improvement_frac = improvement / max(early_mean, 1e-10)

        # Discovery rate: new axioms per session
        new_axioms = [s.n_new_axioms for s in sessions]
        early_disc = statistics.mean(new_axioms[:early_window]) if early_window > 0 else 0.0
        late_disc = statistics.mean(new_axioms[-late_window:]) if late_window > 0 else 0.0

        # Find plateau: where rolling-10 discovery rate drops below 0.1
        plateau = None
        window_size = min(10, n // 5)
        for i in range(window_size, n):
            window_disc = statistics.mean(new_axioms[i-window_size:i])
            if window_disc < 0.1:
                plateau = i + 1  # 1-indexed session
                break

        mean_benefit = statistics.mean(prior_benefits) if prior_benefits else 0.0
        max_benefit = max(prior_benefits) if prior_benefits else 0.0

        analysis = GrowthAnalysis(
            n_sessions=n,
            n_axioms_final=axiom_counts[-1] if axiom_counts else 0,
            mdl_costs=mdl_costs,
            prior_benefits=prior_benefits,
            axiom_counts=axiom_counts,
            early_mean_mdl=early_mean,
            late_mean_mdl=late_mean,
            mdl_improvement=improvement,
            improvement_fraction=improvement_frac,
            discovery_rate_early=early_disc,
            discovery_rate_late=late_disc,
            plateau_session=plateau,
            mean_prior_benefit=mean_benefit,
            max_prior_benefit=max_benefit,
        )

        if verbose:
            print(analysis.summary())

        return analysis

    def _empty_analysis(self) -> GrowthAnalysis:
        return GrowthAnalysis(
            n_sessions=0, n_axioms_final=0, mdl_costs=[], prior_benefits=[],
            axiom_counts=[], early_mean_mdl=0.0, late_mean_mdl=0.0,
            mdl_improvement=0.0, improvement_fraction=0.0,
            discovery_rate_early=0.0, discovery_rate_late=0.0,
            plateau_session=None, mean_prior_benefit=0.0, max_prior_benefit=0.0,
        )

    def generate_growth_curve_data(
        self,
        sessions: List,
        window: int = 10,
    ) -> Dict:
        """
        Generate smoothed growth curve data for plotting.
        Returns dict with keys: 'sessions', 'mdl_smooth', 'axiom_counts', 'benefits_smooth'
        """
        n = len(sessions)
        if n == 0:
            return {}

        mdl_costs = [s.best_mdl_cost for s in sessions]
        benefits = [s.prior_benefit for s in sessions]
        axiom_counts = [s.n_axioms_at_end for s in sessions]

        def rolling_mean(lst, w):
            result = []
            for i in range(len(lst)):
                start = max(0, i - w + 1)
                result.append(statistics.mean(lst[start:i+1]))
            return result

        return {
            "sessions": list(range(1, n + 1)),
            "mdl_raw": mdl_costs,
            "mdl_smooth": rolling_mean(mdl_costs, window),
            "axiom_counts": axiom_counts,
            "benefits_raw": benefits,
            "benefits_smooth": rolling_mean(benefits, window),
        }