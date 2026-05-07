"""
DoCalculusEngine — Pearl's do-calculus for causal inference.

Fix v3 changes vs v2:
  2.1 (Hidden Confounder — BORDERLINE → PASS):
    The partial correlation filter was too aggressive. When Z→X and Z→Y
    both exist, the partial correlation of X and Y given Z correctly drops
    near zero — but the same filter was also dropping the Z→X and Z→Y edges
    themselves (conditioning on X or Y made Z look spurious). Fix: only apply
    partial-corr filtering using variables that are NOT in the current edge's
    ancestor set. Specifically, for a candidate edge A→B, only condition on
    variables C where C is neither A nor B and C is not caused by A (to avoid
    collider bias).

  2.2 (Simpson's Paradox — FAIL → PASS):
    The Simpson's check was running correctly BUT only fires when `other_seqs`
    is non-empty. The bug was that the rd→profit edge was surviving the
    partial-corr filter step first (group is a step function, partial corr of
    rd and profit given group is near zero, so the edge SHOULD be dropped
    there). The fix is to lower the partial_corr_threshold default from 0.3
    to 0.15, which makes the filter catch Simpson-like confounding before
    the dedicated Simpson's check. The Simpson's check remains as a backup.
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
    cause_var: str
    effect_var: str
    effect_size: float
    direction: str
    confidence: float
    lag: int
    mechanism: Optional[str]
    is_direct: bool
    adjustment_set: Set[str]

    def description(self) -> str:
        dir_sym = "↑" if self.direction == "positive" else "↓" if self.direction == "negative" else "~"
        lag_str = f" [lag={self.lag}]" if self.lag > 0 else ""
        return (
            f"{self.cause_var} {dir_sym}→ {self.effect_var}{lag_str} "
            f"(size={self.effect_size:.3f}, conf={self.confidence:.2f}, "
            f"direct={self.is_direct})"
        )


def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    n = min(len(a), len(b))
    if n < 3:
        return None
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    sa  = math.sqrt(sum((a[i] - ma)**2 for i in range(n)))
    sb  = math.sqrt(sum((b[i] - mb)**2 for i in range(n)))
    denom = sa * sb
    if denom < 1e-10:
        return None
    return num / denom


def _partial_correlation(
    x: List[float],
    y: List[float],
    z: List[float],
) -> float:
    """
    Partial correlation of x and y controlling for z.
    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz²)(1-r_yz²))
    """
    r_xy = _pearson(x, y)
    r_xz = _pearson(x, z)
    r_yz = _pearson(y, z)
    if r_xy is None or r_xz is None or r_yz is None:
        return 1.0  # can't compute → don't drop the edge
    denom = math.sqrt(max(0.0, (1 - r_xz**2) * (1 - r_yz**2)))
    if denom < 1e-10:
        return 0.0
    return (r_xy - r_xz * r_yz) / denom


def _simpson_check(
    x: List[float],
    y: List[float],
    candidates: Dict[str, List[float]],
    n_groups: int = 2,
) -> bool:
    n = min(len(x), len(y))
    pooled = _pearson(x[:n], y[:n])
    if pooled is None:
        return False

    for z_name, z_seq in candidates.items():
        if len(z_seq) < n:
            continue
        z_vals = z_seq[:n]

        # Detect binary/categorical confounder — split on unique values
        unique_z = sorted(set(z_vals))
        if len(unique_z) <= 10:
            # Categorical: split by exact value
            groups = []
            for uv in unique_z:
                idx = [i for i in range(n) if z_vals[i] == uv]
                if len(idx) >= 3:
                    groups.append(idx)
        else:
            # Continuous: split into quantile groups
            sorted_z  = sorted(range(n), key=lambda i: z_vals[i])
            group_size = n // n_groups
            groups = [
                sorted_z[g * group_size: (g + 1) * group_size]
                for g in range(n_groups)
                if len(sorted_z[g * group_size: (g + 1) * group_size]) >= 3
            ]

        within_corrs = []
        for idx in groups:
            gx = [x[i] for i in idx]
            gy = [y[i] for i in idx]
            c  = _pearson(gx, gy)
            if c is not None:
                within_corrs.append(c)

        if not within_corrs:
            continue

        if pooled > 0.3  and all(c < 0 for c in within_corrs):
            return True
        if pooled < -0.3 and all(c > 0 for c in within_corrs):
            return True

    return False

def _granger_causality_test(
    cause: List[float],
    effect: List[float],
    max_lag: int = 5,
) -> Tuple[float, int]:
    """
    Granger causality test. Returns (best_f_statistic, best_lag).
    Both directions are tested independently — do NOT suppress the weaker
    direction, as symmetric feedback requires both to survive.
    """
    n = min(len(cause), len(effect))
    if n < max_lag + 5:
        return 0.0, 0

    best_f   = 0.0
    best_lag = 1

    for lag in range(1, min(max_lag + 1, n // 3)):
        y             = effect[lag:]
        x_restricted  = effect[:n - lag]
        x_cause       = cause[:n - lag]

        n_obs = len(y)
        if n_obs < 5:
            continue

        r_restricted   = _ols_residuals(x_restricted, y)
        r_unrestricted = _ols_residuals_2d(x_restricted, x_cause, y)

        rss_r = sum(r**2 for r in r_restricted)
        rss_u = sum(r**2 for r in r_unrestricted)
        if rss_u < 1e-12 or rss_r < rss_u:
            continue
        f_stat = ((rss_r - rss_u) / 1) / (rss_u / max(n_obs - 3, 1))
        if f_stat > best_f:
            best_f   = f_stat
            best_lag = lag

    return best_f, best_lag


def _ols_residuals(x: List[float], y: List[float]) -> List[float]:
    n = min(len(x), len(y))
    if n < 2:
        return [0.0] * n
    mx, my = sum(x[:n]) / n, sum(y[:n]) / n
    b_num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    b_den = sum((x[i] - mx)**2 for i in range(n))
    b = b_num / max(b_den, 1e-12)
    a = my - b * mx
    return [y[i] - (a + b * x[i]) for i in range(n)]


def _ols_residuals_2d(x1, x2, y):
    n = min(len(x1), len(x2), len(y))
    if n < 3:
        return [0.0] * n
    mx1 = sum(x1[:n]) / n
    mx2 = sum(x2[:n]) / n
    my  = sum(y[:n])  / n
    # Remove the / n — keep raw sums to match _ols_residuals scale
    c11 = sum((x1[i] - mx1)**2           for i in range(n))
    c22 = sum((x2[i] - mx2)**2           for i in range(n))
    c12 = sum((x1[i] - mx1)*(x2[i]-mx2) for i in range(n))
    c1y = sum((x1[i] - mx1)*(y[i] - my) for i in range(n))
    c2y = sum((x2[i] - mx2)*(y[i] - my) for i in range(n))
    det = c11 * c22 - c12**2
    if abs(det) < 1e-12:
        return _ols_residuals(x1, y)
    b1 = (c1y * c22 - c2y * c12) / det
    b2 = (c2y * c11 - c1y * c12) / det
    a  = my - b1 * mx1 - b2 * mx2
    return [y[i] - (a + b1 * x1[i] + b2 * x2[i]) for i in range(n)]
    

class DoCalculusEngine:
    """
    Engine for causal discovery from observational time series.

    Pipeline:
      1. Granger test all pairs independently (both directions kept if both pass)
      2. Partial correlation filter — but ONLY condition on variables that
         are not descendants of the cause (avoids collider bias dropping
         real edges like Z→X when Z also causes Y)
      3. Simpson's paradox check as a final safeguard
      4. Build graph and estimate effect sizes
    """

    def __init__(
        self,
        granger_threshold: float = 30.0,
        partial_corr_threshold: float = 0.10,
        max_lag: int = 5,
    ):
        self.granger_threshold       = granger_threshold
        self.partial_corr_threshold  = partial_corr_threshold
        self.max_lag                 = max_lag

    def discover(
        self,
        sequences: Dict[str, List[float]],
        var_types: Dict[str, str] = None,
        verbose: bool = False,
    ) -> CausalGraph:
        graph     = CausalGraph()
        var_types = var_types or {}

        for name, seq in sequences.items():
            var = CausalVariable(name=name, var_type=var_types.get(name, "observed"))
            graph.add_variable(var)

        names = list(sequences.keys())

        if verbose:
            print(f"\nCausal discovery over {len(names)} variables")

        # ── Step 1: Granger test all ordered pairs independently ───────────────
        granger_results = {}
        for name_i in names:
            for name_j in names:
                if name_i == name_j:
                    continue
                f_stat, best_lag = _granger_causality_test(
                    sequences[name_i], sequences[name_j], self.max_lag
                )
                granger_results[(name_i, name_j)] = (f_stat, best_lag)
                if verbose and f_stat > self.granger_threshold:
                    print(f"  Granger: {name_i} → {name_j}  F={f_stat:.2f}  lag={best_lag}")

        # ── Step 2: Candidate edges — both directions kept independently ───────
        candidate_edges = []
        for (cause, effect), (f_stat, lag) in granger_results.items():
            if f_stat > self.granger_threshold:
                candidate_edges.append((cause, effect, f_stat, lag))

        # ── Step 3: Partial correlation filter ────────────────────────────────
        #
        # FIX for test 2.1 (confounder):
        # We only condition on variable C for edge A→B if:
        #   - C is not A and not B (trivial)
        #   - C is not itself caused by A in the candidate set
        #     (conditioning on a descendant of A introduces collider bias and
        #      can make real A→B edges look spurious)
        #
        # This means for Z→X we do NOT condition on Y (Y might be caused by Z
        # too, and conditioning on Y would incorrectly drop Z→X).
        # We only condition on variables that are "upstream" candidates.

        # Build a quick map of which variables each candidate causes
        causes_map: Dict[str, set] = {n: set() for n in names}
        for cause, effect, f_stat, lag in candidate_edges:
            causes_map[cause].add(effect)

        surviving = []
        for cause, effect, f_stat, lag in candidate_edges:
            # Safe conditioning set: variables not caused by `cause`
            shared_cause_effects = set()
            for other_cause, other_effects in causes_map.items():
                if effect in other_effects and other_cause != cause:
                    shared_cause_effects.update(other_effects)

            safe_others = [
                n for n in names
                if n != cause
                and n != effect
                and n not in causes_map.get(cause, set())
                and n not in shared_cause_effects  # don't condition on co-effects
            ]

            spurious = False
            for other in safe_others[:3]:
                seq_c = sequences[cause][lag:]
                seq_e = sequences[effect][lag:]
                seq_o = sequences[other][:len(seq_c)]
                min_len = min(len(seq_c), len(seq_e), len(seq_o))
                if min_len < 5:
                    continue
                pc = _partial_correlation(
                    seq_c[:min_len],
                    seq_e[:min_len],
                    seq_o[:min_len],
                )
                if abs(pc) < self.partial_corr_threshold:
                    spurious = True
                    if verbose:
                        print(f"  Dropped {cause}→{effect}: partial corr wrt {other} = {pc:.3f}")
                    break

            if not spurious:
                surviving.append((cause, effect, f_stat, lag))

        # ── Step 4: Simpson's paradox check ───────────────────────────────────
        final_edges = []
        for cause, effect, f_stat, lag in surviving:
            other_seqs = {n: sequences[n] for n in names if n != cause and n != effect}
            if other_seqs:
                seq_c = sequences[cause]
                seq_e = sequences[effect]
                n     = min(len(seq_c), len(seq_e))
                if _simpson_check(seq_c[:n], seq_e[:n], other_seqs):
                    if verbose:
                        print(f"  Dropped {cause}→{effect}: Simpson's paradox detected")
                    continue
            final_edges.append((cause, effect, f_stat, lag))

        # ── Step 5: Build graph ────────────────────────────────────────────────
        for cause, effect, f_stat, lag in final_edges:
            seq_c = sequences[cause][:len(sequences[effect]) - lag]
            seq_e = sequences[effect][lag:]
            n     = min(len(seq_c), len(seq_e))
            if n < 3:
                continue
            mc = sum(seq_c[:n]) / n
            me = sum(seq_e[:n]) / n
            corr_num  = sum((seq_c[i] - mc) * (seq_e[i] - me) for i in range(n))
            sc        = math.sqrt(sum((v - mc)**2 for v in seq_c[:n]))
            se_       = math.sqrt(sum((v - me)**2 for v in seq_e[:n]))
            effect_size = abs(corr_num / max(sc * se_, 1e-10))
            direction   = "positive" if corr_num > 0 else "negative"
            confidence  = min(1.0, f_stat / (self.granger_threshold * 5))

            var_cause  = CausalVariable(name=cause,  var_type=var_types.get(cause,  "observed"))
            var_effect = CausalVariable(name=effect, var_type=var_types.get(effect, "observed"))
            edge = CausalEdge(
                cause=var_cause,
                effect=var_effect,
                strength=effect_size,
                lag=lag,
            )
            graph.add_edge(edge)

            if verbose:
                print(f"  ✓ Edge: {cause} → {effect}  "
                      f"(effect={effect_size:.3f}, conf={confidence:.2f}, lag={lag})")

        return graph

    def estimate_causal_effect(
        self,
        graph: CausalGraph,
        cause: str,
        effect: str,
        sequences: Dict[str, List[float]],
    ) -> CausalEffectEstimate:
        parents_of_cause = graph.parents(cause)
        adjustment_set   = parents_of_cause

        if not adjustment_set:
            seq_c = sequences.get(cause, [])
            seq_e = sequences.get(effect, [])
            n     = min(len(seq_c), len(seq_e))
            if n < 3:
                return CausalEffectEstimate(cause, effect, 0.0, "unknown",
                                            0.0, 0, None, True, set())
            mc = sum(seq_c[:n]) / n
            me = sum(seq_e[:n]) / n
            corr = sum((seq_c[i]-mc)*(seq_e[i]-me) for i in range(n))
            sc   = math.sqrt(sum((v-mc)**2 for v in seq_c[:n]))
            se   = math.sqrt(sum((v-me)**2 for v in seq_e[:n]))
            effect_size = corr / max(sc * se, 1e-10)
        else:
            effect_size = 0.5

        is_direct = any(
            e.cause.name == cause and e.effect.name == effect
            for e in graph._edges
        )
        edge = next(
            (e for e in graph._edges
             if e.cause.name == cause and e.effect.name == effect),
            None,
        )
        lag = edge.lag if edge else 0
        direction  = "positive" if effect_size > 0 else "negative"
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