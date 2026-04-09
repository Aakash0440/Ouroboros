"""
Hierarchical MDL — compression at multiple temporal scales.

Standard MDL compresses a sequence symbol-by-symbol.
Hierarchical MDL compresses aggregated windows at multiple scales:

    Scale 1  (window=1):   individual symbols
    Scale 4  (window=4):   chunks of 4 symbols aggregated
    Scale 16 (window=16):  chunks of 16 symbols aggregated
    Scale 64 (window=64):  chunks of 64 symbols aggregated

For MultiScaleEnv(slow=100, fast=7):
    Scale 1:  captures the fast 7-period pattern clearly
    Scale 16: noise averages out; slow 100-period becomes visible
    Scale 64: slow pattern dominates

An agent that only runs Scale 1 misses the slow pattern.
An agent that runs ALL scales finds BOTH patterns.
The one with lowest compression ratio at each scale wins.

Why this matters for OUROBOROS:
The axioms that emerge at different scales may be DIFFERENT.
Scale-1 axiom: "t mod 7" (fast pattern)
Scale-16 axiom: "t mod 100" (slow pattern, after aggregation)
These are two INDEPENDENT mathematical rules about the same stream.
Together, they form a richer theory than either alone.

Reference:
    Hoel, E.P. et al. (2013). Quantifying causal emergence.
    PNAS 110(49):19790–19795.
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from ouroboros.compression.mdl import (
    compression_ratio, naive_bits, MDLCost, zstd_compressed_bits,
    sequence_to_bytes
)


def aggregate_sequence(
    sequence: List[int],
    window: int,
    alphabet_size: int,
    method: str = 'sum_mod'
) -> List[int]:
    """
    Aggregate a sequence into non-overlapping windows of size `window`.

    Aggregation methods:
        'sum_mod':  sum of window symbols mod alphabet_size
                    Preserves modular structure at larger scales.
        'majority': most common symbol in window
                    Preserves categorical structure.
        'mean_round': round(mean) mod alphabet_size
                    For detecting mean-level patterns.

    Example:
        sequence = [1, 4, 0, 3, 6, 2, 5, 1, 4, 0, 3, 6]
        window = 4, alphabet_size = 7, method = 'sum_mod'
        chunk1 = [1,4,0,3] → sum=8 → 8%7 = 1
        chunk2 = [6,2,5,1] → sum=14 → 14%7 = 0
        chunk3 = [4,0,3,6] → sum=13 → 13%7 = 6
        → [1, 0, 6]

    Args:
        sequence: Input integer sequence
        window: Aggregation window size
        alphabet_size: For mod reduction
        method: Aggregation function

    Returns:
        Aggregated sequence (length = len(sequence) // window)
    """
    # Validate requested aggregation method even for trivial windows so bad
    # inputs raise consistently (see tests).
    valid_methods = {'sum_mod', 'majority', 'mean_round'}
    if method not in valid_methods:
        raise ValueError(f"Unknown aggregation method: {method}")

    if window <= 1:
        return list(sequence)

    n = len(sequence)
    num_chunks = n // window
    if num_chunks == 0:
        return []

    result = []
    for i in range(num_chunks):
        chunk = sequence[i * window:(i + 1) * window]

        if method == 'sum_mod':
            val = sum(chunk) % alphabet_size
        elif method == 'majority':
            from collections import Counter
            val = Counter(chunk).most_common(1)[0][0]
        elif method == 'mean_round':
            val = round(np.mean(chunk)) % alphabet_size

        result.append(val)

    return result


def compression_at_scale(
    sequence: List[int],
    window: int,
    alphabet_size: int,
    agg_method: str = 'sum_mod'
) -> float:
    """
    Measure compression ratio of a sequence at a given temporal scale.

    Aggregates sequence into windows, then compresses.
    Returns compression ratio of the aggregated sequence.

    Lower ratio = more structure at this scale.
    """
    aggregated = aggregate_sequence(sequence, window, alphabet_size, agg_method)
    if len(aggregated) < 5:
        return 1.0
    return compression_ratio(aggregated, alphabet_size)


class HierarchicalMDL:
    """
    Measures compression at multiple temporal scales simultaneously.

    For each scale (window size), aggregates the sequence and measures
    compression. The scale with lowest compression ratio is the
    'causally dominant scale' — where the most structure lives.

    Args:
        scales: List of window sizes to probe (default: [1, 4, 16, 64])
        alphabet_size: Symbol alphabet size
        agg_method: Aggregation method for windowing
    """

    def __init__(
        self,
        scales: List[int] = None,
        alphabet_size: int = 4,
        agg_method: str = 'sum_mod'
    ):
        self.scales = scales or [1, 4, 16, 64]
        self.alphabet_size = alphabet_size
        self.agg_method = agg_method

    def compression_profile(self, sequence: List[int]) -> Dict[int, float]:
        """
        Compute compression ratio at every scale.

        Returns: dict mapping window_size → compression_ratio

        Example output for MultiScaleEnv(slow=100, fast=7):
            {1: 0.12, 4: 0.18, 16: 0.09, 64: 0.25}
            → Scale 16 has most structure (lowest ratio)
              Scale 1 has second most (fast pattern)

        Example output for NoiseEnv:
            {1: 0.97, 4: 0.96, 16: 0.95, 64: 0.94}
            → No scale compresses well (all near 1.0)
        """
        profile = {}
        for scale in self.scales:
            r = compression_at_scale(
                sequence, scale, self.alphabet_size, self.agg_method
            )
            profile[scale] = r
        return profile

    def dominant_scale(self, sequence: List[int]) -> Tuple[int, float]:
        """
        Find the scale with the best compression (lowest ratio).

        Returns: (best_window_size, best_compression_ratio)

        This is the 'causally dominant scale' — where the mathematical
        structure is most visible.
        """
        profile = self.compression_profile(sequence)
        best_scale = min(profile, key=profile.get)
        return best_scale, profile[best_scale]

    def multi_scale_improvement(self, sequence: List[int]) -> float:
        """
        How much better is the best scale vs. scale-1?

        Returns: ratio_at_scale_1 / ratio_at_best_scale
        Values > 1.0 mean higher scales reveal extra structure.
        Values ≈ 1.0 mean all structure is at the symbol level.
        """
        profile = self.compression_profile(sequence)
        scale1_ratio = profile.get(1, profile[self.scales[0]])
        best_ratio = min(profile.values())
        if best_ratio == 0:
            return float('inf')
        return scale1_ratio / best_ratio

    def scale_structure_report(self, sequence: List[int]) -> str:
        """
        Human-readable report of structure at each scale.
        Used for debugging and paper writing.
        """
        profile = self.compression_profile(sequence)
        best_scale, best_ratio = self.dominant_scale(sequence)

        lines = ["HierarchicalMDL Structure Report"]
        lines.append(f"  Sequence length: {len(sequence)}")
        lines.append(f"  Alphabet size: {self.alphabet_size}")
        lines.append("")

        for scale in self.scales:
            r = profile[scale]
            bar = "█" * int((1.0 - r) * 20)
            marker = " ← DOMINANT" if scale == best_scale else ""
            lines.append(f"  Scale {scale:3d}: {r:.4f}  {bar}{marker}")

        lines.append("")
        lines.append(f"  Dominant scale: {best_scale} (ratio={best_ratio:.4f})")
        improvement = self.multi_scale_improvement(sequence)
        if improvement > 1.5:
            lines.append(f"  Multi-scale improvement: {improvement:.2f}× vs scale-1")
            lines.append(f"  → Structure exists at MULTIPLE scales")
        else:
            lines.append(f"  Multi-scale improvement: {improvement:.2f}×")
            lines.append(f"  → Structure primarily at scale-1")

        return '\n'.join(lines)

    def aggregate_at_scale(self, sequence: List[int], scale: int) -> List[int]:
        """Return the aggregated sequence at a given scale."""
        return aggregate_sequence(sequence, scale, self.alphabet_size, self.agg_method)
