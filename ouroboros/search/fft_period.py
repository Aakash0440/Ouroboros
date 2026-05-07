"""
FFTPeriodFinder — detects dominant periodic cycles in integer sequences
using FFT magnitude spectrum analysis.

Used by benchmarkC.py and the HierarchicalSearchRouter classifier
to identify PERIODIC family sequences before beam search.
"""

from __future__ import annotations
import math
from typing import List, Optional
import numpy as np


class FFTPeriodFinder:
    """
    Detects dominant periods in a sequence via FFT magnitude spectrum.

    Usage:
        finder = FFTPeriodFinder()
        periods = finder.find_periods(values, top_k=3)
        # Returns list of (period_months, strength) sorted by strength desc
    """

    def __init__(
        self,
        min_period: int = 3,
        max_period_fraction: float = 0.5,
        smoothing: bool = True,
    ):
        """
        min_period            : ignore periods shorter than this (noise floor)
        max_period_fraction   : ignore periods longer than len*fraction (trend artifacts)
        smoothing             : apply Hann window before FFT (reduces spectral leakage)
        """
        self.min_period = min_period
        self.max_period_fraction = max_period_fraction
        self.smoothing = smoothing

    def find_periods(
        self,
        sequence: List[int],
        top_k: int = 3,
    ) -> List[tuple]:
        """
        Returns top_k dominant periods as list of (period, strength) tuples,
        sorted by strength descending.

        period   : estimated cycle length in samples (months, steps, etc.)
        strength : normalised FFT magnitude [0, 1]
        """
        n = len(sequence)
        if n < 6:
            return []

        x = np.array(sequence, dtype=float)

        # Remove linear trend to avoid DC/trend components dominating
        t = np.arange(n)
        slope, intercept = np.polyfit(t, x, 1)
        x = x - (slope * t + intercept)

        # Hann window to reduce spectral leakage
        if self.smoothing:
            x = x * np.hanning(n)

        # FFT magnitude spectrum (positive frequencies only)
        spectrum = np.abs(np.fft.rfft(x))
        freqs    = np.fft.rfftfreq(n)  # cycles per sample

        # Convert to periods, filter valid range
        max_period = int(n * self.max_period_fraction)
        results = []

        for i, (freq, mag) in enumerate(zip(freqs, spectrum)):
            if freq <= 0:
                continue
            period = 1.0 / freq
            if period < self.min_period or period > max_period:
                continue
            results.append((period, mag))

        if not results:
            return []

        # Normalise magnitudes
        max_mag = max(r[1] for r in results)
        results = [(p, m / max_mag) for p, m in results]

        # Sort by strength, return top_k with period rounded to nearest int
        results.sort(key=lambda x: x[1], reverse=True)
        top = [(int(round(p)), round(s, 4)) for p, s in results[:top_k]]

        return top

    def dominant_period(self, sequence: List[int]) -> Optional[int]:
        """Returns only the single strongest period, or None if none found."""
        periods = self.find_periods(sequence, top_k=1)
        return periods[0][0] if periods else None

    def has_periodicity(
        self,
        sequence: List[int],
        strength_threshold: float = 0.3,
    ) -> bool:
        """Returns True if any period has normalised strength above threshold."""
        periods = self.find_periods(sequence, top_k=1)
        return bool(periods and periods[0][1] >= strength_threshold)