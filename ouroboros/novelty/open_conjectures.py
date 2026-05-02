"""
OpenConjectureEnvironments — Environments built around unsolved problems.

These environments generate sequences from open mathematical conjectures.
The goal: run OUROBOROS on these sequences and flag any expression that
compresses them significantly better than the current best known formula.

Environments:
  CollatzStoppingTimesEnv — stopping times of the Collatz map
  GoldbachGapEnv          — smallest prime gap at each even number
  PrimeGapEnv             — gaps between consecutive primes
  TwinPrimeDensityEnv     — local density of twin primes
"""

from __future__ import annotations
import math
from typing import List
from ouroboros.environments.base import Environment


class CollatzStoppingTimesEnv(Environment):
    """
    obs[t] = stopping_time(t) = steps to reach 1 under Collatz map.

    Open conjecture: every positive integer eventually reaches 1.
    No closed-form formula for stopping times is known.

    Best known approximation: ~log2(n) * C for some constant C ≈ 6.95.
    If OUROBOROS finds an expression substantially better than this,
    it would be a significant finding.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="CollatzStoppingTimesEnv", seed=seed)
        self._cache: List[int] = []
        self._build_cache(2000)

    def _stopping_time(self, n: int) -> int:
        steps = 0
        while n != 1 and steps < 100000:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        return steps

    def _build_cache(self, n: int) -> None:
        for t in range(n):
            self._cache.append(self._stopping_time(t) if t > 0 else 0)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def alphabet_size(self) -> int:
        return max(self._cache[:500]) + 2

    def current_best_formula(self) -> str:
        return "~6.95 * log2(n)"

    def conjecture(self) -> str:
        return "Every positive integer eventually reaches 1 under the Collatz map"


class PrimeGapEnv(Environment):
    """
    obs[t] = p_{t+1} - p_t  — gap between consecutive primes.

    Cramér's conjecture: gaps are O(log(p)^2).
    No closed form. Heuristics suggest gaps follow a Poisson-like distribution.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="PrimeGapEnv", seed=seed)
        self._cache: List[int] = []
        self._build_cache(500)

    def _sieve(self, n: int) -> List[int]:
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i in range(2, n + 1) if sieve[i]]

    def _build_cache(self, n_primes: int) -> None:
        primes = self._sieve(n_primes * 20)
        for i in range(1, min(len(primes), n_primes + 1)):
            self._cache.append(primes[i] - primes[i-1])

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def alphabet_size(self) -> int:
        return max(self._cache[:200]) + 2 if self._cache else 100

    def conjecture(self) -> str:
        return "Cramér's conjecture: max prime gap near x is O(log(x)^2)"


class TwinPrimeDensityEnv(Environment):
    """
    obs[t] = number of twin prime pairs (p, p+2) with p ≤ t*10.

    Twin prime conjecture: there are infinitely many twin prime pairs.
    No closed form. Hardy-Littlewood conjecture gives asymptotic estimate.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="TwinPrimeDensityEnv", seed=seed)
        self._cache: List[int] = []
        self._build_cache(500)

    def _is_prime(self, n: int) -> bool:
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True

    def _build_cache(self, n: int) -> None:
        count = 0
        for t in range(1, n + 1):
            upper = t * 10
            lower = (t - 1) * 10
            for p in range(max(3, lower), upper):
                if self._is_prime(p) and self._is_prime(p + 2):
                    count += 1
            self._cache.append(count)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def alphabet_size(self) -> int:
        return max(self._cache[:200]) + 2 if self._cache else 100

    def conjecture(self) -> str:
        return "Twin prime conjecture: infinitely many pairs (p, p+2) both prime"