"""
Long-range observation environments requiring PREV(k) for k > 3.

These environments emit integer sequences whose structure can only
be captured by expressions referencing observations from many steps back.

The existing FibonacciModEnv (Day 1) uses lag=2 (PREV(1)+PREV(2)).
These environments require lag 3..20.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from ouroboros.environments.base import Environment


class LongRangeEnvironment(Environment):
    """
    Base class for environments requiring long-range dependencies.
    
    Generates sequences where obs[t] depends on obs[t-1]..obs[t-max_lag].
    The max_lag property tells the search engine how far back to look.
    """
    
    @property
    def max_lag(self) -> int:
        """Maximum lag required to express this environment's rule."""
        raise NotImplementedError
    
    @property
    def recurrence_order(self) -> int:
        """Order of the linear recurrence (same as max_lag for linear recurrences)."""
        return self.max_lag


class TribonacciModEnv(LongRangeEnvironment):
    """
    obs[t] = (obs[t-1] + obs[t-2] + obs[t-3]) % modulus
    
    Generalizes Fibonacci (order 2) to order 3.
    Requires PREV(1) + PREV(2) + PREV(3) expression.
    
    Default: modulus=7, seeds=(0, 1, 1)
    """

    def __init__(
        self,
        modulus: int = 7,
        seeds: Tuple[int, int, int] = (0, 1, 1),
        name: str = None,
        seed: int = 42,
    ):
        super().__init__(name=name or f"TribonacciMod({modulus})", seed=seed)
        self.modulus = modulus
        self.seeds = seeds
        self._cache: List[int] = list(seeds)
        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        while len(self._cache) < length:
            nxt = (self._cache[-1] + self._cache[-2] + self._cache[-3]) % self.modulus
            self._cache.append(nxt)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return 3

    @property
    def alphabet_size(self) -> int:
        return self.modulus

    def ground_truth_rule(self) -> str:
        return f"(PREV(1) + PREV(2) + PREV(3)) % {self.modulus}"


class LucasSequenceEnv(LongRangeEnvironment):
    """
    Lucas sequence: L(n) = L(n-1) + L(n-2) with L(0)=2, L(1)=1
    taken modulo a prime.
    
    Structurally identical to Fibonacci but with different initial conditions.
    Agents must discover the recurrence rule AND the seed values independently.
    
    Default: modulus=7
    """

    def __init__(self, modulus: int = 7, name: str = None, seed: int = 42):
        super().__init__(name=name or f"LucasMod({modulus})", seed=seed)
        self.modulus = modulus
        self._cache: List[int] = [2 % modulus, 1 % modulus]
        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        while len(self._cache) < length:
            nxt = (self._cache[-1] + self._cache[-2]) % self.modulus
            self._cache.append(nxt)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return 2

    @property
    def alphabet_size(self) -> int:
        return self.modulus

    def ground_truth_rule(self) -> str:
        return f"(PREV(1) + PREV(2)) % {self.modulus}  [seeds: 2, 1]"


class LinearRecurrenceEnv(LongRangeEnvironment):
    """
    General linear recurrence: obs[t] = (c1*obs[t-1] + c2*obs[t-2] + ... + ck*obs[t-k]) % N
    
    The most general order-k linear recurrence over Z/NZ.
    
    Example: coefficients=[1, 0, 1], modulus=7 → Tribonacci with c3=0 → same as Fibonacci
    Example: coefficients=[2, 1], modulus=7 → obs[t] = (2*obs[t-1] + obs[t-2]) % 7
    """

    def __init__(
        self,
        coefficients: List[int],
        modulus: int = 7,
        seeds: Optional[List[int]] = None,
        name: str = None,
        seed: int = 42,
    ):
        k = len(coefficients)
        super().__init__(name=name or f"LinearRec(order={k},mod={modulus})", seed=seed)
        self.coefficients = coefficients
        self.modulus = modulus
        self._seeds = seeds if seeds is not None else list(range(k))
        self._cache: List[int] = [s % modulus for s in self._seeds]
        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        k = len(self.coefficients)
        while len(self._cache) < length:
            nxt = sum(
                self.coefficients[i] * self._cache[-i - 1]
                for i in range(k)
            ) % self.modulus
            self._cache.append(nxt)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return len(self.coefficients)

    @property
    def alphabet_size(self) -> int:
        return self.modulus

    def ground_truth_rule(self) -> str:
        terms = [f"{c}*PREV({i+1})" for i, c in enumerate(self.coefficients) if c != 0]
        return f"({' + '.join(terms)}) % {self.modulus}"


class SlidingWindowEnv(LongRangeEnvironment):
    """
    obs[t] = (sum of last W observations) % modulus
    
    Requires knowing observations from the last W steps.
    Default: W=7 (weekly pattern), modulus=7
    
    This models weekly seasonality: the current value is the moving
    sum of the past week, taken modulo 7.
    """

    def __init__(
        self,
        window_size: int = 7,
        modulus: int = 7,
        base_sequence: Optional[List[int]] = None,
        name: str = None,
        seed: int = 42,
    ):
        super().__init__(name=name or f"SlidingWindow(W={window_size},mod={modulus})", seed=seed)
        self.window_size = window_size
        self.modulus = modulus
        
        # Generate a base sequence to sum over
        if base_sequence is None:
            import random
            rng = random.Random(seed)
            self._base = [rng.randint(0, modulus - 1) for _ in range(2000)]
        else:
            self._base = base_sequence * (2000 // len(base_sequence) + 1)
        
        self._cache: List[int] = []
        self._build_cache(2000)

    def _build_cache(self, length: int) -> None:
        for t in range(length):
            window_sum = sum(
                self._base[max(0, t - self.window_size + 1 + i]
                for i in range(min(self.window_size, t + 1))
            ) % self.modulus
            self._cache.append(window_sum)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return self.window_size

    @property
    def alphabet_size(self) -> int:
        return self.modulus

    def ground_truth_rule(self) -> str:
        terms = [f"PREV({i+1})" for i in range(self.window_size)]
        return f"({' + '.join(terms)}) % {self.modulus}"


class AutoregressiveEnv(LongRangeEnvironment):
    """
    AR(p) process over integers modulo N.
    
    obs[t] = (a1*obs[t-1] + a2*obs[t-p] + noise_term) % N
    
    Where the non-zero lags are far apart — e.g., AR(1,7) with
    coefficients at lag 1 and lag 7. This models "same time last week"
    patterns in weekly data.
    
    Default: AR(1,7) with coefficients [1, 0, 0, 0, 0, 0, 1], modulus=7
    """

    def __init__(
        self,
        nonzero_lags: List[Tuple[int, int]],  # [(lag, coefficient), ...]
        modulus: int = 7,
        name: str = None,
        seed: int = 42,
    ):
        max_l = max(lag for lag, _ in nonzero_lags)
        super().__init__(
            name=name or f"AR({','.join(str(l) for l,_ in nonzero_lags)},mod={modulus})",
            seed=seed
        )
        self.nonzero_lags = sorted(nonzero_lags)
        self.modulus = modulus
        self._max_lag = max_l

        # Build cache
        import random
        rng = random.Random(seed)
        self._cache: List[int] = [rng.randint(0, modulus - 1) for _ in range(max_l)]
        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        while len(self._cache) < length:
            val = sum(
                coeff * self._cache[-lag]
                for lag, coeff in self.nonzero_lags
            ) % self.modulus
            self._cache.append(val)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return self._max_lag

    @property
    def alphabet_size(self) -> int:
        return self.modulus

    def ground_truth_rule(self) -> str:
        terms = [f"{c}*PREV({l})" for l, c in self.nonzero_lags]
        return f"({' + '.join(terms)}) % {self.modulus}"