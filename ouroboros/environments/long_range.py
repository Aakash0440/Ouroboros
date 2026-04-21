"""
Long-range observation environments requiring PREV(k) for k > 3.

These environments emit integer sequences whose structure can only
be captured by expressions referencing observations from many steps back.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from ouroboros.environments.base import Environment


class LongRangeEnvironment(Environment):
    """
    Base class for environments requiring long-range dependencies.
    """

    def generate(self, length: int, start: int = 0) -> List[int]:
        raise NotImplementedError

    @property
    def max_lag(self) -> int:
        raise NotImplementedError

    @property
    def recurrence_order(self) -> int:
        return self.max_lag

    @property
    def is_linear(self) -> bool:
        return True


# ============================================================
# Tribonacci
# ============================================================

class TribonacciModEnv(LongRangeEnvironment):
    """
    obs[t] = (obs[t-1] + obs[t-2] + obs[t-3]) % modulus
    """

    def __init__(
        self,
        modulus: int = 7,
        seeds: Tuple[int, int, int] = (0, 1, 1),
        name: str = None,
        seed: int = 42,
    ):
        super().__init__(
            alphabet_size=modulus,
            seed=seed,
            name=name or f"TribonacciMod({modulus})",
        )
        self.modulus = modulus
        self.seeds = seeds
        self._cache: List[int] = list(seeds)
        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        while len(self._cache) < length:
            nxt = (
                self._cache[-1]
                + self._cache[-2]
                + self._cache[-3]
            ) % self.modulus
            self._cache.append(nxt)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return 3

    def ground_truth_rule(self) -> str:
        return f"(PREV(1) + PREV(2) + PREV(3)) % {self.modulus}"


# ============================================================
# Lucas Sequence
# ============================================================

class LucasSequenceEnv(LongRangeEnvironment):
    """
    Lucas sequence:
    L(n) = L(n-1) + L(n-2), with seeds (2, 1)
    """

    def __init__(self, modulus: int = 7, name: str = None, seed: int = 42):
        super().__init__(
            alphabet_size=modulus,
            seed=seed,
            name=name or f"LucasMod({modulus})",
        )
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

    def ground_truth_rule(self) -> str:
        return f"(PREV(1) + PREV(2)) % {self.modulus}  [seeds: 2, 1]"


# ============================================================
# General Linear Recurrence
# ============================================================

class LinearRecurrenceEnv(LongRangeEnvironment):
    """
    obs[t] = (c1*obs[t-1] + ... + ck*obs[t-k]) % N
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
        super().__init__(
            alphabet_size=modulus,
            seed=seed,
            name=name or f"LinearRec(order={k},mod={modulus})",
        )

        self.coefficients = coefficients
        self.modulus = modulus

        self._nonzero = [(i + 1, c) for i, c in enumerate(coefficients) if c != 0]

        self._seeds = seeds if seeds is not None else list(range(k))
        self._cache: List[int] = [s % modulus for s in self._seeds]

        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        while len(self._cache) < length:
            nxt = sum(
                c * self._cache[-lag]
                for lag, c in self._nonzero
            ) % self.modulus
            self._cache.append(nxt)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return len(self.coefficients)

    def ground_truth_rule(self) -> str:
        terms = [f"{c}*PREV({l})" for l, c in self._nonzero]
        return f"({' + '.join(terms)}) % {self.modulus}"


# ============================================================
# Sliding Window
# ============================================================

class SlidingWindowEnv(LongRangeEnvironment):
    """
    obs[t] = (PREV(1) + PREV(2) + ... + PREV(W)) % modulus
    """

    def __init__(
        self,
        window_size: int = 7,
        modulus: int = 7,
        seeds: Optional[List[int]] = None,
        name: str = None,
        seed: int = 42,
    ):
        super().__init__(
            alphabet_size=modulus,
            seed=seed,
            name=name or f"SlidingWindow(W={window_size},mod={modulus})",
        )
        self.window_size = window_size
        self.modulus = modulus

        if seeds is None:
            import random
            rng = random.Random(seed)
            self._cache: List[int] = [
                rng.randint(0, modulus - 1)
                for _ in range(window_size)
            ]
        else:
            self._cache = [s % modulus for s in seeds]

        self._extend_cache(2000)

    def _extend_cache(self, length: int) -> None:
        while len(self._cache) < length:
            k = min(self.window_size, len(self._cache))
            nxt = sum(
                self._cache[-i - 1]
                for i in range(k)
            ) % self.modulus
            self._cache.append(nxt)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._extend_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def max_lag(self) -> int:
        return self.window_size

    def ground_truth_rule(self) -> str:
        terms = [f"PREV({i+1})" for i in range(self.window_size)]
        return f"({' + '.join(terms)}) % {self.modulus}"


# ============================================================
# Sparse Autoregressive
# ============================================================

class AutoregressiveEnv(LongRangeEnvironment):
    """
    obs[t] = sum(ai * obs[t - lag_i]) % N
    """

    def __init__(
        self,
        nonzero_lags: List[Tuple[int, int]],
        modulus: int = 7,
        name: str = None,
        seed: int = 42,
    ):
        max_l = max(lag for lag, _ in nonzero_lags)

        super().__init__(
            alphabet_size=modulus,
            seed=seed,
            name=name or f"AR({','.join(str(l) for l, _ in nonzero_lags)},mod={modulus})",
        )

        self.nonzero_lags = sorted(nonzero_lags)
        self.modulus = modulus
        self._max_lag = max_l

        import random
        rng = random.Random(seed)
        self._cache: List[int] = [
            rng.randint(0, modulus - 1)
            for _ in range(max_l)
        ]

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

    def ground_truth_rule(self) -> str:
        terms = [f"{c}*PREV({l})" for l, c in self.nonzero_lags]
        return f"({' + '.join(terms)}) % {self.modulus}"