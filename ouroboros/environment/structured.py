# ouroboros/environment/structured.py

"""
Concrete observation environments with hidden mathematical structure.

ENVIRONMENT DIFFICULTY (hardest to discover rule → hardest to compress):

1. BinaryRepeatEnv     TRIVIAL   — 0,1,0,1,0,1,...  rule: t mod 2
2. ModularArithmeticEnv MEDIUM   — (3t+1) mod 7 hidden rule
3. FibonacciModEnv      HARD     — Fib(t) mod 11 — recurrence + mod
4. PrimeSequenceEnv     VERY HARD — is_prime(t)? — no algebraic form
5. NoiseEnv             IMPOSSIBLE — random — no rule exists

WHY THIS ORDER MATTERS:
If agents compress NoiseEnv, something is broken.
If agents fail on BinaryRepeatEnv, something is broken.
The interesting regime is 2 and 3 — where algebraic primitives
(t, +, *, mod) are SUFFICIENT to describe the rule.

THE LANDMARK: ModularArithmeticEnv(7, 3, 1)
An agent that independently finds "(t * 3 + 1) mod 7" on this
environment has discovered modular arithmetic from compression alone.
"""

import numpy as np
from typing import List
from ouroboros.environment.base import ObservationEnvironment


class BinaryRepeatEnv(ObservationEnvironment):
    """
    Stream: 0, 1, 0, 1, 0, 1, ...  (alternating)
    Hidden rule: t mod 2
    Optimal compression: ~0.001 bits/symbol (near-perfect)

    USE: Sanity check. If agents can't compress this, nothing works.
    """

    def __init__(self, seed: int = 42):
        super().__init__(alphabet_size=2, seed=seed)

    def _generate_stream(self, length: int) -> List[int]:
        return [t % 2 for t in range(length)]

    def known_rule(self) -> str:
        return "t mod 2"


class ModularArithmeticEnv(ObservationEnvironment):
    """
    Stream: (slope * t + intercept) mod modulus  for t = 0, 1, 2, ...

    The modulus, slope, and intercept are ALL HIDDEN from agents.
    Agents must discover the rule purely from the symbol sequence.

    An agent that finds the rule compresses to ~log2(N^2) bits TOTAL
    (just to specify slope and intercept) instead of log2(N) per symbol.

    THIS IS THE LANDMARK ENVIRONMENT.
    When an agent finds "(t * 3 + 1) mod 7", it has discovered
    modular arithmetic from compression pressure alone.

    Args:
        modulus: N (hidden)
        slope: a (hidden) — typically coprime to modulus
        intercept: b (hidden)
        seed: Random seed (not used for generation — only for base class)
    """

    def __init__(
        self,
        modulus: int = 7,
        slope: int = 3,
        intercept: int = 1,
        seed: int = 42
    ):
        super().__init__(alphabet_size=modulus, seed=seed)
        self.modulus = modulus
        self.slope = slope
        self.intercept = intercept

    def _generate_stream(self, length: int) -> List[int]:
        return [(self.slope * t + self.intercept) % self.modulus
                for t in range(length)]

    def known_rule(self) -> str:
        return f"({self.slope} * t + {self.intercept}) mod {self.modulus}"

    def optimal_bits(self) -> float:
        """
        Theoretical minimum bits if rule is known.
        ≈ 2 * log2(modulus) to specify (slope, intercept)
        + 0 per symbol (rule predicts perfectly)
        """
        import math
        return 2.0 * math.log2(self.modulus)

    def check_expression(self, expr) -> bool:
        """
        Check if an expression correctly captures the rule.
        Tests first 100 symbols.
        """
        for t in range(100):
            expected = (self.slope * t + self.intercept) % self.modulus
            got = expr.evaluate(t) % self.modulus
            if got != expected:
                return False
        return True


class FibonacciModEnv(ObservationEnvironment):
    """
    Stream: Fibonacci(t) mod N  for t = 0, 1, 2, ...

    Hidden rules: (1) Fibonacci recurrence: F(t) = F(t-1) + F(t-2)
                  (2) Modular reduction: F(t) mod N

    An agent needs BOTH rules to compress this well.
    Much harder than ModularArithmeticEnv because the recurrence
    cannot be expressed as a simple linear formula in t.

    Note: Fibonacci sequences mod N are periodic (Pisano period).
    An n-gram table with the right period length achieves good compression.
    The real win comes from finding the recurrence rule.

    Args:
        modulus: N (hidden)
    """

    def __init__(self, modulus: int = 11, seed: int = 42):
        super().__init__(alphabet_size=modulus, seed=seed)
        self.modulus = modulus

    def _generate_stream(self, length: int) -> List[int]:
        stream = []
        a, b = 0, 1
        for _ in range(length):
            stream.append(a % self.modulus)
            a, b = b, (a + b) % self.modulus
        return stream

    def known_rule(self) -> str:
        return f"Fibonacci(t) mod {self.modulus}"

    def pisano_period(self) -> int:
        """
        Compute Pisano period π(N) — the period of Fibonacci mod N.
        An n-gram agent that finds this period achieves good compression.
        """
        prev, curr = 0, 1
        for i in range(1, self.modulus * self.modulus * 6 + 1):
            prev, curr = curr, (prev + curr) % self.modulus
            if prev == 0 and curr == 1:
                return i
        return -1  # Should not reach here for reasonable N


class PrimeSequenceEnv(ObservationEnvironment):
    """
    Stream: is_prime(t) ? 1 : 0  for t = 0, 1, 2, ...

    Hidden rule: primality testing.
    There is NO short algebraic expression for this.
    The best known compression uses the sieve — which n-gram agents
    cannot represent.

    USE: Test that the system doesn't hallucinate structure.
    Agents should NOT find a good symbolic expression here.
    This validates that emergence is real, not noise.

    Expected compression ratio: 0.90–0.97 (barely better than random)
    """

    def __init__(self, seed: int = 42):
        super().__init__(alphabet_size=2, seed=seed)

    def _generate_stream(self, length: int) -> List[int]:
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            if n == 2:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(n ** 0.5) + 1, 2):
                if n % i == 0:
                    return False
            return True

        return [int(is_prime(t)) for t in range(length)]

    def known_rule(self) -> str:
        return "is_prime(t)  [no short expression]"


class NoiseEnv(ObservationEnvironment):
    """
    Stream: uniformly random symbols from alphabet.

    No hidden structure — truly random.
    Incompressible by any agent (up to statistical fluctuations).

    USE: UPPER BOUND BASELINE.
    Any agent claiming to compress this is broken.
    Expected compression ratio: 0.95–1.05

    Args:
        alphabet_size: Size of symbol alphabet
    """

    def __init__(self, alphabet_size: int = 4, seed: int = 42):
        super().__init__(alphabet_size=alphabet_size, seed=seed)

    def _generate_stream(self, length: int) -> List[int]:
        return list(self.rng.integers(0, self.alphabet_size, size=length))

    def known_rule(self) -> str:
        return "NONE — random stream"


class MultiScaleEnv(ObservationEnvironment):
    """
    Stream: superposition of two periodic patterns at different scales.

    Pattern = (slow_pattern * 2 + fast_pattern) + noise
    where slow_pattern = t mod slow_period (changes every slow_period steps)
          fast_pattern = t mod fast_period (changes every fast_period steps)

    An agent that only sees one scale misses part of the structure.
    An agent that finds BOTH patterns compresses dramatically better.

    This is the Phase 3 environment — used for causal hierarchy tests.

    Args:
        slow_period: Period of slow-scale pattern (default 100)
        fast_period: Period of fast-scale pattern (default 7)
        noise_fraction: Fraction of symbols randomly flipped (default 0.03)
    """

    def __init__(
        self,
        slow_period: int = 100,
        fast_period: int = 7,
        noise_fraction: float = 0.03,
        seed: int = 42
    ):
        super().__init__(alphabet_size=4, seed=seed)
        self.slow_period = slow_period
        self.fast_period = fast_period
        self.noise_fraction = noise_fraction

    def _generate_stream(self, length: int) -> List[int]:
        slow = np.array([t % self.slow_period for t in range(length)]) % 2
        fast = np.array([t % self.fast_period for t in range(length)]) % 2
        combined = (slow * 2 + fast).astype(int)  # values in {0,1,2,3}

        # Inject noise
        noise_mask = self.rng.random(length) < self.noise_fraction
        noise_vals = self.rng.integers(0, 4, size=length)
        combined[noise_mask] = noise_vals[noise_mask]

        return combined.tolist()

    def known_rule(self) -> str:
        return (
            f"(t mod {self.slow_period}) mod 2 * 2 "
            f"+ (t mod {self.fast_period}) mod 2"
            f"  [+ {self.noise_fraction*100:.0f}% noise]"
        )


class PiecewiseModEnv(ObservationEnvironment):
    """
    Stream alternating between two modular rules every `switch_period` steps.

    Pattern:
        if (t // switch_period) % 2 == 0: obs[t] = (s1*t + i1) mod mod1
        else:                              obs[t] = (s2*t + i2) mod mod2

    Requires IF node to discover. Without IF, best compression ~0.50.
    With IF correctly found, compression ~0.005.

    This environment verifies that IF-containing expressions work.

    Args:
        switch_period: How many steps before switching rules
        mod1, slope1, intercept1: First rule parameters
        mod2, slope2, intercept2: Second rule parameters
    """

    def __init__(
        self,
        switch_period: int = 10,
        mod1: int = 5, slope1: int = 2, intercept1: int = 1,
        mod2: int = 7, slope2: int = 3, intercept2: int = 2,
        seed: int = 42
    ):
        alphabet = max(mod1, mod2)
        super().__init__(alphabet_size=alphabet, seed=seed)
        self.switch_period = switch_period
        self.mod1, self.slope1, self.intercept1 = mod1, slope1, intercept1
        self.mod2, self.slope2, self.intercept2 = mod2, slope2, intercept2

    def _generate_stream(self, length: int) -> List[int]:
        stream = []
        for t in range(length):
            phase = (t // self.switch_period) % 2
            if phase == 0:
                val = (self.slope1 * t + self.intercept1) % self.mod1
            else:
                val = (self.slope2 * t + self.intercept2) % self.mod2
            stream.append(val % self.alphabet_size)
        return stream


class RecurrenceEnv(ObservationEnvironment):
    """
    General linear recurrence environment.

    obs[t] = (sum(coeff[k] * obs[t-k-1] for k in 0..order-1) + const) mod modulus

    Fibonacci: coefficients=[1,1], const=0, modulus=N
    Tribonacci: coefficients=[1,1,1], const=0, modulus=N
    With PREV(1), PREV(2) etc., agents can discover these.

    Args:
        coefficients: List of coefficients for obs[t-1], obs[t-2], ...
        const: Additive constant
        modulus: Modulus for reduction
        seed_values: Initial values (default: [0, 1, 1, ...])
    """

    def __init__(
        self,
        coefficients: List[int] = None,
        const: int = 0,
        modulus: int = 11,
        seed_values: List[int] = None,
        seed: int = 42
    ):
        super().__init__(alphabet_size=modulus, seed=seed)
        self.coefficients = coefficients or [1, 1]
        self.const = const
        self.modulus = modulus
        self.seed_values = seed_values or [0, 1]

    def _generate_stream(self, length: int) -> List[int]:
        order = len(self.coefficients)
        # Pad seed to at least order length
        history = list(self.seed_values[:order])
        while len(history) < order:
            history.append(0)

        stream = list(history[:length])

        while len(stream) < length:
            t = len(stream)
            val = self.const
            for k, coeff in enumerate(self.coefficients):
                if t - k - 1 >= 0:
                    val += coeff * stream[t - k - 1]
            stream.append(val % self.modulus)

        return stream[:length]