"""
AlgorithmEnvironment — Input-output pair environments for algorithm synthesis.

Unlike formula environments (SpringMass, ModularArithmetic) where obs[t]
follows a single rule, algorithm environments present a sequence of
(input, output) pairs where the output is computed by a procedure.

An agent that can discover a procedure (not just a formula) must use
STATE nodes — persistent variables that carry state across timesteps.

Environments implemented:
  GCDEnv          — given pairs (a, b), output gcd(a, b)
  FibonacciEnv    — given n, output F(n) using recurrence
  BubbleSortEnv   — given a list state, output next swap position
  BinarySearchEnv — given (target, lo, hi), output next midpoint
  PrimeCountEnv   — given n, output π(n) = number of primes ≤ n

Key insight: these environments output a SINGLE NUMBER per input.
The sequence obs[t] = f(input[t]) where input[t] changes each step.
An agent must discover f, which may require STATE to remember sub-computations.
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional
from ouroboros.environments.base import Environment


class GCDEnv(Environment):
    """
    Input: (a, b) — two positive integers
    Output: gcd(a, b)
    
    Encoding: input is encoded as a single integer: a * MAX_B + b
    Output: gcd(a, b)
    
    The Euclidean algorithm:
      GCD(a, b) = GCD(b, a % b)
      GCD(a, 0) = a
    
    With STATE nodes and MOD, an agent can discover this procedure.
    Without STATE, it can only find special-case GCD formulas.
    """

    MAX_VAL = 20   # a, b in [1, MAX_VAL]
    ENCODING = MAX_VAL + 1

    def __init__(self, seed: int = 42, n_pairs: int = 500):
        super().__init__(name="GCDEnv", seed=seed)
        self._rng = random.Random(seed)
        self._pairs: List[Tuple[int, int]] = []
        self._outputs: List[int] = []
        self._generate_pairs(n_pairs)

    def _generate_pairs(self, n: int) -> None:
        for _ in range(n):
            a = self._rng.randint(1, self.MAX_VAL)
            b = self._rng.randint(1, self.MAX_VAL)
            self._pairs.append((a, b))
            self._outputs.append(math.gcd(a, b))

    def generate(self, length: int, start: int = 0) -> List[int]:
        """Returns gcd(a, b) for each (a,b) pair starting at index start."""
        return self._outputs[start:start + length]

    def get_input(self, t: int) -> Tuple[int, int]:
        """Returns the (a, b) input for timestep t."""
        return self._pairs[t % len(self._pairs)]

    def generate_with_inputs(self, length: int, start: int = 0) -> Tuple[List[int], List[int]]:
        """Returns (inputs_encoded, outputs)."""
        inputs = [a * self.ENCODING + b for a, b in self._pairs[start:start + length]]
        outputs = self._outputs[start:start + length]
        return inputs, outputs

    @property
    def alphabet_size(self) -> int:
        return self.MAX_VAL + 1

    def ground_truth_algorithm(self) -> str:
        return "GCD(a, b) via Euclidean: GCD(a, b) = GCD(b, a%b), GCD(a, 0) = a"


class FibonacciDirectEnv(Environment):
    """
    obs[t] = F(t) = Fibonacci number at position t.
    
    Unlike FibonacciModEnv (which uses PREV(1)+PREV(2)), this environment
    is designed to test STATE nodes:
    An agent can maintain STATE[0] = prev, STATE[1] = curr
    and update them each step.
    
    Without STATE: must use PREV(1)+PREV(2) — already covered
    With STATE: can implement the direct iterative algorithm
    """

    def __init__(self, modulus: int = 1000, seed: int = 42):
        super().__init__(name=f"FibDirect(mod={modulus})", seed=seed)
        self.modulus = modulus
        self._cache: List[int] = []
        self._build_cache(2000)

    def _build_cache(self, n: int) -> None:
        a, b = 0, 1
        for _ in range(n):
            self._cache.append(a % self.modulus)
            a, b = b, (a + b) % self.modulus

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def alphabet_size(self) -> int:
        return self.modulus

    def ground_truth_algorithm(self) -> str:
        return "F(t): a,b=0,1; for _ in range(t): a,b = b, a+b; return a"


class PrimeCountEnv(Environment):
    """
    obs[t] = π(t) = number of primes ≤ t.
    
    The prime counting function. No simple closed form.
    An agent needs ISPRIME + CUMSUM (or STATE) to compute this.
    CUMSUM(ISPRIME(TIME)) starting from t=2 gives π(t).
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="PrimeCountEnv", seed=seed)
        self._cache: List[int] = []
        self._build_cache(1000)

    def _is_prime(self, n: int) -> bool:
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5)+1, 2):
            if n % i == 0: return False
        return True

    def _build_cache(self, n: int) -> None:
        count = 0
        for t in range(n):
            if self._is_prime(t):
                count += 1
            self._cache.append(count)

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def alphabet_size(self) -> int:
        return max(self._cache[:300]) + 2 if self._cache else 50

    def ground_truth_algorithm(self) -> str:
        return "π(t) = #{p ≤ t : p is prime} = CUMSUM(ISPRIME(TIME))[t]"


class CollatzEnv(Environment):
    """
    obs[t] = Collatz stopping time for n=t.
    Collatz(n): while n>1: n = n/2 if even else 3n+1; count steps.
    
    No known closed form. Tests whether agents can discover 
    conditional (IF/THRESHOLD) patterns in the stopping times.
    """

    def __init__(self, seed: int = 42):
        super().__init__(name="CollatzEnv", seed=seed)
        self._cache: List[int] = []
        self._build_cache(500)

    def _collatz_steps(self, n: int) -> int:
        if n <= 1: return 0
        steps = 0
        while n != 1 and steps < 10000:
            n = n // 2 if n % 2 == 0 else 3 * n + 1
            steps += 1
        return steps

    def _build_cache(self, n: int) -> None:
        for t in range(n):
            self._cache.append(self._collatz_steps(t))

    def generate(self, length: int, start: int = 0) -> List[int]:
        if start + length > len(self._cache):
            self._build_cache(start + length + 100)
        return self._cache[start:start + length]

    @property
    def alphabet_size(self) -> int:
        return max(self._cache[:300]) + 2 if self._cache else 200