"""JointEnvironment — generates CRT-joint sequences for two moduli."""
from __future__ import annotations
import random
from typing import List


def _crt_solve(a1: int, m1: int, a2: int, m2: int) -> int:
    """Return smallest non-negative x s.t. x≡a1(mod m1) and x≡a2(mod m2)."""
    diff = (a2 - a1) % m2
    inv_m1 = pow(m1, -1, m2)
    t = (diff * inv_m1) % m2
    return a1 + m1 * t


class JointEnvironment:
    """
    Generates a sequence whose t-th element is the unique CRT solution in
    [0, mod1*mod2) satisfying:
        x ≡ (slope1*t + int1)  (mod mod1)
        x ≡ (slope2*t + int2)  (mod mod2)
    """

    def __init__(
        self,
        mod1: int = 7,
        slope1: int = 3,
        int1: int = 1,
        mod2: int = 11,
        slope2: int = 5,
        int2: int = 2,
        seed: int = 0,
    ):
        self.mod1 = mod1
        self.slope1 = slope1
        self.int1 = int1
        self.mod2 = mod2
        self.slope2 = slope2
        self.int2 = int2
        self.alphabet_size = mod1 * mod2
        self._rng = random.Random(seed)

    def generate(self, length: int, start: int = 0) -> List[int]:
        out = []
        for i in range(length):
            t = start + i
            a1 = (self.slope1 * t + self.int1) % self.mod1
            a2 = (self.slope2 * t + self.int2) % self.mod2
            out.append(_crt_solve(a1, self.mod1, a2, self.mod2))
        return out