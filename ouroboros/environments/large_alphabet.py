"""
Large-alphabet environments for testing scalability.

These environments operate over alphabets of size N > 100,
previously infeasible due to beam search scoring overhead.
With SparseBeamSearch (Day 27), they become tractable.
"""

from __future__ import annotations
from typing import List
from ouroboros.environments.base import Environment


class CRTLargeEnv(Environment):
    """
    Joint CRT environment with large alphabet.

    Interleaves two modular streams with moduli p and q.
    The joint alphabet has size p*q.

    At (p=13, q=17): alphabet_size=221
    At (p=13, q=11, r=7): alphabet_size=1001 (three-way CRT)

    obs[t] = (slope1 * t + int1) % p  for even t
    obs[t] = (slope2 * t + int2) % q  for odd t
    (expressed in the joint alphabet Z_{p*q})
    """

    def __init__(
        self,
        mod1: int = 13,
        slope1: int = 3,
        int1: int = 1,
        mod2: int = 17,
        slope2: int = 5,
        int2: int = 2,
        name: str = None,
        seed: int = 42,
    ):
        joint_mod = mod1 * mod2
        super().__init__(
            alphabet_size=joint_mod,
            name=name or f"CRTLarge({mod1}×{mod2})",
            seed=seed,
        )
        self.mod1 = mod1
        self.slope1 = slope1
        self.int1 = int1
        self.mod2 = mod2
        self.slope2 = slope2
        self.int2 = int2
        self.joint_mod = joint_mod

    def generate(self, length: int, start: int = 0) -> List[int]:
        result = []
        for t in range(start, start + length):
            if t % 2 == 0:
                val1 = (self.slope1 * t + self.int1) % self.mod1
                joint_val = val1 * self.mod2
            else:
                val2 = (self.slope2 * t + self.int2) % self.mod2
                joint_val = val2 * self.mod1
            result.append(joint_val % self.joint_mod)
        return result

    @property
    def ground_truth_joint_mod(self) -> int:
        return self.joint_mod


class TripleCRTEnv(Environment):
    """
    Three-way CRT environment: alphabet = p*q*r.

    At (p=7, q=11, r=13): alphabet = 1001
    Requires SparseBeamSearch with GPU for practical runtime.
    """

    def __init__(
        self,
        mod1: int = 7,
        mod2: int = 11,
        mod3: int = 13,
        seed: int = 42,
    ):
        joint = mod1 * mod2 * mod3
        super().__init__(
            alphabet_size=joint,
            name=f"TripleCRT({mod1}×{mod2}×{mod3})",
            seed=seed,
        )
        self.mod1, self.mod2, self.mod3 = mod1, mod2, mod3
        self.joint_mod = joint

    def generate(self, length: int, start: int = 0) -> List[int]:
        result = []
        for t in range(start, start + length):
            cycle = t % 3
            if cycle == 0:
                val = ((3 * t + 1) % self.mod1) * (self.mod2 * self.mod3)
            elif cycle == 1:
                val = ((5 * t + 2) % self.mod2) * (self.mod1 * self.mod3)
            else:
                val = ((7 * t + 3) % self.mod3) * (self.mod1 * self.mod2)
            result.append(val % self.joint_mod)
        return result