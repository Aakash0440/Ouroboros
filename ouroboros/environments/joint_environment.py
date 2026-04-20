"""
JointEnvironment — interleaves two observation environments.

For the CRT experiment, we interleave two modular arithmetic streams:
    Stream 1: (slope1 * t + int1) mod mod1
    Stream 2: (slope2 * t + int2) mod mod2

The joint stream alternates: [s1[0], s2[0], s1[1], s2[1], ...]
Or concatenates in blocks: [s1[0..n], s2[0..n], s1[n..2n], ...]

An agent that can predict BOTH streams from a SINGLE expression has
derived a relationship between the two modular systems.
By CRT: if gcd(mod1, mod2) = 1, then a joint expression of the form
    (slope * t + intercept) mod (mod1 * mod2)
predicts both streams simultaneously.

This is what we test: does compression pressure cause agents to discover
the CRT joint expression?

Args:
    env1: First environment
    env2: Second environment
    interleave: If True, interleave symbol-by-symbol; else concatenate
"""

from typing import List, Tuple
import numpy as np
from ouroboros.environments.base import ObservationEnvironment
from ouroboros.environments.structured import ModularArithmeticEnv


class JointEnvironment(ObservationEnvironment):
    """
    Interleaved dual-environment for CRT experiment.

    The joint stream is env1 and env2 interleaved:
        joint[2t]   = env1[t]   (even positions: env1 symbols)
        joint[2t+1] = env2[t]   (odd positions: env2 symbols)

    Both environments must have the same alphabet size (use LCM or max).

    For CRT experiment:
        env1 = ModularArith(7, slope1, int1)
        env2 = ModularArith(11, slope2, int2)
        joint alphabet_size = 7 * 11 = 77

    An expression that correctly predicts ALL positions of the joint stream
    has effectively captured the relationship between mod-7 and mod-11 arithmetic.
    """

    def __init__(
        self,
        env1: ModularArithmeticEnv,
        env2: ModularArithmeticEnv,
        seed: int = 42
    ):
        # Joint alphabet = product of the two moduli (for CRT encoding)
        joint_alpha = env1.alphabet_size * env2.alphabet_size
        super().__init__(alphabet_size=joint_alpha, seed=seed)
        self.env1 = env1
        self.env2 = env2
        self.mod1 = env1.modulus
        self.mod2 = env2.modulus
        self.joint_modulus = self.mod1 * self.mod2

    def _generate_stream(self, length: int) -> List[int]:
        """
        Interleaved joint stream.

        Encoding: joint[2t] = env1[t], joint[2t+1] = env2[t]
        We encode each pair as: env1_val * mod2 + env2_val
        This is the CRT encoding — exactly as the theorem describes.
        """
        n_pairs = length // 2
        self.env1.reset(n_pairs)
        self.env2.reset(n_pairs)
        s1 = self.env1.peek_all()[:n_pairs]
        s2 = self.env2.peek_all()[:n_pairs]

        # Interleaved
        stream = []
        for v1, v2 in zip(s1, s2):
            stream.append(v1)
            stream.append(v2)

        return stream[:length]

    def decode_to_pairs(
        self,
        sequence: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Split interleaved stream back into env1 and env2 components."""
        s1 = sequence[0::2]
        s2 = sequence[1::2]
        return s1, s2

    def verify_crt_expression(
        self,
        expr,
        test_length: int = 100
    ) -> Tuple[float, float]:
        """
        Test if an expression captures CRT structure.

        Returns (env1_accuracy, env2_accuracy) — how well the expression
        predicts each sub-stream independently.

        If both accuracies > 0.90, the expression has captured CRT.
        """
        self.reset(test_length * 2)
        stream = self.peek_all()
        s1, s2 = self.decode_to_pairs(stream)

        # Test predictions against each stream
        acc1 = 0.0
        acc2 = 0.0
        n = min(len(s1), len(s2), test_length)

        for t in range(n):
            pred = expr.evaluate(t) % self.alphabet_size
            # Check if prediction correctly captures both components
            pred_mod1 = pred % self.mod1
            pred_mod2 = pred % self.mod2
            if pred_mod1 == s1[t]:
                acc1 += 1
            if pred_mod2 == s2[t]:
                acc2 += 1

        return acc1 / n, acc2 / n