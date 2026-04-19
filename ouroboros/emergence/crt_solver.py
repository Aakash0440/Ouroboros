"""
Analytical CRT Solver — ground truth for the landmark experiment.

The Chinese Remainder Theorem gives us the EXACT joint expression
analytically. This module computes it.

Usage:
    solver = CRTSolver(mod1=7, slope1=3, int1=1, mod2=11, slope2=5, int2=2)
    expr = solver.exact_expression()
    stream = solver.generate_joint_stream(length=2000)
    accuracy = solver.verify_expression(found_expr)

The solver is NOT given to agents — they still search from scratch.
The solver is used to:
    1. Verify what agents find (is it correct?)
    2. Generate the exact stream (so agents have optimal data)
    3. Compute the ground-truth accuracy baseline
    4. Confirm the analytical solution before claiming CRT emerged

The CRT joint expression has the form:
    joint(t) = (joint_slope * t + joint_intercept) mod joint_modulus

where joint_slope and joint_intercept are computed analytically from
the two individual rules using the extended Euclidean algorithm.
"""

import math
from typing import Optional, Tuple, List
from ouroboros.compression.program_synthesis import (
    ExprNode, build_linear_modular, BeamSearchSynthesizer
)
from ouroboros.compression.mdl import compression_ratio, naive_bits
from ouroboros.emergence.crt_detector import (
    gcd, extended_gcd, crt_solution, check_behavioral_crt
)


class CRTSolver:
    """
    Exact CRT solution for two linear modular streams.

    Environment 1: obs1[t] = (slope1 * t + int1) mod mod1
    Environment 2: obs2[t] = (slope2 * t + int2) mod mod2

    Joint stream (interleaved):
        joint[2t]   = obs1[t]
        joint[2t+1] = obs2[t]

    CRT joint expression:
        A single expression j(t) such that:
            j(t) % mod1 == obs1[t] for all t
            j(t) % mod2 == obs2[t] for all t
        This exists iff gcd(mod1, mod2) == 1.

    Args:
        mod1, slope1, int1: Parameters of first environment
        mod2, slope2, int2: Parameters of second environment
    """

    def __init__(
        self,
        mod1: int, slope1: int, int1: int,
        mod2: int, slope2: int, int2: int
    ):
        self.mod1, self.slope1, self.int1 = mod1, slope1, int1
        self.mod2, self.slope2, self.int2 = mod2, slope2, int2
        self.joint_mod = mod1 * mod2

        if gcd(mod1, mod2) != 1:
            raise ValueError(
                f"CRT requires coprime moduli. "
                f"gcd({mod1}, {mod2}) = {gcd(mod1, mod2)} ≠ 1"
            )

    def obs1(self, t: int) -> int:
        return (self.slope1 * t + self.int1) % self.mod1

    def obs2(self, t: int) -> int:
        return (self.slope2 * t + self.int2) % self.mod2

    def joint_value(self, t: int) -> int:
        """
        Compute the exact joint CRT value at timestep t.
        j(t) = CRT_solution(obs1(t) mod mod1, obs2(t) mod mod2)
        """
        a1 = self.obs1(t)
        a2 = self.obs2(t)
        return crt_solution(a1, self.mod1, a2, self.mod2)

    def find_exact_expression(self) -> Tuple[int, int]:
        """
        Find joint_slope and joint_intercept analytically.

        The joint expression is:
            j(t) = (joint_slope * t + joint_intercept) mod joint_mod

        We solve this by computing j(0) and j(1):
            joint_intercept = j(0)
            joint_slope = (j(1) - j(0)) mod joint_mod

        Returns:
            (joint_slope, joint_intercept)
        """
        j0 = self.joint_value(0)
        j1 = self.joint_value(1)
        joint_intercept = j0
        joint_slope = (j1 - j0) % self.joint_mod
        return joint_slope, joint_intercept

    def exact_expression(self) -> ExprNode:
        """
        Build the exact CRT joint ExprNode.
        This is what agents SHOULD find from compression pressure.
        """
        slope, intercept = self.find_exact_expression()
        return build_linear_modular(slope, intercept, self.joint_mod)

    def generate_joint_stream(self, length: int) -> List[int]:
        """
        Generate the exact joint CRT stream.
        All values are in [0, joint_mod).
        """
        return [self.joint_value(t) for t in range(length)]

    def verify_expression(
        self,
        expr: ExprNode,
        test_length: int = 500
    ) -> Tuple[float, float, float]:
        """
        Measure how well an expression captures the CRT structure.

        Returns:
            (overall_accuracy, mod1_accuracy, mod2_accuracy)

        All three should be > 0.95 for a "CRT-equivalent" expression.
        """
        correct_all = 0
        correct_mod1 = 0
        correct_mod2 = 0

        for t in range(test_length):
            pred = expr.evaluate(t) % self.joint_mod
            true_j = self.joint_value(t)
            true_m1 = self.obs1(t)
            true_m2 = self.obs2(t)

            if pred == true_j:
                correct_all += 1
            if pred % self.mod1 == true_m1:
                correct_mod1 += 1
            if pred % self.mod2 == true_m2:
                correct_mod2 += 1

        n = test_length
        return correct_all / n, correct_mod1 / n, correct_mod2 / n

    def compression_ratio_exact(self, length: int = 1000) -> float:
        """
        Compression ratio of the exact CRT expression on its own stream.
        This is the THEORETICAL MINIMUM — what agents are trying to achieve.
        """
        stream = self.generate_joint_stream(length)
        expr = self.exact_expression()
        preds = [expr.evaluate(t) % self.joint_mod for t in range(length)]
        from ouroboros.compression.mdl import MDLCost, naive_bits as nb_fn
        mdl = MDLCost()
        cost = mdl.total_cost(expr.to_bytes(), preds, stream, self.joint_mod)
        nb = nb_fn(stream, self.joint_mod)
        return cost / nb if nb > 0 else 1.0

    def report(self) -> str:
        slope, intercept = self.find_exact_expression()
        expr = self.exact_expression()
        cr = self.compression_ratio_exact(500)
        lines = [
            f"CRTSolver: mod1={self.mod1}, mod2={self.mod2}",
            f"  Rule 1: ({self.slope1}t + {self.int1}) mod {self.mod1}",
            f"  Rule 2: ({self.slope2}t + {self.int2}) mod {self.mod2}",
            f"  Joint modulus: {self.joint_mod}",
            f"  Exact joint slope: {slope}",
            f"  Exact joint intercept: {intercept}",
            f"  Exact expression: {expr.to_string()!r}",
            f"  Theoretical compression ratio: {cr:.6f}",
        ]
        return '\n'.join(lines)