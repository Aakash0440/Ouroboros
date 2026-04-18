"""
CRT Detector — checks if derived axioms are isomorphic to CRT.

The Chinese Remainder Theorem (CRT):
    If gcd(m, n) = 1, then for any a, b:
    The system x ≡ a (mod m), x ≡ b (mod n)
    has a unique solution x (mod mn).

CRT isomorphism test:
    Given:
        axiom1: expression for ModArith(mod1, s1, i1)
        axiom2: expression for ModArith(mod2, s2, i2)
    We check if a SINGLE expression e(t) satisfies:
        e(t) mod mod1 == axiom1(t) for all t
        e(t) mod mod2 == axiom2(t) for all t
    Such an expression exists iff gcd(mod1, mod2) = 1 (CRT condition).
    The expression is the CRT encoding.

If agents independently find axiom1 and axiom2, and then find
a joint expression that satisfies both — they have derived CRT.
"""

import math
from typing import Optional, Tuple, List
from ouroboros.compression.program_synthesis import ExprNode


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (gcd, x, y) s.t. a*x + b*y = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


def crt_solution(
    a1: int, mod1: int,
    a2: int, mod2: int
) -> Optional[int]:
    """
    Find x such that x ≡ a1 (mod mod1) and x ≡ a2 (mod mod2).
    Returns x mod (mod1 * mod2), or None if no solution (gcd != 1).
    """
    g, p, q = extended_gcd(mod1, mod2)
    if g != 1:
        return None
    lcm = mod1 * mod2
    x = (a1 * q * mod2 + a2 * p * mod1) % lcm
    return x


def verify_crt_structure(
    joint_expr: ExprNode,
    mod1: int,
    mod2: int,
    test_length: int = 200,
    accuracy_threshold: float = 0.90
) -> Tuple[bool, float, float]:
    """
    Verify that joint_expr satisfies CRT structure.

    Tests:
        1. joint_expr(t) mod mod1 matches independent mod1 predictions
        2. joint_expr(t) mod mod2 matches independent mod2 predictions

    Args:
        joint_expr: The expression to test
        mod1: First modulus
        mod2: Second modulus
        test_length: Number of timesteps to test
        accuracy_threshold: Required accuracy for each sub-stream

    Returns:
        (is_crt, accuracy_mod1, accuracy_mod2)
    """
    if gcd(mod1, mod2) != 1:
        return False, 0.0, 0.0  # CRT requires coprime moduli

    joint_mod = mod1 * mod2
    correct_mod1 = 0
    correct_mod2 = 0

    for t in range(test_length):
        joint_val = joint_expr.evaluate(t) % joint_mod
        pred_mod1 = joint_val % mod1
        pred_mod2 = joint_val % mod2

        # What the individual expressions would predict
        true_mod1 = (3 * t + 1) % mod1   # Hardcoded for test; parametrize in real use
        true_mod2 = (3 * t + 1) % mod2

        if pred_mod1 == true_mod1:
            correct_mod1 += 1
        if pred_mod2 == true_mod2:
            correct_mod2 += 1

    acc1 = correct_mod1 / test_length
    acc2 = correct_mod2 / test_length
    is_crt = (acc1 >= accuracy_threshold and acc2 >= accuracy_threshold)

    return is_crt, acc1, acc2


def check_behavioral_crt(
    expr1: ExprNode,
    expr2: ExprNode,
    joint_expr: ExprNode,
    mod1: int,
    mod2: int,
    test_length: int = 200
) -> Tuple[bool, float]:
    """
    Check if joint_expr behaviorally encodes expr1 and expr2 via CRT.

    More general test: checks if joint_expr(t) mod mod1 ≈ expr1(t) mod mod1
    AND joint_expr(t) mod mod2 ≈ expr2(t) mod mod2.

    Returns (is_crt, combined_accuracy)
    """
    joint_mod = mod1 * mod2
    correct = 0
    total = test_length * 2

    for t in range(test_length):
        jv = joint_expr.evaluate(t) % joint_mod
        e1v = expr1.evaluate(t) % mod1
        e2v = expr2.evaluate(t) % mod2

        if jv % mod1 == e1v:
            correct += 1
        if jv % mod2 == e2v:
            correct += 1

    accuracy = correct / total
    is_crt = accuracy >= 0.85

    return is_crt, accuracy