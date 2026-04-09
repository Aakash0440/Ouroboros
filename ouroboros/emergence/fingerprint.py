# ouroboros/emergence/fingerprint.py

"""
Behavioral fingerprinting for expression equivalence.

WHY NOT STRING EQUALITY:
Two expressions can be behaviorally identical but textually different.
    "(t * 3 + 1) mod 7"  and  "(1 + t * 3) mod 7"
Both predict the same sequence. Both should count as the same axiom.

WHY NOT SYMBOLIC SIMPLIFICATION:
Symbolic simplification (algebra) is complex to implement correctly.
Behavioral fingerprinting is simpler and sufficient:
    fingerprint(expr) = tuple of predictions on t=0..TEST_LENGTH-1

Two expressions with identical fingerprints are treated as one axiom.
Two expressions with different fingerprints are different axioms.

False negative risk: Two expressions with identical predictions on
TEST_LENGTH symbols but different behavior elsewhere. Mitigation:
use TEST_LENGTH=200 (probability of collision is negligible for
the expression sizes we're working with).

Usage:
    fp = behavioral_fingerprint(expr, alphabet_size=7, length=200)
    # Returns a 200-tuple of integers in 0..6
"""

from typing import Tuple
from ouroboros.compression.program_synthesis import ExprNode


def behavioral_fingerprint(
    expr: ExprNode,
    alphabet_size: int,
    length: int = 200
) -> Tuple[int, ...]:
    """
    Compute the behavioral fingerprint of an expression.

    The fingerprint is the tuple of predictions on t=0..length-1,
    clamped to 0..alphabet_size-1.

    Two expressions are behaviorally equivalent iff their fingerprints
    are identical.

    Args:
        expr: The symbolic expression
        alphabet_size: Clamp predictions to 0..alphabet_size-1
        length: Number of timesteps to evaluate (default 200)

    Returns:
        Tuple of length `length` with integer predictions
    """
    return tuple(
        expr.evaluate(t) % alphabet_size
        for t in range(length)
    )


def expressions_equivalent(
    expr1: ExprNode,
    expr2: ExprNode,
    alphabet_size: int,
    length: int = 200
) -> bool:
    """
    Check if two expressions are behaviorally equivalent.

    True iff they predict the same sequence for t=0..length-1.
    """
    return (
        behavioral_fingerprint(expr1, alphabet_size, length) ==
        behavioral_fingerprint(expr2, alphabet_size, length)
    )


def compression_fingerprint(
    expr: ExprNode,
    test_sequence: list,
    alphabet_size: int
) -> float:
    """
    Measure how well an expression predicts a test sequence.

    Returns: error_rate in [0, 1]  (0 = perfect, 1 = all wrong)

    Used to verify that a proposed axiom actually captures
    the environment's structure — not just any pattern.
    """
    n = len(test_sequence)
    if n == 0:
        return 1.0
    preds = expr.predict_sequence(n, alphabet_size)
    errors = sum(p != a for p, a in zip(preds, test_sequence))
    return errors / n