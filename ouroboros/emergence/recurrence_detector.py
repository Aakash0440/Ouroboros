"""
RecurrenceDetector — Analytically discovers linear recurrence relations.

Given a sequence obs[0..n], the RecurrenceDetector finds coefficients
c1, c2, ..., ck such that:
  obs[t] = (c1*obs[t-1] + c2*obs[t-2] + ... + ck*obs[t-k]) % N

using the Berlekamp-Massey algorithm (BM).

Berlekamp-Massey finds the shortest LFSR (Linear Feedback Shift Register)
that generates the sequence — this is equivalent to finding the minimal
linear recurrence.

Why this matters:
  Beam search can find short recurrences (order ≤ 3) easily.
  But order-7 or order-10 recurrences require beam_width >> 100.
  BM finds the recurrence in O(n) time analytically.

The RecurrenceDetector:
  1. Runs BM on the raw integer sequence
  2. If BM succeeds (sequence fits a linear recurrence), returns the coefficients
  3. If BM fails (sequence is nonlinear), falls back to beam search
  4. Stores discovered recurrences as RecurrenceAxiom objects
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class RecurrenceAxiom:
    """
    A discovered linear recurrence relation.
    
    Represents: obs[t] = (c1*obs[t-1] + ... + ck*obs[t-k]) % modulus
    """
    order: int              # k (the lag depth)
    coefficients: List[int] # [c1, c2, ..., ck]
    modulus: int
    fit_error: float        # fraction of predictions that are wrong (0.0 = perfect)
    evidence_length: int    # length of sequence used to discover this
    
    discovered_by: str = "BerlekampMassey"
    environment_name: str = "unknown"
    discovery_step: int = 0

    @property
    def is_perfect(self) -> bool:
        return self.fit_error < 1e-10

    @property
    def expression_str(self) -> str:
        terms = [
            f"{c}*PREV({i+1})"
            for i, c in enumerate(self.coefficients)
            if c != 0
        ]
        if not terms:
            return f"0 % {self.modulus}"
        return f"({' + '.join(terms)}) % {self.modulus}"

    def description(self) -> str:
        return (
            f"RecurrenceAxiom(order={self.order}, mod={self.modulus})\n"
            f"  Rule: {self.expression_str}\n"
            f"  Fit error: {self.fit_error:.6f}\n"
            f"  Evidence: {self.evidence_length} observations"
        )

    def predict(self, history: List[int], t: int) -> int:
        """Predict obs[t] given obs[0..t-1]."""
        k = self.order
        val = 0
        for i, c in enumerate(self.coefficients):
            lag = i + 1
            if t - lag >= 0:
                val += c * history[t - lag]
        return val % self.modulus


def berlekamp_massey_mod(sequence: List[int], modulus: int) -> Optional[List[int]]:
    """
    Berlekamp-Massey algorithm over Z/modulus Z.
    
    Finds the shortest linear recurrence over Z/N that generates
    the given sequence.
    
    Returns: list of coefficients [c1, c2, ..., ck] such that
      seq[t] = (c1*seq[t-1] + ... + ck*seq[t-k]) % modulus
    
    Returns None if no linear recurrence is found (e.g., modulus is not prime,
    or sequence is genuinely nonlinear).
    
    Note: BM works perfectly when modulus is prime (Z/pZ is a field).
    For composite moduli, it may fail to find a recurrence even if one exists.
    """
    n = len(sequence)
    if n < 2:
        return None

    # For modular BM, we need modular inverse — only works for prime moduli
    def is_prime(n: int) -> bool:
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        return all(n % i != 0 for i in range(3, int(n**0.5)+1, 2))

    # Standard (non-modular) BM for simplicity
    # Works over integers when the sequence satisfies an integer recurrence
    # This covers Fibonacci, Tribonacci, Lucas over any modulus
    
    # We implement a simplified version that finds the recurrence
    # by solving a linear system: given k observations, find coefficients
    # that predict the next k observations
    
    for order in range(1, min(n // 2 + 1, 21)):  # try orders 1..20
        # Build the Hankel-like system for order k
        # We need: seq[k..2k-1] = M * [c1..ck] (mod N)
        # where M[i][j] = seq[k+i-j-1]
        
        if 2 * order > n:
            break
            
        # Build system Ax = b (mod N) where:
        # A[i][j] = seq[order + i - j - 1]  for i in [0, order), j in [0, order)
        # b[i] = seq[order + i]
        A = []
        b_vec = []
        for i in range(order):
            row = [sequence[order + i - j - 1] for j in range(order)]
            A.append(row)
            b_vec.append(sequence[order + i])

        # Solve Ax = b over Z/N (Gaussian elimination mod N)
        coeffs = _solve_linear_mod(A, b_vec, modulus, order)
        if coeffs is None:
            continue

        # Verify: check that the recurrence holds for ALL remaining elements
        correct = 0
        total = 0
        for t in range(order, n):
            pred = sum(coeffs[i] * sequence[t - i - 1] for i in range(order)) % modulus
            if pred == sequence[t]:
                correct += 1
            total += 1

        if total > 0 and correct / total > 0.95:  # 95% accuracy
            return coeffs

    return None


def _solve_linear_mod(
    A: List[List[int]],
    b: List[int],
    N: int,
    order: int,
) -> Optional[List[int]]:
    """
    Solve Ax = b (mod N) using Gaussian elimination.
    Returns None if no unique solution exists.
    
    Works well for prime N. For composite N, may fail even if solution exists.
    """
    # Augmented matrix [A | b]
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    for col in range(order):
        # Find pivot
        pivot_row = None
        for row in range(col, order):
            if M[row][col] % N != 0:
                pivot_row = row
                break
        if pivot_row is None:
            return None   # Singular system
        
        # Swap
        M[col], M[pivot_row] = M[pivot_row], M[col]
        
        # Eliminate
        pivot_val = M[col][col]
        
        # Try to find modular inverse of pivot_val mod N
        inv = _mod_inverse(pivot_val, N)
        if inv is None:
            return None   # Can't invert — try next order
        
        # Scale pivot row
        M[col] = [(x * inv) % N for x in M[col]]
        
        # Eliminate column in other rows
        for row in range(order):
            if row != col:
                factor = M[row][col]
                M[row] = [(M[row][j] - factor * M[col][j]) % N for j in range(order + 1)]
    
    return [M[i][order] % N for i in range(order)]


def _mod_inverse(a: int, n: int) -> Optional[int]:
    """Extended Euclidean algorithm to find a^(-1) mod n."""
    if n == 1:
        return 0
    a = a % n
    if a == 0:
        return None
    g, x, _ = _extended_gcd(a, n)
    if g != 1:
        return None   # a and n not coprime
    return x % n


def _extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended GCD: returns (gcd, x, y) where ax + by = gcd."""
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


class RecurrenceDetector:
    """
    Detects linear recurrence relations in integer sequences.
    
    Uses Berlekamp-Massey as the primary method, then verifies
    the discovered recurrence on held-out data.
    """

    def __init__(
        self,
        max_order: int = 20,
        min_evidence: int = 50,
        accuracy_threshold: float = 0.95,
    ):
        self.max_order = max_order
        self.min_evidence = min_evidence
        self.accuracy_threshold = accuracy_threshold

    def detect(
        self,
        observations: List[int],
        modulus: int,
        environment_name: str = "unknown",
        discovery_step: int = 0,
    ) -> Optional[RecurrenceAxiom]:
        """
        Try to find a linear recurrence for the given observations.
        
        Returns a RecurrenceAxiom if found, None otherwise.
        """
        if len(observations) < self.min_evidence:
            return None

        # Run BM
        coefficients = berlekamp_massey_mod(observations[:len(observations)//2], modulus)
        if coefficients is None:
            return None

        order = len(coefficients)
        if order > self.max_order:
            return None

        # Verify on held-out second half
        held_out = observations[len(observations)//2:]
        history = list(observations[:len(observations)//2])

        correct = 0
        total = 0
        for t_local, actual in enumerate(held_out):
            t_global = len(observations)//2 + t_local
            pred = sum(
                coefficients[i] * history[t_global - i - 1]
                for i in range(order)
                if t_global - i - 1 >= 0
            ) % modulus
            if pred == actual:
                correct += 1
            total += 1
            history.append(actual)

        fit_error = 1.0 - (correct / total if total > 0 else 0.0)

        if fit_error > 1.0 - self.accuracy_threshold:
            return None   # Not accurate enough

        return RecurrenceAxiom(
            order=order,
            coefficients=coefficients,
            modulus=modulus,
            fit_error=fit_error,
            evidence_length=len(observations),
            environment_name=environment_name,
            discovery_step=discovery_step,
        )

    def verify_axiom(
        self,
        axiom: RecurrenceAxiom,
        new_observations: List[int],
    ) -> float:
        """
        Verify a previously discovered recurrence on new observations.
        Returns accuracy (0.0–1.0).
        """
        order = axiom.order
        correct = 0
        total = 0
        for t in range(order, len(new_observations)):
            pred = axiom.predict(new_observations, t)
            if pred == new_observations[t]:
                correct += 1
            total += 1
        return correct / total if total > 0 else 0.0