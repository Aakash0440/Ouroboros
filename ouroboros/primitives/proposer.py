"""
PrimitiveProposer — Proposes new mathematical primitives when vocabulary is insufficient.

When CompletenessChecker proves no expression of depth ≤ D achieves MDL ≤ C,
PrimitiveProposer analyzes the RESIDUALS of the best existing expression to
find structure that the current vocabulary cannot capture.

Residual analysis:
  best_expr produces predictions p[0], p[1], ..., p[n]
  residuals = obs[t] - p[t]  for each t

  The residuals are the part of the signal the current vocabulary cannot explain.
  If the residuals have structure, a new primitive is needed to capture it.

Structure detection in residuals:
  1. MULTIPLICATIVE STRUCTURE: residual(mn) = residual(m) * residual(n)?
     → Propose MULTIPLICATIVE_ARITHMETIC_FN node
  2. ADDITIVE RECURRENCE: residual(n) = a*residual(n-1) + b*residual(n-2)?
     → Propose HIGHER_ORDER_RECURRENCE node with detected coefficients
  3. LOOKUP TABLE PATTERN: residuals are periodic with a complex pattern?
     → Propose LOOKUP_TABLE_FN node
  4. GROWTH RATE CHANGE: residuals have exponential growth with changing exponent?
     → Propose VARIABLE_EXPONENT node

For each detected structure, PrimitiveProposer:
  1. Names the proposed node
  2. Specifies its input-output behavior on the first 200 values
  3. Generates a Python implementation
  4. Generates a Lean4 definition skeleton
  5. Adds it to the vocabulary (pending verification)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable, Tuple


@dataclass
class ProposedPrimitive:
    """
    A proposed new mathematical primitive.

    Contains everything needed to add it to the OUROBOROS vocabulary:
    - Name and description
    - Input-output behavior on test cases (for verification)
    - Python implementation
    - Grammar rule (what can be its parent/children)
    - Lean4 definition skeleton
    """
    name: str
    description: str
    structure_type: str     # "multiplicative", "recurrence", "lookup", "growth"
    arity: int              # 0=terminal, 1=unary, 2=binary
    category: str           # "NUMBER", "CALCULUS", "MEMORY", etc.

    # Behavior specification
    test_inputs: List[int]          # inputs used to define behavior
    test_outputs: List[float]       # expected outputs for test_inputs
    implementation_code: str        # Python code implementing this primitive
    lean4_definition: str           # Lean4 definition skeleton

    # Quality metrics
    residual_reduction: float       # how much does adding this reduce residuals? (0-1)
    confidence: float               # 0-1, how confident are we in this structure?

    # Grammar integration
    grammar_rule: str               # human-readable grammar rule
    description_bits: float         # estimated MDL description cost

    def is_worth_adding(self) -> bool:
        """True if this primitive is worth adding to the vocabulary."""
        return (self.residual_reduction > 0.3 and
                self.confidence > 0.6 and
                len(self.test_inputs) >= 10)

    def summary(self) -> str:
        status = "✅ WORTH ADDING" if self.is_worth_adding() else "⚠️  Marginal"
        return (
            f"{status}: {self.name} ({self.structure_type})\n"
            f"  Description: {self.description}\n"
            f"  Arity: {self.arity}, Category: {self.category}\n"
            f"  Residual reduction: {self.residual_reduction:.2%}\n"
            f"  Confidence: {self.confidence:.2%}\n"
            f"  Grammar: {self.grammar_rule}"
        )


class PrimitiveProposer:
    """
    Analyzes residuals to propose new mathematical primitives.

    Usage:
        proposer = PrimitiveProposer()

        # After completeness check fails:
        best_expr = completeness_result.best_expr
        residuals = compute_residuals(best_expr, observations)
        proposals = proposer.propose(residuals, observations)

        for proposal in proposals:
            if proposal.is_worth_adding():
                print(proposal.summary())
    """

    def __init__(self, min_residual_reduction: float = 0.25):
        self.min_reduction = min_residual_reduction

    def compute_residuals(
        self,
        best_expr,
        observations: List[int],
    ) -> List[float]:
        """Compute residuals: obs[t] - best_expr_prediction[t]."""
        residuals = []
        for t, obs_val in enumerate(observations):
            try:
                pred = best_expr.evaluate(t, observations[:t], {})
                if not math.isfinite(pred):
                    pred = 0.0
                residuals.append(float(obs_val) - pred)
            except Exception:
                residuals.append(0.0)
        return residuals

    def propose(
        self,
        residuals: List[float],
        observations: List[int],
        max_proposals: int = 3,
    ) -> List[ProposedPrimitive]:
        """
        Analyze residuals and propose new primitives.
        Returns up to max_proposals candidates, sorted by residual_reduction.
        """
        proposals = []

        # Strategy 1: Check for multiplicative structure
        mult_proposal = self._check_multiplicative(residuals, observations)
        if mult_proposal:
            proposals.append(mult_proposal)

        # Strategy 2: Check for higher-order recurrence
        rec_proposal = self._check_higher_order_recurrence(residuals)
        if rec_proposal:
            proposals.append(rec_proposal)

        # Strategy 3: Check for lookup table / periodic complex pattern
        lookup_proposal = self._check_lookup_pattern(residuals, observations)
        if lookup_proposal:
            proposals.append(lookup_proposal)

        # Strategy 4: Check for variable exponent growth
        growth_proposal = self._check_variable_growth(residuals)
        if growth_proposal:
            proposals.append(growth_proposal)

        # Sort by residual reduction
        proposals.sort(key=lambda p: -p.residual_reduction)
        return proposals[:max_proposals]

    def _check_multiplicative(
        self,
        residuals: List[float],
        observations: List[int],
    ) -> Optional[ProposedPrimitive]:
        """
        Check if residuals have multiplicative structure: r(mn) = r(m) * r(n).
        This is the signature of multiplicative arithmetic functions like
        Euler's totient, Ramanujan's tau, Möbius function, etc.
        """
        n = len(residuals)
        if n < 20:
            return None

        # Test multiplicativity: for coprime (a,b), r(a*b) ≈ r(a) * r(b)?
        import math as _math
        n_tests = 0
        n_pass = 0
        for a in range(2, min(10, n)):
            for b in range(2, min(10, n)):
                ab = a * b
                if ab >= n:
                    continue
                if _math.gcd(a, b) != 1:
                    continue
                n_tests += 1
                if abs(residuals[a]) < 0.01 or abs(residuals[b]) < 0.01:
                    continue
                ratio = residuals[ab] / (residuals[a] * residuals[b] + 1e-10)
                if abs(ratio - 1.0) < 0.15:  # within 15% of multiplicative
                    n_pass += 1

        if n_tests < 5:
            return None
        multiplicative_score = n_pass / n_tests

        if multiplicative_score < 0.5:
            return None

        # Estimate residual reduction
        # If we could perfectly predict the multiplicative part,
        # residuals would drop by roughly multiplicative_score fraction
        residual_reduction = multiplicative_score * 0.6

        test_inputs = list(range(1, min(20, n)))
        test_outputs = [residuals[i] for i in test_inputs]

        impl_code = '''
def multiplicative_arithmetic_fn(n: int, precomputed: dict) -> float:
    """
    A multiplicative arithmetic function defined by its values at primes.
    f(1) = 1; f(p^k) = base case; f(mn) = f(m)*f(n) when gcd(m,n)=1.
    """
    if n in precomputed:
        return precomputed[n]
    # Compute from prime factorization
    result = 1.0
    temp = n
    for p in range(2, int(n**0.5) + 1):
        if temp % p == 0:
            pk = 1
            while temp % p == 0:
                pk *= p
                temp //= p
            result *= precomputed.get(pk, float(pk))
    if temp > 1:
        result *= precomputed.get(temp, float(temp))
    precomputed[n] = result
    return result
'''

        lean4_def = '''
-- Proposed: MULTIPLICATIVE_ARITHMETIC_FN
-- A multiplicative function f : ℕ → ℝ satisfying f(mn) = f(m)f(n) when gcd(m,n)=1
noncomputable def multiplicativeArithFn (f_primes : ℕ → ℝ) (n : ℕ) : ℝ :=
  -- Definition via prime factorization
  -- To be filled in with concrete values once the function is identified
  sorry
'''

        return ProposedPrimitive(
            name="MULTIPLICATIVE_ARITH",
            description="Multiplicative arithmetic function: f(mn) = f(m)f(n) for coprime m,n",
            structure_type="multiplicative",
            arity=1,
            category="NUMBER",
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            implementation_code=impl_code,
            lean4_definition=lean4_def,
            residual_reduction=residual_reduction,
            confidence=multiplicative_score,
            grammar_rule="MULTIPLICATIVE_ARITH takes NUMBER or TERMINAL child",
            description_bits=7.0,
        )

    def _check_higher_order_recurrence(
        self,
        residuals: List[float],
    ) -> Optional[ProposedPrimitive]:
        """
        Check if residuals satisfy a higher-order linear recurrence
        that the current PREV(1..5) nodes cannot capture (order > 5).
        """
        n = len(residuals)
        if n < 15:
            return None

        # Try to fit recurrences of order 6-10
        for order in range(6, min(11, n // 3)):
            # Build system: r[t] = sum_k a_k * r[t-k] for t >= order
            m = n - order
            if m < order + 2:
                continue

            # OLS for recurrence coefficients
            A = []
            b_vec = []
            for t in range(order, n):
                row = [residuals[t - k - 1] for k in range(order)]
                A.append(row)
                b_vec.append(residuals[t])

            # Simple least-squares (no numpy)
            coeffs = self._simple_ols(A, b_vec)
            if coeffs is None:
                continue

            # Measure fit
            predictions = list(residuals[:order])
            for t in range(order, n):
                pred = sum(coeffs[k] * predictions[t - k - 1]
                           for k in range(min(order, len(predictions))))
                predictions.append(pred)

            errors = [abs(predictions[t] - residuals[t]) for t in range(order, n)]
            mean_error = sum(errors) / max(len(errors), 1)
            baseline = sum(abs(r) for r in residuals) / max(n, 1)

            if baseline < 1e-10:
                continue
            relative_error = mean_error / baseline

            if relative_error < 0.2:  # good fit
                residual_reduction = 1.0 - relative_error

                test_inputs = list(range(order, min(order + 20, n)))
                test_outputs = [residuals[i] for i in test_inputs]

                coeffs_str = ", ".join(f"{c:.3f}" for c in coeffs[:6])
                impl_code = f'''
def higher_order_recurrence(history: list, t: int) -> float:
    """Order-{order} linear recurrence: f(t) = {coeffs_str} * f(t-1..{order})"""
    if len(history) < {order}:
        return 0.0
    return sum({coeffs[k]:.6f} * history[-{k+1}] for k in range({order}))
'''

                return ProposedPrimitive(
                    name=f"RECUR_{order}",
                    description=f"Order-{order} linear recurrence (beyond PREV(1..5))",
                    structure_type="recurrence",
                    arity=1,
                    category="MEMORY",
                    test_inputs=test_inputs,
                    test_outputs=test_outputs,
                    implementation_code=impl_code,
                    lean4_definition=f"-- Order-{order} recurrence definition",
                    residual_reduction=residual_reduction,
                    confidence=1.0 - relative_error,
                    grammar_rule=f"RECUR_{order} takes any child (the sequence to recurse on)",
                    description_bits=4.0 + order * 0.5,
                )

        return None

    def _check_lookup_pattern(
        self,
        residuals: List[float],
        observations: List[int],
    ) -> Optional[ProposedPrimitive]:
        """
        Check if residuals form a complex periodic lookup table.
        This occurs when the sequence has a long period that cannot
        be expressed as a simple linear modular function.
        """
        n = len(residuals)
        if n < 30:
            return None

        # Try periods from 6 to 30
        best_period = None
        best_score = 0.0

        for period in range(6, min(31, n // 3)):
            # Check if residuals repeat with this period
            errors = []
            for t in range(period, n):
                errors.append(abs(residuals[t] - residuals[t % period]))
            mean_error = sum(errors) / max(len(errors), 1)
            baseline = sum(abs(r) for r in residuals[period:]) / max(len(residuals[period:]), 1)
            if baseline < 1e-10:
                continue
            score = 1.0 - mean_error / baseline
            if score > best_score:
                best_score = score
                best_period = period

        if best_period is None or best_score < 0.5:
            return None

        # The lookup table is the first `best_period` residual values
        lookup_values = residuals[:best_period]
        residual_reduction = best_score * 0.7

        test_inputs = list(range(best_period, min(best_period + 20, n)))
        test_outputs = [residuals[i % best_period] for i in test_inputs]

        impl_code = f'''
LOOKUP_TABLE = {lookup_values[:best_period]}
def lookup_fn(t: int) -> float:
    """Periodic lookup table with period {best_period}."""
    return LOOKUP_TABLE[t % {best_period}]
'''

        return ProposedPrimitive(
            name=f"LOOKUP_P{best_period}",
            description=f"Period-{best_period} lookup table",
            structure_type="lookup",
            arity=0,  # terminal — no children
            category="MEMORY",
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            implementation_code=impl_code,
            lean4_definition=f"-- Lookup table with period {best_period}",
            residual_reduction=residual_reduction,
            confidence=best_score,
            grammar_rule=f"LOOKUP_P{best_period} is a terminal node",
            description_bits=3.0 + best_period * 0.3,
        )

    def _check_variable_growth(
        self,
        residuals: List[float],
    ) -> Optional[ProposedPrimitive]:
        """
        Check if residuals have variable-exponent growth: r(t) ≈ C * t^f(t).
        This occurs in sequences like partition numbers, where the growth
        rate itself changes with t.
        """
        n = len(residuals)
        if n < 20:
            return None

        # Take logs and look for changing slope (variable exponent)
        log_abs = []
        log_t = []
        for t in range(2, n):
            if abs(residuals[t]) > 0.01:
                try:
                    log_abs.append(math.log(abs(residuals[t])))
                    log_t.append(math.log(t))
                except Exception:
                    pass

        if len(log_abs) < 10:
            return None

        # Check if the exponent (slope in log-log space) is itself changing
        # Compute local slopes in windows
        window = 5
        slopes = []
        for i in range(0, len(log_t) - window, window // 2):
            x = log_t[i:i+window]
            y = log_abs[i:i+window]
            if len(x) < 2:
                continue
            mx, my = sum(x)/len(x), sum(y)/len(y)
            slope = sum((xi-mx)*(yi-my) for xi,yi in zip(x,y)) / max(
                sum((xi-mx)**2 for xi in x), 1e-10
            )
            slopes.append(slope)

        if len(slopes) < 3:
            return None

        slope_variance = sum((s - sum(slopes)/len(slopes))**2 for s in slopes) / len(slopes)

        # High variance in slope → variable exponent
        if slope_variance < 0.1:
            return None  # constant slope — regular power law, no new primitive needed

        residual_reduction = min(0.6, slope_variance)
        confidence = min(0.8, slope_variance * 2)

        test_inputs = list(range(2, min(22, n)))
        test_outputs = [residuals[i] for i in test_inputs]

        impl_code = '''
def variable_exponent_fn(t: int, history: list) -> float:
    """Variable-exponent growth: C * t^(f(t)) where f(t) changes over time."""
    if not history or t < 2:
        return float(t)
    # Estimate current exponent from recent growth
    recent = history[-min(5, len(history)):]
    if len(recent) < 2 or recent[-2] <= 0:
        return float(t)
    log_ratio = math.log(max(recent[-1], 1e-10) / max(recent[-2], 1e-10))
    current_exponent = log_ratio / math.log(t / max(t-1, 1))
    return t ** current_exponent
'''

        return ProposedPrimitive(
            name="VAR_EXP",
            description="Variable-exponent growth: t^(f(t)) where exponent changes",
            structure_type="growth",
            arity=1,
            category="CALCULUS",
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            implementation_code=impl_code,
            lean4_definition="-- Variable exponent growth definition",
            residual_reduction=residual_reduction,
            confidence=confidence,
            grammar_rule="VAR_EXP takes CALCULUS or TERMINAL child",
            description_bits=6.0,
        )

    def _simple_ols(
        self,
        A: List[List[float]],
        b: List[float],
    ) -> Optional[List[float]]:
        """Simple OLS for small systems (no numpy)."""
        n = len(b)
        k = len(A[0]) if A else 0
        if n < k + 1 or k == 0:
            return None

        # Normal equations: A'A x = A'b
        # Build A'A and A'b
        AtA = [[sum(A[i][r]*A[i][c] for i in range(n)) for c in range(k)]
               for r in range(k)]
        Atb = [sum(A[i][j]*b[i] for i in range(n)) for j in range(k)]

        # Gaussian elimination (simplified for k ≤ 10)
        try:
            # Forward elimination
            for col in range(k):
                if abs(AtA[col][col]) < 1e-12:
                    return None
                pivot = AtA[col][col]
                for row in range(col+1, k):
                    factor = AtA[row][col] / pivot
                    for c in range(k):
                        AtA[row][c] -= factor * AtA[col][c]
                    Atb[row] -= factor * Atb[col]

            # Back substitution
            x = [0.0] * k
            for i in range(k-1, -1, -1):
                x[i] = Atb[i]
                for j in range(i+1, k):
                    x[i] -= AtA[i][j] * x[j]
                x[i] /= max(AtA[i][i], 1e-12)
            return x
        except Exception:
            return None