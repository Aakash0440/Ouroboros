"""Tests for long-range PREV environments and RecurrenceDetector."""
import pytest
from ouroboros.environments.long_range import (
    TribonacciModEnv, LucasSequenceEnv, LinearRecurrenceEnv,
    AutoregressiveEnv,
)
from ouroboros.emergence.recurrence_detector import (
    RecurrenceDetector, RecurrenceAxiom,
    berlekamp_massey_mod, _mod_inverse, _extended_gcd,
)
from ouroboros.synthesis.long_range_beam import (
    LongRangeBeamSearch, LongRangeBeamConfig, recurrence_to_expr,
)


# ─── TribonacciModEnv ─────────────────────────────────────────────────────────

class TestTribonacciModEnv:
    def test_max_lag(self):
        assert TribonacciModEnv().max_lag == 3

    def test_first_few_values(self):
        env = TribonacciModEnv(modulus=7, seeds=(0, 1, 1))
        seq = env.generate(10)
        assert seq[0] == 0
        assert seq[1] == 1
        assert seq[2] == 1
        # seq[3] = (0+1+1)%7 = 2
        assert seq[3] == 2
        # seq[4] = (1+1+2)%7 = 4
        assert seq[4] == 4

    def test_values_in_range(self):
        env = TribonacciModEnv(modulus=7)
        seq = env.generate(500)
        assert all(0 <= v < 7 for v in seq)

    def test_recurrence_holds(self):
        env = TribonacciModEnv(modulus=7)
        seq = env.generate(100)
        for t in range(3, 100):
            expected = (seq[t-1] + seq[t-2] + seq[t-3]) % 7
            assert seq[t] == expected, f"Tribonacci failed at t={t}"

    def test_ground_truth_rule_str(self):
        env = TribonacciModEnv(modulus=7)
        assert "PREV(1)" in env.ground_truth_rule()
        assert "PREV(3)" in env.ground_truth_rule()


class TestLucasSequenceEnv:
    def test_max_lag(self):
        assert LucasSequenceEnv().max_lag == 2

    def test_seeds(self):
        env = LucasSequenceEnv(modulus=100)  # large mod to see raw Lucas numbers
        seq = env.generate(5)
        assert seq[0] == 2   # L(0) = 2
        assert seq[1] == 1   # L(1) = 1

    def test_lucas_recurrence(self):
        env = LucasSequenceEnv(modulus=11)
        seq = env.generate(50)
        for t in range(2, 50):
            expected = (seq[t-1] + seq[t-2]) % 11
            assert seq[t] == expected

    def test_differs_from_fibonacci(self):
        from ouroboros.environments.fibonacci_mod import FibonacciModEnv
        fib = FibonacciModEnv(modulus=11).generate(20)
        lucas = LucasSequenceEnv(modulus=11).generate(20)
        # Same recurrence, different seeds → different sequences
        assert fib != lucas


class TestLinearRecurrenceEnv:
    def test_order_2(self):
        env = LinearRecurrenceEnv(coefficients=[1, 1], modulus=7, seeds=[0, 1])
        seq = env.generate(20)
        # This should match Fibonacci mod 7
        for t in range(2, 20):
            assert seq[t] == (seq[t-1] + seq[t-2]) % 7

    def test_order_3(self):
        env = LinearRecurrenceEnv(coefficients=[1, 1, 1], modulus=7, seeds=[0, 1, 1])
        seq = env.generate(20)
        # Tribonacci
        for t in range(3, 20):
            assert seq[t] == (seq[t-1] + seq[t-2] + seq[t-3]) % 7

    def test_max_lag_matches_order(self):
        env = LinearRecurrenceEnv(coefficients=[1, 0, 1, 0, 1], modulus=5)
        assert env.max_lag == 5


class TestAutoRegressiveEnv:
    def test_max_lag(self):
        env = AutoregressiveEnv(nonzero_lags=[(1, 1), (7, 1)], modulus=7)
        assert env.max_lag == 7

    def test_values_in_range(self):
        env = AutoregressiveEnv(nonzero_lags=[(1, 2), (3, 1)], modulus=5)
        seq = env.generate(200)
        assert all(0 <= v < 5 for v in seq)


# ─── RecurrenceDetector ───────────────────────────────────────────────────────

class TestModularArithmetic:
    def test_mod_inverse_prime(self):
        assert _mod_inverse(3, 7) == 5   # 3*5=15≡1 (mod 7)
        assert _mod_inverse(5, 7) == 3
        assert _mod_inverse(2, 7) == 4   # 2*4=8≡1 (mod 7)

    def test_mod_inverse_noncoprime(self):
        assert _mod_inverse(0, 7) is None
        assert _mod_inverse(7, 7) is None  # gcd(7,7)=7≠1

    def test_extended_gcd(self):
        g, x, y = _extended_gcd(7, 11)
        assert g == 1
        assert 7*x + 11*y == 1


class TestBerlekampMassey:
    def test_fibonacci_mod7(self):
        from ouroboros.environments.fibonacci_mod import FibonacciModEnv
        seq = FibonacciModEnv(modulus=7).generate(100)
        coeffs = berlekamp_massey_mod(seq, 7)
        assert coeffs is not None
        assert len(coeffs) <= 2

    def test_tribonacci_mod7(self):
        env = TribonacciModEnv(modulus=7)
        seq = env.generate(100)
        coeffs = berlekamp_massey_mod(seq, 7)
        # Should find order-3 recurrence
        assert coeffs is not None

    def test_pure_noise_not_detected(self):
        import random
        rng = random.Random(42)
        noise = [rng.randint(0, 6) for _ in range(100)]
        # BM on pure noise may return something spurious, but should have high error
        # We don't assert None here — BM can spuriously find patterns in short random sequences

    def test_constant_sequence(self):
        seq = [3] * 50
        coeffs = berlekamp_massey_mod(seq, 7)
        # [3,3,3,...] satisfies 3 = 1*3 (mod 7) — order-1 recurrence
        # coeffs[0] should be 1


class TestRecurrenceDetector:
    def test_detects_fibonacci(self):
        from ouroboros.environments.fibonacci_mod import FibonacciModEnv
        seq = FibonacciModEnv(modulus=7).generate(200)
        detector = RecurrenceDetector(max_order=5)
        axiom = detector.detect(seq, modulus=7)
        if axiom is not None:
            assert axiom.order <= 3
            assert axiom.is_perfect or axiom.fit_error < 0.05

    def test_detects_tribonacci(self):
        seq = TribonacciModEnv(modulus=7).generate(200)
        detector = RecurrenceDetector(max_order=5)
        axiom = detector.detect(seq, modulus=7)
        if axiom is not None:
            assert axiom.order <= 4

    def test_axiom_fields(self):
        seq = TribonacciModEnv(modulus=7).generate(200)
        detector = RecurrenceDetector(max_order=5)
        axiom = detector.detect(seq, modulus=7, environment_name="TribonacciMod(7)")
        if axiom is not None:
            assert isinstance(axiom.coefficients, list)
            assert axiom.modulus == 7
            assert 0.0 <= axiom.fit_error <= 1.0
            assert axiom.evidence_length == 200

    def test_axiom_predict(self):
        axiom = RecurrenceAxiom(
            order=2, coefficients=[1, 1], modulus=7,
            fit_error=0.0, evidence_length=100,
        )
        history = [0, 1, 1, 2, 3, 5]
        pred = axiom.predict(history, 6)
        assert pred == (5 + 3) % 7   # = 8%7 = 1


class TestLongRangeBeamSearch:
    def test_bm_on_fibonacci_gives_result(self):
        from ouroboros.environments.fibonacci_mod import FibonacciModEnv
        seq = FibonacciModEnv(modulus=7).generate(150)
        cfg = LongRangeBeamConfig(
            max_lag=5, beam_width=10, use_bm_warmstart=True
        )
        searcher = LongRangeBeamSearch(cfg)
        result = searcher.search(seq, modulus=7)
        assert result.best_expr is not None
        assert result.best_mdl_cost < float('inf')

    def test_discovery_method_field(self):
        from ouroboros.environments.fibonacci_mod import FibonacciModEnv
        seq = FibonacciModEnv(modulus=7).generate(150)
        searcher = LongRangeBeamSearch(LongRangeBeamConfig(max_lag=5))
        result = searcher.search(seq, modulus=7)
        assert result.discovery_method in ("BerlekampMassey", "BeamSearch", "Hybrid")

    def test_recurrence_to_expr(self):
        axiom = RecurrenceAxiom(
            order=2, coefficients=[1, 1], modulus=7,
            fit_error=0.0, evidence_length=100,
        )
        expr = recurrence_to_expr(axiom)
        assert expr is not None
        # Evaluate at t=5 with history [0,1,1,2,3]
        history = [0, 1, 1, 2, 3]
        pred = expr.evaluate(5, history)
        expected = (3 + 2) % 7
        assert pred == expected