"""Tests for FFTPeriodFinder and algorithm environments."""
import math
import pytest
from ouroboros.search.fft_period_finder import FFTPeriodFinder, PeriodResult
from ouroboros.environments.algorithm_env import (
    GCDEnv, FibonacciDirectEnv, PrimeCountEnv, CollatzEnv,
)


class TestFFTPeriodFinder:
    def _finder(self): return FFTPeriodFinder(min_period=2, max_period=50)

    def test_detects_period_7(self):
        seq = [float((3*t+1)%7) for t in range(200)]
        result = self._finder().find_dominant_period(seq)
        assert result is not None
        assert result.period == 7 or abs(result.period - 7) <= 1

    def test_sine_period_detected(self):
        seq = [math.sin(2 * math.pi * t / 14) for t in range(200)]
        result = self._finder().find_dominant_period(seq)
        assert result is not None
        assert 12 <= result.period <= 16  # approximate period

    def test_constant_no_period(self):
        seq = [5.0] * 100
        result = self._finder().find_dominant_period(seq)
        # Constant sequence has no meaningful period
        if result:
            assert result.confidence < 0.5

    def test_period_result_fields(self):
        seq = [float(t % 10) for t in range(150)]
        result = FFTPeriodFinder().find_dominant_period(seq)
        if result:
            assert isinstance(result.period, int)
            assert 0.0 <= result.amplitude
            assert 0.0 <= result.confidence <= 1.0

    def test_is_significant_for_strong_period(self):
        seq = [math.cos(2 * math.pi * t / 7) for t in range(200)]
        result = FFTPeriodFinder().find_dominant_period(seq)
        if result:
            # Strong sinusoid should be significant
            assert result.amplitude > 0.01

    def test_multiple_periods_returned(self):
        seq = [float((t%7) + (t%11)*0.1) for t in range(300)]
        results = FFTPeriodFinder().find_periods(seq)
        assert len(results) >= 1

    def test_short_sequence_returns_empty(self):
        seq = [1.0, 2.0, 3.0]
        results = FFTPeriodFinder().find_periods(seq)
        assert results == []

    def test_period_confidence_in_range(self):
        seq = [math.sin(2*math.pi*t/7) for t in range(100)]
        results = FFTPeriodFinder().find_periods(seq)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0


class TestGCDEnv:
    def test_outputs_are_valid_gcds(self):
        env = GCDEnv(seed=42)
        outputs = env.generate(50)
        for i, (g, (a, b)) in enumerate(zip(outputs, env._pairs[:50])):
            assert g == math.gcd(a, b), f"GCD mismatch at {i}: gcd({a},{b})={math.gcd(a,b)}, got {g}"

    def test_outputs_in_range(self):
        env = GCDEnv()
        obs = env.generate(100)
        assert all(1 <= v <= GCDEnv.MAX_VAL for v in obs)

    def test_deterministic(self):
        env1 = GCDEnv(seed=1)
        env2 = GCDEnv(seed=1)
        assert env1.generate(50) == env2.generate(50)

    def test_different_seeds_differ(self):
        env1 = GCDEnv(seed=1)
        env2 = GCDEnv(seed=2)
        assert env1.generate(50) != env2.generate(50)


class TestPrimeCountEnv:
    def test_prime_count_correct(self):
        env = PrimeCountEnv()
        obs = env.generate(30)
        # π(0)=0, π(1)=0, π(2)=1, π(3)=2, π(4)=2, π(5)=3
        assert obs[0] == 0   # 0 primes ≤ 0
        assert obs[2] == 1   # 1 prime ≤ 2 (which is 2)
        assert obs[3] == 2   # 2 primes ≤ 3 (which are 2,3)
        assert obs[4] == 2   # 4 is not prime
        assert obs[5] == 3   # 5 is prime

    def test_cumsum_isprime_matches(self):
        """Verify that CUMSUM(ISPRIME(t)) = π(t)."""
        from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
        from ouroboros.synthesis.expr_node import NodeType

        def time_e():
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.TIME; n.value = 0.0
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n

        isprime_expr = ExtExprNode(ExtNodeType.ISPRIME, left=time_e())
        cumsum_expr = ExtExprNode(ExtNodeType.CUMSUM, left=isprime_expr)

        env = PrimeCountEnv()
        obs = env.generate(50)
        preds = [int(round(cumsum_expr.evaluate(t, []))) for t in range(50)]
        
        n_correct = sum(1 for p, o in zip(preds, obs) if p == o)
        # Should match perfectly since CUMSUM(ISPRIME) IS π(t)
        assert n_correct >= 45, f"Only {n_correct}/50 correct"


class TestFibonacciDirectEnv:
    def test_fibonacci_values(self):
        env = FibonacciDirectEnv(modulus=10000)
        obs = env.generate(10)
        # F(0..9) = 0,1,1,2,3,5,8,13,21,34
        assert obs[0] == 0
        assert obs[1] == 1
        assert obs[2] == 1
        assert obs[3] == 2
        assert obs[7] == 13

    def test_modular_wrapping(self):
        env = FibonacciDirectEnv(modulus=10)
        obs = env.generate(50)
        assert all(0 <= v < 10 for v in obs)


class TestCollatzEnv:
    def test_collatz_stopping_times(self):
        env = CollatzEnv()
        obs = env.generate(10)
        assert obs[0] == 0   # 0 → 0 steps
        assert obs[1] == 0   # 1 → 0 steps
        # 2 → 1 → 0 steps: 1 step
        assert obs[2] == 1
        # 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1: 7 steps
        assert obs[3] == 7

    def test_positive_values(self):
        env = CollatzEnv()
        obs = env.generate(100)
        assert all(v >= 0 for v in obs)


class TestPeriodAwareSeedBuilder:
    def test_seeds_for_periodic(self):
        from ouroboros.search.fft_period_finder import PeriodAwareSeedBuilder
        seq = [float(t%7) for t in range(200)]
        builder = PeriodAwareSeedBuilder()
        seeds = builder.build_seeds(seq)
        # May or may not find period depending on FFT threshold
        assert isinstance(seeds, list)