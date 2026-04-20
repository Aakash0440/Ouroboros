"""Tests for continuous observation environments."""
import math
import pytest
from ouroboros.continuous.environments import (
    SineEnv, PolynomialEnv, ExponentialEnv,
    DampedOscillatorEnv, LogisticMapEnv, ContinuousNoiseEnv,
    make_continuous_environment_suite,
)


class TestSineEnv:
    def test_noiseless_exact(self):
        env = SineEnv(amplitude=1.0, frequency=1/7, phase=0.0, offset=0.0)
        for t in range(50):
            expected = math.sin(t / 7.0)
            assert abs(env.observe(t) - expected) < 1e-10

    def test_generate_length(self):
        env = SineEnv()
        seq = env.generate(100)
        assert len(seq) == 100

    def test_noisy_differs(self):
        env_clean = SineEnv(noise_sigma=0.0, seed=1)
        env_noisy = SineEnv(noise_sigma=0.5, seed=1)
        clean = env_clean.generate(20)
        noisy = env_noisy.generate(20)
        # At least some values should differ
        diffs = [abs(c - n) for c, n in zip(clean, noisy)]
        assert max(diffs) > 0.01

    def test_ground_truth_str(self):
        env = SineEnv(amplitude=2.0, frequency=0.5)
        s = env.ground_truth_expr()
        assert "sin" in s and "t" in s


class TestPolynomialEnv:
    def test_degree_2_values(self):
        env = PolynomialEnv(coefficients=[1.0, -2.0, 0.5])
        # f(t) = 1 - 2t + 0.5t²
        assert abs(env.observe(0) - 1.0) < 1e-10
        assert abs(env.observe(1) - (-0.5)) < 1e-10
        # 1.0 + (-2.0)*2 + 0.5*4 = 1 - 4 + 2 = -1
        assert abs(env.observe(2) - (-1.0)) < 1e-10

    def test_constant_polynomial(self):
        env = PolynomialEnv(coefficients=[5.0])
        for t in range(10):
            assert abs(env.observe(t) - 5.0) < 1e-10

    def test_linear_polynomial(self):
        env = PolynomialEnv(coefficients=[0.0, 3.0])
        for t in range(10):
            assert abs(env.observe(t) - 3.0 * t) < 1e-10


class TestExponentialEnv:
    def test_at_t0(self):
        env = ExponentialEnv(scale=1.0, rate=0.1, offset=0.0)
        assert abs(env.observe(0) - 1.0) < 1e-10

    def test_growth(self):
        env = ExponentialEnv(scale=1.0, rate=0.1)
        seq = env.generate(10)
        # Exponential growth: each value should be larger than previous
        for i in range(1, len(seq)):
            assert seq[i] > seq[i-1]

    def test_with_offset(self):
        env = ExponentialEnv(scale=1.0, rate=0.0, offset=5.0)
        # exp(0*t) = 1, so value = 1 + 5 = 6
        for t in range(10):
            assert abs(env.observe(t) - 6.0) < 1e-10


class TestDampedOscillatorEnv:
    def test_decaying(self):
        env = DampedOscillatorEnv(amplitude=2.0, decay=0.1, omega=0.8)
        seq = env.generate(100)
        # The envelope should decay: max(abs) should decrease
        early_max = max(abs(v) for v in seq[:10])
        late_max = max(abs(v) for v in seq[90:])
        assert late_max < early_max

    def test_oscillating(self):
        env = DampedOscillatorEnv(amplitude=2.0, decay=0.01, omega=1.0)
        seq = env.generate(50)
        # Should have sign changes (oscillation)
        sign_changes = sum(
            1 for i in range(1, len(seq)) if seq[i] * seq[i-1] < 0
        )
        assert sign_changes >= 5

    def test_zero_at_t0(self):
        env = DampedOscillatorEnv(amplitude=1.0, decay=0.1, omega=1.0, phase=0.0)
        # sin(0) = 0, so value at t=0 is 0
        assert abs(env.observe(0)) < 1e-10


class TestLogisticMapEnv:
    def test_fixed_point_convergence(self):
        env = LogisticMapEnv(r=2.8, x0=0.5)
        seq = env.generate(200)
        # Should converge to 1 - 1/r = 1 - 1/2.8 ≈ 0.6429
        fixed_point = 1.0 - 1.0/2.8
        assert abs(seq[-1] - fixed_point) < 0.01

    def test_chaotic_range(self):
        env = LogisticMapEnv(r=3.9, x0=0.5)
        seq = env.generate(100)
        # Values should be in (0, 1)
        for v in seq:
            assert 0.0 < v < 1.0

    def test_deterministic(self):
        env1 = LogisticMapEnv(r=3.5, x0=0.5, seed=1)
        env2 = LogisticMapEnv(r=3.5, x0=0.5, seed=1)
        assert env1.generate(50) == env2.generate(50)


class TestContinuousNoiseEnv:
    def test_different_each_time(self):
        env = ContinuousNoiseEnv(sigma=1.0, seed=99)
        seq = env.generate(50)
        # Should not be all zeros (noise is added)
        assert any(abs(v) > 0.01 for v in seq)

    def test_no_structure_ground_truth(self):
        env = ContinuousNoiseEnv()
        assert "no structure" in env.ground_truth_expr()


class TestSuite:
    def test_make_suite_length(self):
        suite = make_continuous_environment_suite()
        assert len(suite) == 8

    def test_all_generate(self):
        suite = make_continuous_environment_suite()
        for env in suite:
            seq = env.generate(50)
            assert len(seq) == 50
            assert all(isinstance(v, float) for v in seq)