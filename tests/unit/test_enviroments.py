# tests/unit/test_environments.py

"""Unit tests for all observation environments."""

import pytest
import numpy as np
from ouroboros.environment import (
    BinaryRepeatEnv, ModularArithmeticEnv, FibonacciModEnv,
    PrimeSequenceEnv, NoiseEnv
)


class TestBinaryRepeatEnv:

    def test_stream_alternates(self):
        env = BinaryRepeatEnv()
        env.reset(100)
        stream = env.peek_all()
        for i, s in enumerate(stream):
            assert s == i % 2, f"Position {i}: expected {i%2}, got {s}"

    def test_alphabet_size_is_two(self):
        env = BinaryRepeatEnv()
        assert env.alphabet_size == 2

    def test_reset_clears_position(self):
        env = BinaryRepeatEnv()
        env.reset(50)
        env.observe(25)
        assert env.position == 25
        env.reset(50)
        assert env.position == 0

    def test_observe_advances_position(self):
        env = BinaryRepeatEnv()
        env.reset(100)
        obs = env.observe(10)
        assert len(obs) == 10
        assert env.position == 10

    def test_exhausted_flag(self):
        env = BinaryRepeatEnv()
        env.reset(5)
        env.observe(5)
        assert env.exhausted

    def test_remaining(self):
        env = BinaryRepeatEnv()
        env.reset(100)
        env.observe(30)
        assert env.remaining == 70


class TestModularArithmeticEnv:

    def test_stream_follows_rule(self):
        env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
        env.reset(100)
        stream = env.peek_all()
        for t, s in enumerate(stream):
            expected = (3 * t + 1) % 7
            assert s == expected, f"t={t}: got {s}, expected {expected}"

    def test_symbols_in_alphabet(self):
        env = ModularArithmeticEnv(modulus=11)
        env.reset(1000)
        stream = env.peek_all()
        assert all(0 <= s < 11 for s in stream)

    def test_different_moduli(self):
        for mod in [3, 5, 7, 11, 13]:
            env = ModularArithmeticEnv(modulus=mod, slope=2, intercept=0)
            env.reset(100)
            stream = env.peek_all()
            for t, s in enumerate(stream):
                assert s == (2 * t) % mod

    def test_check_expression(self):
        from ouroboros.compression.program_synthesis import build_linear_modular
        env = ModularArithmeticEnv(7, 3, 1)
        env.reset(100)
        expr = build_linear_modular(3, 1, 7)
        assert env.check_expression(expr)

    def test_optimal_bits_reasonable(self):
        env = ModularArithmeticEnv(modulus=7)
        bits = env.optimal_bits()
        # 2 * log2(7) ≈ 5.6 bits
        assert 5.0 < bits < 6.5


class TestFibonacciModEnv:

    def test_starts_with_zero_one(self):
        env = FibonacciModEnv(modulus=100)
        env.reset(10)
        stream = env.peek_all()
        # Fibonacci: 0,1,1,2,3,5,8,13,21,34,...
        assert stream[0] == 0
        assert stream[1] == 1

    def test_recurrence_satisfied(self):
        env = FibonacciModEnv(modulus=7)
        env.reset(50)
        stream = env.peek_all()
        for i in range(2, len(stream)):
            assert stream[i] == (stream[i-1] + stream[i-2]) % 7

    def test_pisano_period_positive(self):
        env = FibonacciModEnv(7)
        p = env.pisano_period()
        assert p > 0


class TestPrimeSequenceEnv:

    def test_first_few_primes(self):
        env = PrimeSequenceEnv()
        env.reset(20)
        stream = env.peek_all()
        # 0,1=non-prime, 2,3=prime, 4=non, 5=prime, 6=non, 7=prime
        assert stream[0] == 0   # 0 not prime
        assert stream[1] == 0   # 1 not prime
        assert stream[2] == 1   # 2 is prime
        assert stream[3] == 1   # 3 is prime
        assert stream[4] == 0   # 4 not prime
        assert stream[5] == 1   # 5 is prime

    def test_alphabet_size_is_two(self):
        env = PrimeSequenceEnv()
        assert env.alphabet_size == 2

    def test_no_symbol_outside_alphabet(self):
        env = PrimeSequenceEnv()
        env.reset(1000)
        stream = env.peek_all()
        assert all(s in (0, 1) for s in stream)


class TestNoiseEnv:

    def test_all_symbols_valid(self):
        env = NoiseEnv(alphabet_size=4)
        env.reset(10000)
        stream = env.peek_all()
        assert all(0 <= s < 4 for s in stream)

    def test_approximately_uniform(self):
        env = NoiseEnv(alphabet_size=4, seed=0)
        env.reset(10000)
        stream = env.peek_all()
        counts = np.bincount(stream, minlength=4)
        fracs = counts / len(stream)
        for f in fracs:
            assert abs(f - 0.25) < 0.04, f"Expected 25%±4%, got {f*100:.1f}%"

    def test_different_seeds_different_streams(self):
        env1 = NoiseEnv(seed=1)
        env2 = NoiseEnv(seed=2)
        env1.reset(100)
        env2.reset(100)
        assert env1.peek_all() != env2.peek_all()