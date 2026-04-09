# tests/unit/test_mdl.py

"""Unit tests for the MDL compression engine."""

import math
import pytest
from ouroboros.compression.mdl import (
    entropy_bits, naive_bits, compression_ratio,
    zstd_compressed_bits, sequence_to_bytes, MDLCost
)


class TestEntropyBits:

    def test_uniform_binary_is_one_bit(self):
        """Perfectly uniform binary stream → 1 bit/symbol."""
        seq = [0, 1] * 500
        h = entropy_bits(seq, 2)
        assert abs(h - 1.0) < 0.01

    def test_all_same_symbol_is_zero(self):
        """All same symbol → 0 bits (perfectly predictable)."""
        seq = [0] * 1000
        h = entropy_bits(seq, 2)
        assert h < 0.001

    def test_uniform_four_symbols_is_two_bits(self):
        """Uniform over 4 symbols → 2 bits/symbol."""
        seq = [0, 1, 2, 3] * 250
        h = entropy_bits(seq, 4)
        assert abs(h - 2.0) < 0.05

    def test_empty_sequence_is_zero(self):
        assert entropy_bits([], 4) == 0.0

    def test_skewed_distribution_less_than_max(self):
        """Skewed distribution → entropy < log2(alphabet_size)."""
        # 90% zeros, 10% ones
        seq = [0] * 900 + [1] * 100
        h = entropy_bits(seq, 2)
        assert h < 1.0
        assert h > 0.0


class TestNaiveBits:

    def test_binary_stream(self):
        seq = [0, 1] * 50
        bits = naive_bits(seq, 2)
        assert abs(bits - 100 * math.log2(2)) < 0.001

    def test_mod7_stream(self):
        seq = [(3*t+1) % 7 for t in range(100)]
        bits = naive_bits(seq, 7)
        expected = 100 * math.log2(7)
        assert abs(bits - expected) < 0.001

    def test_empty_is_zero(self):
        assert naive_bits([], 7) == 0.0

    def test_alphabet_size_1_is_zero(self):
        assert naive_bits([0]*100, 1) == 0.0


class TestCompressionRatio:

    def test_structured_beats_noise(self):
        """Structured stream should compress better than random."""
        import random; random.seed(42)
        structured = [(3*t+1) % 7 for t in range(1000)]
        noise = [random.randint(0, 6) for _ in range(1000)]

        r_structured = compression_ratio(structured, 7)
        r_noise = compression_ratio(noise, 7)

        assert r_structured < r_noise, (
            f"Structured ratio {r_structured:.3f} should be < noise {r_noise:.3f}"
        )

    def test_alternating_binary_highly_compressible(self):
        """0,1,0,1,... should compress very well."""
        seq = [t % 2 for t in range(10_000)]
        r = compression_ratio(seq, 2)
        assert r < 0.15, f"Expected < 0.15, got {r:.3f}"

    def test_random_barely_compressible(self):
        """Random stream should have ratio close to 1.0."""
        import numpy as np
        rng = np.random.default_rng(0)
        seq = list(rng.integers(0, 7, 1000))
        r = compression_ratio(seq, 7)
        assert r > 0.85, f"Expected > 0.85, got {r:.3f}"

    def test_ratio_positive(self):
        seq = [t % 5 for t in range(500)]
        r = compression_ratio(seq, 5)
        assert r > 0

    def test_empty_returns_one(self):
        assert compression_ratio([], 7) == 1.0


class TestMDLCost:

    def setup_method(self):
        self.mdl = MDLCost(lambda_weight=1.0)

    def test_perfect_predictions_zero_error_bits(self):
        """Perfect predictions → 0 error bits."""
        preds = actuals = [0, 1, 2, 0, 1, 2] * 20
        err = self.mdl.prediction_error_bits(preds, actuals, 3)
        assert err == 0.0

    def test_all_wrong_max_error_bits(self):
        """All wrong → maximum error bits."""
        preds = [0] * 100
        actuals = [1] * 100
        err = self.mdl.prediction_error_bits(preds, actuals, 2)
        # Should be close to 100 * log2(2) = 100 bits
        assert abs(err - 100.0) < 5.0

    def test_shorter_program_wins_equal_accuracy(self):
        """With same predictions, shorter program → lower total cost."""
        preds = actuals = [0] * 200
        short_bytes = b"rule=0"
        long_bytes = b"rule=0; " + b"padding;" * 100

        cost_short = self.mdl.total_cost(short_bytes, preds, actuals, 2)
        cost_long = self.mdl.total_cost(long_bytes, preds, actuals, 2)

        assert cost_short < cost_long

    def test_improvement_positive_for_structured(self):
        """Good program → positive improvement over naive."""
        structured = [t % 2 for t in range(500)]
        preds = [t % 2 for t in range(500)]
        prog = b"t mod 2"

        improvement = self.mdl.improvement_over_naive(prog, preds, structured, 2)
        assert improvement > 0, f"Expected positive improvement, got {improvement}"

    def test_relative_cost_range(self):
        """Relative cost should be in reasonable range."""
        seq = [t % 7 for t in range(200)]
        preds = [t % 7 for t in range(200)]
        prog = b"t mod 7"

        rc = self.mdl.relative_cost(prog, preds, seq, 7)
        assert 0.0 <= rc <= 2.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            self.mdl.prediction_error_bits([0, 1], [0], 2)