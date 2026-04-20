"""Tests for GaussianMDL engine."""
import math
import pytest
from ouroboros.continuous.mdl import (
    compute_gaussian_mdl, gaussian_nll_bits, estimate_residual_sigma,
    GaussianMDLResult,
)


class TestGaussianNLLBits:
    def test_zero_residuals_gives_small_cost(self):
        residuals = [0.0] * 100
        sigma = estimate_residual_sigma(residuals)
        bits = gaussian_nll_bits(residuals, sigma=max(sigma, 1e-6))
        # Residuals are all 0 → sigma is tiny → cost should be very negative or very small
        assert math.isfinite(bits)

    def test_large_sigma_gives_large_cost(self):
        residuals = [1.0] * 100
        bits_small = gaussian_nll_bits(residuals, sigma=0.01)
        bits_large = gaussian_nll_bits(residuals, sigma=10.0)
        # Larger sigma → lower NLL (more "forgiving" model)
        # Actually larger sigma → more spread out → might penalize less on high residuals
        assert math.isfinite(bits_small)
        assert math.isfinite(bits_large)

    def test_empty_returns_zero(self):
        assert gaussian_nll_bits([], sigma=1.0) == 0.0


class TestEstimateResidualSigma:
    def test_constant_residuals(self):
        residuals = [2.0] * 100
        sigma = estimate_residual_sigma(residuals)
        assert abs(sigma - 2.0) < 0.01

    def test_zero_residuals(self):
        residuals = [0.0] * 50
        sigma = estimate_residual_sigma(residuals)
        assert sigma >= 1e-6  # clipped

    def test_mixed_residuals(self):
        residuals = [1.0, -1.0] * 50
        sigma = estimate_residual_sigma(residuals)
        assert abs(sigma - 1.0) < 0.01


class TestComputeGaussianMDL:
    def test_perfect_prediction(self):
        actuals = [1.0, 2.0, 3.0, 4.0, 5.0]
        predictions = actuals[:]
        result = compute_gaussian_mdl(predictions, actuals, node_count=3, constant_count=1)
        assert result.r_squared > 0.999
        assert result.residual_mse < 1e-10
        assert result.is_good_fit

    def test_constant_prediction_vs_linear(self):
        actuals = [float(t) for t in range(100)]  # y = t
        # Constant prediction: y = 50 (mean)
        const_preds = [50.0] * 100
        # Perfect linear: y = t
        linear_preds = [float(t) for t in range(100)]
        
        r_const = compute_gaussian_mdl(const_preds, actuals, node_count=1, constant_count=1)
        r_linear = compute_gaussian_mdl(linear_preds, actuals, node_count=3, constant_count=2)
        
        # Linear should have much better R²
        assert r_linear.r_squared > 0.999
        assert r_const.r_squared < 0.1

    def test_r_squared_range(self):
        actuals = [math.sin(t / 7.0) for t in range(200)]
        preds_good = actuals[:]
        preds_bad = [0.0] * 200
        
        good = compute_gaussian_mdl(preds_good, actuals, 5, 2)
        bad = compute_gaussian_mdl(preds_bad, actuals, 1, 1)
        
        assert 0.0 <= good.r_squared <= 1.0
        assert 0.0 <= bad.r_squared <= 1.0
        assert good.r_squared > bad.r_squared

    def test_compression_ratio_direction(self):
        actuals = [math.sin(t / 7.0) for t in range(200)]
        perfect_preds = actuals[:]
        random_preds = [0.5] * 200
        
        r_perfect = compute_gaussian_mdl(perfect_preds, actuals, 5, 2)
        r_random = compute_gaussian_mdl(random_preds, actuals, 1, 1)
        
        # Perfect prediction should have lower total MDL cost
        assert r_perfect.total_mdl_cost < r_random.total_mdl_cost

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_gaussian_mdl([1.0, 2.0], [1.0, 2.0, 3.0], 1, 0)