"""
Gaussian MDL engine for continuous observation streams.

Replaces Shannon entropy (used for discrete sequences) with:
  MDL_cost = program_bits + data_bits_given_program
  data_bits_given_program = N/2 * log2(2πe*σ²_residual)

where σ²_residual = mean squared error of the program's predictions.

The key insight: if a program predicts perfectly (σ²→0), the data
cost → -∞ (infinite compression). In practice we clip at a minimum
sigma to avoid numerical issues.

For model selection between two programs P1 and P2:
  Pick P1 if MDL(P1) < MDL(P2)
  i.e., prefer the program with lower total bits

Program cost (description length):
  Each expression tree node contributes to program_bits.
  Constants require encoding their precision.
  We use a penalized complexity measure:
    program_bits = lambda_prog * node_count + lambda_const * sum(constant_precision_bits)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional


# Minimum residual sigma to prevent division by zero / log(0)
MIN_SIGMA = 1e-6

# Bits per unit of program complexity
LAMBDA_PROG = 2.0       # cost per AST node (same as discrete λ)
LAMBDA_CONST = 8.0      # bits to encode one float constant to 3 decimal places


@dataclass
class GaussianMDLResult:
    """Result of evaluating a continuous program under Gaussian MDL."""
    # Raw prediction errors
    predictions: List[float]
    actuals: List[float]
    residuals: List[float]

    # Gaussian model parameters
    residual_mean: float
    residual_sigma: float      # estimated noise level
    residual_mse: float        # mean squared error

    # MDL costs (in bits)
    program_bits: float        # description length of the program
    data_bits: float           # NLL of data given program (Gaussian)
    total_mdl_cost: float      # program_bits + data_bits

    # Compression ratio (vs naive: predict 0.0 always)
    compression_ratio: float
    naive_data_bits: float

    # Derived quality metrics
    r_squared: float           # 1 - (SS_res / SS_tot), from 0.0 to 1.0
    is_good_fit: bool          # r_squared > 0.95


def gaussian_nll_bits(residuals: List[float], sigma: float) -> float:
    """
    Gaussian negative log-likelihood in bits.
    
    For N samples with residual sigma:
      NLL = N/2 * log2(2πe) + N * log2(sigma)
    
    This is the data description length assuming Gaussian errors.
    """
    n = len(residuals)
    if n == 0:
        return 0.0
    sigma = max(sigma, MIN_SIGMA)
    nll_nats = (n / 2.0) * math.log(2 * math.pi * math.e) + n * math.log(sigma)
    return nll_nats / math.log(2.0)


def estimate_residual_sigma(residuals: List[float]) -> float:
    """MLE estimate of Gaussian sigma from residuals."""
    n = len(residuals)
    if n == 0:
        return MIN_SIGMA
    mse = sum(r ** 2 for r in residuals) / n
    return max(math.sqrt(mse), MIN_SIGMA)


def program_description_bits(node_count: int, constant_count: int) -> float:
    """
    Penalized description length of a program.
    Larger AST → more bits.
    More constants → more bits (each constant encodes at 3 decimal places ≈ 10 bits).
    """
    return LAMBDA_PROG * node_count + LAMBDA_CONST * constant_count


def compute_gaussian_mdl(
    predictions: List[float],
    actuals: List[float],
    program_node_count: int = 0,
    program_constant_count: int = 0,
    node_count: Optional[int] = None,
    constant_count: Optional[int] = None,
) -> GaussianMDLResult:
    # Support both naming conventions
    if node_count is not None:
        program_node_count = node_count
    if constant_count is not None:
        program_constant_count = constant_count
    """
    Compute the full Gaussian MDL cost for a continuous program.
    
    Args:
        predictions: program's predicted values (length N)
        actuals: observed values (length N)
        program_node_count: number of AST nodes in the program
        program_constant_count: number of float constants in the program
    
    Returns:
        GaussianMDLResult with all cost components broken down
    """
    n = len(actuals)
    assert len(predictions) == n, "predictions and actuals must have same length"

    residuals = [p - a for p, a in zip(predictions, actuals)]
    mse = sum(r ** 2 for r in residuals) / n if n > 0 else 0.0
    residual_mean = sum(residuals) / n if n > 0 else 0.0
    residual_sigma = estimate_residual_sigma(residuals)

    # Naive baseline: predict 0.0 always
    naive_sigma = estimate_residual_sigma(actuals)
    naive_data_bits = gaussian_nll_bits(actuals, naive_sigma)

    # Program under evaluation
    prog_bits = program_description_bits(program_node_count, program_constant_count)
    data_bits = gaussian_nll_bits(residuals, residual_sigma)
    total = prog_bits + data_bits

    # Compression ratio: how much better than naive?
    naive_total = naive_data_bits  # naive program has 0 bits (trivial)
    compression_ratio = total / naive_total if naive_total > 0 else 1.0

    # R² metric
    ss_res = sum(r ** 2 for r in residuals)
    ss_tot = sum((a - sum(actuals) / n) ** 2 for a in actuals) if n > 1 else 1.0
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)
    r_squared = max(0.0, min(1.0, r_squared))

    return GaussianMDLResult(
        predictions=predictions,
        actuals=actuals,
        residuals=residuals,
        residual_mean=residual_mean,
        residual_sigma=residual_sigma,
        residual_mse=mse,
        program_bits=prog_bits,
        data_bits=data_bits,
        total_mdl_cost=total,
        compression_ratio=compression_ratio,
        naive_data_bits=naive_data_bits,
        r_squared=r_squared,
        is_good_fit=r_squared > 0.95,
    )


@dataclass
class ContinuousMDLConfig:
    """Configuration for the Gaussian MDL engine."""
    lambda_prog: float = LAMBDA_PROG
    lambda_const: float = LAMBDA_CONST
    min_sigma: float = MIN_SIGMA
    good_fit_threshold: float = 0.95   # R² above this → axiom candidate
    noise_detection_threshold: float = 0.05  # R² below this → treat as noise