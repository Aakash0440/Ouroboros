from dataclasses import dataclass
from typing import List
import math
from .mdl import MDLCost

@dataclass
class MDLResult:
    total_bits: float
    program_bits: float
    error_bits: float


class MDLEngine:
    def __init__(self, lambda_weight: float = 1.0):
        self.cost = MDLCost(lambda_weight=lambda_weight)

    def evaluate(self, program_bytes, predictions, actuals, alphabet_size):
        program_bits = self.cost.program_description_bits(program_bytes)
        error_bits = self.cost.prediction_error_bits(predictions, actuals, alphabet_size)
        total_bits = self.cost.total_cost(program_bytes, predictions, actuals, alphabet_size)
        return MDLResult(total_bits=total_bits, program_bits=program_bits, error_bits=error_bits)

    def compute(self, predictions, actuals, node_count=0, constant_count=0, alphabet_size=0):
        import math

        n = len(actuals)
        if n == 0:
            return MDLResult(total_bits=float('inf'), program_bits=0, error_bits=float('inf'))

        # Mean squared error
        mse = sum((p - a) ** 2 for p, a in zip(predictions, actuals)) / n

        if mse < 1e-9:
            # Perfect or near-perfect: 0 error bits
            error_bits = 0.0
        else:
            # Gaussian MDL — but shift so minimum is 0, not negative
            # log2(sqrt(2*pi*e)) ≈ 2.047 bits — this is the per-sample floor
            # We add it so error_bits is always >= 0
            GAUSS_FLOOR = 0.5 * math.log2(2 * math.pi * math.e)  # ~2.047
            raw = 0.5 * math.log2(mse)
            error_bits = n * max(0.0, raw + GAUSS_FLOOR)

        # Program bits: nodes cost log2(n_node_types) ≈ 6 bits each
        # Constants cost log2(const_range) bits — use alphabet as proxy
        alpha_bits = math.log2(max(alphabet_size, 2))
        program_bits = (node_count * 6.0) + (constant_count * alpha_bits)

        total = program_bits + error_bits

        r = MDLResult(total_bits=total, program_bits=program_bits, error_bits=error_bits)
        r.total_mdl_cost = total
        return r