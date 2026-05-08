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

        # --- FIX 1: Use integer residuals + Shannon entropy, not Gaussian MSE ---
        # Count how often each residual value appears
        residuals = [int(round(p)) - int(round(a)) for p, a in zip(predictions, actuals)]
        max_resid = max(abs(r) for r in residuals) if residuals else 0

        if max_resid == 0:
            # Perfect prediction
            error_bits = 0.0
        else:
            # Encode each residual as log2(2 * max_resid + 1) bits (uniform over range)
            # This never goes negative
            error_bits = n * math.log2(2 * max_resid + 1)

        # --- FIX 2: Scale program bits by log2(alphabet_size) for context ---
        # Larger alphabets need more bits to specify constants
        alpha_bits = math.log2(max(alphabet_size, 2))
        program_bits = (node_count * 4.0) + (constant_count * alpha_bits)

        total = program_bits + error_bits

        r = MDLResult(total_bits=total, program_bits=program_bits, error_bits=error_bits)
        r.total_mdl_cost = total
        return r