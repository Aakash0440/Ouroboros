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
        int_preds = [int(p) for p in predictions]
        int_acts  = [int(a) for a in actuals]
        if alphabet_size <= 0:
            alphabet_size = max(len(set(int_acts)), 2)
        perfect = (int_preds == int_acts)
        if perfect:
            free_params = max(1, constant_count - 1) if constant_count > 0 else max(1, node_count - 1)
            program_bits = free_params * math.log2(max(alphabet_size, 2))
            r = MDLResult(total_bits=program_bits, program_bits=program_bits, error_bits=0.0)
        else:
            complexity_nodes = max(1, constant_count if constant_count > 0 else node_count)
            program_bytes = b'x' * complexity_nodes
            r = self.evaluate(program_bytes, int_preds, int_acts, alphabet_size)
        r.total_mdl_cost = r.total_bits
        return r
