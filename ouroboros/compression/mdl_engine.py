# ouroboros/compression/mdl_engine.py

from dataclasses import dataclass
from typing import List, Optional

from .mdl import MDLCost


@dataclass
class MDLResult:
    total_bits: float
    program_bits: float
    error_bits: float


class MDLEngine:
    """
    Thin wrapper around existing MDLCost to match expected interface.
    """

    def __init__(self, lambda_weight: float = 1.0):
        self.cost = MDLCost(lambda_weight=lambda_weight)

    def evaluate(
        self,
        program_bytes: bytes,
        predictions: List[int],
        actuals: List[int],
        alphabet_size: int,
    ) -> MDLResult:
        program_bits = self.cost.program_description_bits(program_bytes)
        error_bits = self.cost.prediction_error_bits(
            predictions, actuals, alphabet_size
        )
        total_bits = self.cost.total_cost(
            program_bytes, predictions, actuals, alphabet_size
        )

        return MDLResult(
            total_bits=total_bits,
            program_bits=program_bits,
            error_bits=error_bits,
        )

    def compute(
        self,
        predictions,
        actuals,
        node_count: int = 0,
        constant_count: int = 0,
        alphabet_size: int = 256,
    ):
        from dataclasses import dataclass
        program_bytes = b'x' * max(1, node_count)
        r = self.evaluate(program_bytes, [int(p) for p in predictions],
                          [int(a) for a in actuals], alphabet_size)
        r.total_mdl_cost = r.total_bits
        return r
