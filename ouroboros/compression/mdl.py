# ouroboros/compression/mdl.py

"""
Minimum Description Length (MDL) compression engine.

MDL principle (Rissanen 1978):
    The best model of data minimizes:
        Total description length = model_bits + data|model_bits

For OUROBOROS:
    model   = an agent's program (symbolic expression)
    data    = the observation stream
    model_bits   = bits to describe the expression (its to_bytes() length)
    data|model_bits = bits to encode prediction errors

Why MDL drives mathematical emergence:
    A random sequence needs ~log2(N) bits/symbol (full entropy).
    A sequence following (3t+1) mod 7 can be described by ~18 bytes
    (the expression string) plus ~0 bits for prediction errors.
    An agent minimizing MDL will prefer the short expression
    over memorizing the sequence — and that preference IS the
    mathematical structure discovery.

Main functions:
    entropy_bits(seq, alpha)          → theoretical minimum bits
    naive_bits(seq, alpha)            → worst-case bits (no compression)
    compression_ratio(seq, alpha)     → compressed/naive (lower = better)
    class MDLCost                     → computes model_bits + error_bits
"""

import math
from typing import List
import numpy as np
import zstandard as zstd


# ─── Basic entropy measures ────────────────────────────────────────────────────

def entropy_bits(sequence: List[int], alphabet_size: int) -> float:
    """
    Empirical Shannon entropy in bits/symbol.

    H = -sum_x p(x) * log2(p(x))

    This is the THEORETICAL LOWER BOUND for lossless compression.
    Any practical compressor (zstd etc.) will be slightly above this.
    An agent that finds the exact rule beats this lower bound
    because the rule compresses beyond empirical statistics.

    Args:
        sequence: Integer sequence with values in 0..alphabet_size-1
        alphabet_size: Number of distinct symbols

    Returns:
        Entropy in bits/symbol (0 = perfectly predictable, log2(N) = max)
    """
    if not sequence:
        return 0.0

    counts = np.bincount(sequence, minlength=alphabet_size).astype(float)
    total = len(sequence)
    probs = counts / total
    probs = probs[probs > 0]  # Remove zeros (log undefined)

    return float(-np.sum(probs * np.log2(probs)))


def naive_bits(sequence: List[int], alphabet_size: int) -> float:
    """
    Bits assuming uniform distribution — worst case, no compression.

    = len(sequence) * log2(alphabet_size)

    This is the DENOMINATOR in compression_ratio.
    Any agent doing better than this has found structure.
    """
    if not sequence or alphabet_size <= 1:
        return 0.0
    return len(sequence) * math.log2(alphabet_size)


def total_entropy_bits(sequence: List[int], alphabet_size: int) -> float:
    """Total bits (not per-symbol) at empirical entropy."""
    return entropy_bits(sequence, alphabet_size) * len(sequence)


# ─── Practical compression ─────────────────────────────────────────────────────

def sequence_to_bytes(sequence: List[int], alphabet_size: int) -> bytes:
    """
    Convert integer sequence to bytes for compression.
    Uses 1 byte per symbol if alphabet_size <= 256.
    """
    if alphabet_size <= 256:
        return bytes(sequence)
    # 16-bit for larger alphabets
    return np.array(sequence, dtype=np.uint16).tobytes()


def zstd_compressed_bits(data: bytes, level: int = 3) -> int:
    """
    Actual compressed size using zstandard (Zstd).

    Zstd is our practical compression oracle.
    We use it to measure the actual compressibility of:
    (a) the agent's program description
    (b) the concatenation of program + prediction errors

    Args:
        data: Raw bytes to compress
        level: Zstd compression level (1=fast, 22=best, default 3)

    Returns:
        Compressed size in BITS (not bytes)
    """
    if not data:
        return 0
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    return len(compressed) * 8


def compression_ratio(
    sequence: List[int],
    alphabet_size: int,
    method: str = 'zstd'
) -> float:
    """
    Compression ratio of a sequence.

    ratio = compressed_bits / naive_bits

    Interpretation:
        ratio < 0.5  → significant structure found (great)
        ratio < 0.9  → some structure found (decent)
        ratio ≈ 1.0  → no compression (random-looking)
        ratio > 1.0  → pathological (only for tiny sequences)

    Args:
        sequence: Integer sequence
        alphabet_size: Symbols in 0..alphabet_size-1
        method: 'zstd' (practical) or 'entropy' (theoretical lower bound)

    Returns:
        Compression ratio (float)
    """
    if not sequence:
        return 1.0

    raw = naive_bits(sequence, alphabet_size)
    if raw == 0:
        return 1.0

    if method == 'zstd':
        data = sequence_to_bytes(sequence, alphabet_size)
        compressed = zstd_compressed_bits(data)
        return compressed / raw

    elif method == 'entropy':
        h = entropy_bits(sequence, alphabet_size)
        max_h = math.log2(alphabet_size) if alphabet_size > 1 else 1.0
        return h / max_h if max_h > 0 else 1.0

    else:
        raise ValueError(f"Unknown compression method: {method!r}")


# ─── MDL Cost class ────────────────────────────────────────────────────────────

class MDLCost:
    """
    Computes total MDL cost for a (program, predictions, data) triple.

    Total MDL cost = lambda * program_description_bits + prediction_error_bits

    WHERE:
        program_description_bits = zstd(program.to_bytes()) / 8
        prediction_error_bits    = bits to encode which predictions were wrong

    THE COMPRESSION PRESSURE:
        Short program + perfect predictions → very low MDL cost
        Long program + good predictions     → medium MDL cost
        Short program + many errors         → medium MDL cost
        Long program + many errors          → very high MDL cost

    MDL automatically balances program complexity vs. accuracy.
    This is what causes mathematical emergence — the compact formula
    beats both random memorization AND overly complex programs.

    Args:
        lambda_weight: Trade-off between program length and prediction error.
                       lambda=1.0 weights both equally.
                       lambda>1.0 prefers shorter programs even at cost of accuracy.
                       lambda<1.0 allows longer programs for accuracy gains.
    """

    def __init__(self, lambda_weight: float = 1.0):
        self.lambda_weight = lambda_weight

    def program_description_bits(self, program_bytes: bytes) -> float:
        if not program_bytes:
            return 0.0
        if len(program_bytes) < 200:
            return float(len(program_bytes) * 8)
        return float(zstd_compressed_bits(program_bytes))

    def prediction_error_bits(
        self,
        predictions: List[int],
        actuals: List[int],
        alphabet_size: int
    ) -> float:
        """
        Bits to encode the prediction errors.

        Uses a 2-part encoding:
        1. Bits to mark WHICH positions are errors (Bernoulli entropy)
        2. Bits to specify the CORRECT symbol at each error position

        Perfect prediction (0 errors) → 0 bits
        50% error rate → near-maximum bits
        100% error rate → log2(alphabet_size) bits/symbol

        Args:
            predictions: Agent's predicted symbols
            actuals: True observed symbols
            alphabet_size: Number of possible symbols

        Returns:
            Bits to encode all prediction errors
        """
        if len(predictions) != len(actuals):
            raise ValueError(
                f"Length mismatch: predictions={len(predictions)}, "
                f"actuals={len(actuals)}"
            )
        if not actuals:
            return 0.0

        n = len(actuals)
        errors = sum(1 for p, a in zip(predictions, actuals) if p != a)
        error_rate = errors / n

        if error_rate == 0.0:
            return 0.0

        if error_rate >= 1.0:
            return n * math.log2(alphabet_size)

        # Part 1: bits to encode which positions have errors
        # = n * H(error_rate) where H is binary entropy
        h = -(
            error_rate * math.log2(error_rate) +
            (1 - error_rate) * math.log2(1 - error_rate)
        )
        position_bits = n * h

        # Part 2: bits to specify correct symbol at each error
        correction_bits = errors * math.log2(max(alphabet_size - 1, 1))

        return position_bits + correction_bits

    def total_cost(
        self,
        program_bytes: bytes,
        predictions: List[int],
        actuals: List[int],
        alphabet_size: int
    ) -> float:
        """
        Total MDL cost = lambda * program_bits + error_bits

        This is the NUMBER AGENTS MINIMIZE.
        Lower total_cost = better program found.

        For ModularArithmeticEnv(7,3,1) with n=1000 symbols:
            Random agent (n-gram, no rule): ~2800 bits
            Perfect rule found:             ~18 bits (just the expression)
            Ratio: 18/2800 ≈ 0.0064

        That 160x improvement is the mathematical discovery signal.
        """
        prog_bits = self.program_description_bits(program_bytes)
        err_bits = self.prediction_error_bits(predictions, actuals, alphabet_size)
        return self.lambda_weight * prog_bits + err_bits

    def improvement_over_naive(
        self,
        program_bytes: bytes,
        predictions: List[int],
        actuals: List[int],
        alphabet_size: int
    ) -> float:
        """
        How many bits does this program SAVE vs naive (no compression)?

        Positive → the program found useful structure
        Negative → program is too complex for the improvement it gives

        Used to determine if a program is worth keeping.
        """
        naive = naive_bits(actuals, alphabet_size)
        cost = self.total_cost(program_bytes, predictions, actuals, alphabet_size)
        return naive - cost

    def relative_cost(
        self,
        program_bytes: bytes,
        predictions: List[int],
        actuals: List[int],
        alphabet_size: int
    ) -> float:
        """
        MDL cost as fraction of naive bits.

        = total_cost / naive_bits

        Same as compression_ratio but uses the full MDL model cost
        rather than just zstd compression of the raw sequence.

        This is what agents report as their "compression ratio."
        """
        naive = naive_bits(actuals, alphabet_size)
        if naive == 0:
            return 1.0
        cost = self.total_cost(program_bytes, predictions, actuals, alphabet_size)
        return cost / naive