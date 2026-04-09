# ouroboros/agents/base_agent.py

"""
BaseAgent — Phase 1 agent using n-gram program search under MDL pressure.

A Phase 1 agent is defined by three things:
1. PROGRAM — an n-gram prediction table: context → predicted_next_symbol
2. SEARCH — try different context lengths (k), keep the MDL-optimal one
3. METRICS — track compression ratio over time

Why n-gram first, then symbolic in Day 2?
Because n-gram IS a valid program representation — it has
a measurable description length (the table size) and makes
predictions (look up context in table). The MDL framework
treats it just like a symbolic expression.
The n-gram agent is the BASELINE that symbolic synthesis must beat.

The key insight:
    A modular arithmetic stream with period 7 has ONLY 7 distinct
    n-grams of length 1 (it cycles). That n-gram table has 7 entries.
    A truly random stream of alphabet-7 symbols might have many more.
    MDL penalizes the larger table → periodic structure gets rewarded.

Phase 2 will add symbolic programs that find the algebraic rule.
The symbolic program "(3t+1) mod 7" is ~18 bytes. The n-gram table
for the same rule is ~100 bytes. MDL picks the 18-byte version.
THAT is when modular arithmetic "emerges."
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import numpy as np

from ouroboros.compression.mdl import MDLCost, naive_bits


@dataclass
class NGramProgram:
    """
    An n-gram prediction program.

    For context length k:
        table maps (last k symbols) → most likely next symbol

    The program description length = compressed bytes of the table.
    Larger tables = more bits = MDL penalty.

    Fields:
        context_length: k (the n in n-gram)
        table: context_tuple → predicted_next_symbol
        description_bits: cached MDL description length
    """
    context_length: int = 1
    table: Dict[Tuple[int, ...], int] = field(default_factory=dict)
    description_bits: float = 0.0

    def predict(self, history: List[int]) -> int:
        """
        Predict next symbol using the n-gram table.

        Fallback strategy if context not seen:
        1. Try to find a matching shorter context
        2. Return 0 if all else fails
        """
        if len(history) < self.context_length:
            return 0

        # Try exact context
        context = tuple(history[-self.context_length:])
        if context in self.table:
            return self.table[context]

        # Backoff: try shorter contexts
        for k in range(self.context_length - 1, 0, -1):
            short_ctx = tuple(history[-k:])
            # Find any table entry that ends with this context
            for tbl_ctx, pred in self.table.items():
                if tbl_ctx[-k:] == short_ctx:
                    return pred

        # Final fallback: most common prediction in table
        if self.table:
            return Counter(self.table.values()).most_common(1)[0][0]
        return 0

    def to_bytes(self) -> bytes:
        """Serialize table to bytes for MDL measurement."""
        if not self.table:
            return b"empty"
        parts = [f"k={self.context_length}"]
        for ctx, pred in sorted(self.table.items()):
            ctx_str = ','.join(map(str, ctx))
            parts.append(f"{ctx_str}:{pred}")
        return '\n'.join(parts).encode('utf-8')

    @property
    def num_entries(self) -> int:
        return len(self.table)

    def __repr__(self) -> str:
        return (
            f"NGramProgram(k={self.context_length}, "
            f"entries={self.num_entries}, "
            f"bits={self.description_bits:.1f})"
        )


def build_ngram_table(
    history: List[int],
    context_length: int
) -> Dict[Tuple[int, ...], int]:
    """
    Build an n-gram prediction table from observation history.

    For each distinct context of length k, records the most
    frequently occurring next symbol.

    This is Maximum Likelihood estimation of the n-gram model.

    KEY INSIGHT FOR EMERGENCE:
    A modular arithmetic stream (3t+1) mod 7 has exactly 7 distinct
    length-1 contexts (0..6). Each context maps to exactly one
    next symbol (deterministic). The table has 7 entries.

    A random stream of alphabet-7 also has ~7 length-1 contexts,
    but each maps to a DIFFERENT most-common symbol — and the
    predictions are often wrong, so error_bits is much higher.

    MDL distinguishes these cases via the error_bits term.
    """
    context_counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)

    for i in range(context_length, len(history)):
        ctx = tuple(history[i - context_length:i])
        next_sym = history[i]
        context_counts[ctx][next_sym] += 1

    # Maximum likelihood: most frequent next symbol per context
    return {
        ctx: counts.most_common(1)[0][0]
        for ctx, counts in context_counts.items()
    }


class BaseAgent:
    """
    Phase 1 agent — n-gram compression under MDL pressure.

    The agent's goal: find the shortest program (smallest MDL cost)
    that predicts the observation stream.

    Lifecycle per evaluation step:
    1. observe() — receive new symbols
    2. search_and_update() — try different context lengths k=1..max_k
    3. keep the k with lowest MDL cost
    4. measure_compression_ratio() — report performance

    Across 10,000 symbols with eval_interval=200:
    50 search rounds × 8 context lengths = 400 programs evaluated.

    Args:
        agent_id: Unique integer ID
        alphabet_size: Number of symbols in the environment
        max_context_length: Maximum k to search (default 8)
        lambda_weight: MDL regularization (default 1.0)
        seed: Random seed (for any stochastic elements)
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        max_context_length: int = 8,
        lambda_weight: float = 1.0,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.alphabet_size = alphabet_size
        self.max_context_length = max_context_length
        self.rng = np.random.default_rng(seed + agent_id * 17)

        self.mdl = MDLCost(lambda_weight=lambda_weight)
        self.program = NGramProgram(context_length=1)

        # All observations seen so far
        self.observation_history: List[int] = []

        # Tracked metrics
        self.compression_ratios: List[float] = []
        self.mdl_costs: List[float] = []
        self.search_count: int = 0

    # ── Observation ─────────────────────────────────────────────────────────────

    def observe(self, symbols: List[int]) -> None:
        """Add symbols to observation history."""
        self.observation_history.extend(symbols)

    def set_history(self, symbols: List[int]) -> None:
        """Replace observation history (used in batch experiments)."""
        self.observation_history = list(symbols)

    # ── Prediction ──────────────────────────────────────────────────────────────

    def predict(self) -> int:
        """Predict next symbol using current program."""
        return self.program.predict(self.observation_history)

    # ── MDL Search ──────────────────────────────────────────────────────────────

    def search_and_update(self) -> float:
        """
        Try all context lengths k=1..max_context_length.
        Keep the program with lowest MDL cost.

        Returns:
            Best MDL cost found (lower is better)
        """
        history = self.observation_history
        if len(history) < 4:
            return float('inf')

        best_cost = float('inf')
        best_program = self.program
        max_k = min(self.max_context_length, len(history) // 4)

        for k in range(1, max_k + 1):
            # Build n-gram table for context length k
            table = build_ngram_table(history, k)
            candidate = NGramProgram(context_length=k, table=table)

            # Generate predictions over the full history
            predictions = []
            for i in range(k, len(history)):
                pred = candidate.predict(history[:i])
                predictions.append(pred)

            actuals = history[k:]

            # Compute MDL cost
            prog_bytes = candidate.to_bytes()
            cost = self.mdl.total_cost(
                prog_bytes, predictions, actuals, self.alphabet_size
            )
            candidate.description_bits = self.mdl.program_description_bits(prog_bytes)

            if cost < best_cost:
                best_cost = cost
                best_program = candidate

        self.program = best_program
        self.mdl_costs.append(best_cost)
        self.search_count += 1
        return best_cost

    # ── Compression Measurement ─────────────────────────────────────────────────

    def measure_compression_ratio(self) -> float:
        """
        Measure how well the current program compresses the history.

        ratio = total_mdl_cost / naive_bits

        < 1.0 → agent found useful structure
        ≈ 1.0 → no useful compression
        > 1.0 → pathological (shouldn't happen with decent history)

        Returns: compression ratio (float)
        """
        history = self.observation_history
        if not history:
            return 1.0

        k = self.program.context_length
        if len(history) <= k:
            return 1.0

        predictions = [
            self.program.predict(history[:i])
            for i in range(k, len(history))
        ]
        actuals = history[k:]

        prog_bytes = self.program.to_bytes()
        cost = self.mdl.total_cost(
            prog_bytes, predictions, actuals, self.alphabet_size
        )

        nb = naive_bits(actuals, self.alphabet_size)
        ratio = cost / nb if nb > 0 else 1.0

        self.compression_ratios.append(ratio)
        return ratio

    # ── Status ──────────────────────────────────────────────────────────────────

    def latest_ratio(self) -> float:
        """Return most recent compression ratio (1.0 if none yet)."""
        return self.compression_ratios[-1] if self.compression_ratios else 1.0

    def status_dict(self) -> dict:
        """Return status as a loggable dictionary."""
        return {
            'agent_id': self.agent_id,
            'observations': len(self.observation_history),
            'context_k': self.program.context_length,
            'table_entries': self.program.num_entries,
            'prog_bits': round(self.program.description_bits, 1),
            'compression_ratio': round(self.latest_ratio(), 4),
            'search_count': self.search_count,
        }

    def __repr__(self) -> str:
        return (
            f"Agent(id={self.agent_id}, "
            f"k={self.program.context_length}, "
            f"ratio={self.latest_ratio():.3f})"
        )