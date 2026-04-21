"""
AlphabetScaler — Automatically selects the best evaluation strategy.

Given an alphabet size N and stream length L, the scaler decides:
  N ≤ 20, L ≤ 1000: use standard BeamSearchSynthesizer (already fast enough)
  N ≤ 100, L ≤ 5000: use SparseBeamSearch with NumPy (5-15x speedup)
  N > 100, L > 1000: use SparseBeamSearch with GPU if available (20-100x speedup)

This allows OUROBOROS to handle:
  - JOINT_MOD=77  (CRT experiment): N=77 → NumPy sparse
  - JOINT_MOD=143 (13×11):         N=143 → NumPy sparse or GPU
  - JOINT_MOD=221 (13×17):         N=221 → GPU (or slow NumPy)
  - JOINT_MOD=1001 (7×11×13):      N=1001 → GPU required

The AlphabetScaler is the entry point for any code that needs to
search expressions over a large alphabet — it transparently routes
to the right backend.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Literal

from ouroboros.synthesis.expr_node import ExprNode


@dataclass
class ScalerConfig:
    """Thresholds for strategy selection."""
    small_alphabet_threshold: int = 20
    medium_alphabet_threshold: int = 100
    small_stream_threshold: int = 1000
    beam_width: int = 25
    max_depth: int = 4
    max_lag: int = 5
    const_range_multiplier: float = 1.5  # const_range = alphabet * multiplier
    mcmc_iterations: int = 150
    random_seed: int = 42


class AlphabetScaler:
    """
    Routes expression search to the appropriate backend based on
    alphabet size and stream length.
    """

    def __init__(self, config: ScalerConfig = None):
        self.cfg = config or ScalerConfig()

    def _choose_strategy(
        self,
        alphabet_size: int,
        stream_length: int,
    ) -> Literal["standard", "sparse_numpy", "sparse_gpu"]:
        """Choose evaluation strategy."""
        if alphabet_size <= self.cfg.small_alphabet_threshold:
            return "standard"
        if alphabet_size <= self.cfg.medium_alphabet_threshold:
            return "sparse_numpy"
        return "sparse_gpu"   # may fall back to sparse_numpy if no GPU

    def search(
        self,
        observations: List[int],
        alphabet_size: int,
        verbose: bool = False,
    ) -> Optional[ExprNode]:
        """
        Search for the best expression, automatically choosing the
        evaluation strategy based on alphabet size.
        """
        strategy = self._choose_strategy(alphabet_size, len(observations))
        const_range = int(alphabet_size * self.cfg.const_range_multiplier)

        if verbose:
            print(f"  AlphabetScaler: alphabet={alphabet_size}, "
                  f"stream={len(observations)}, strategy={strategy}")

        if strategy == "standard":
            from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
            cfg = BeamConfig(
                beam_width=self.cfg.beam_width,
                const_range=const_range,
                max_depth=self.cfg.max_depth,
                max_lag=self.cfg.max_lag,
                mcmc_iterations=self.cfg.mcmc_iterations,
                random_seed=self.cfg.random_seed,
            )
            return BeamSearchSynthesizer(cfg).search(observations)

        else:  # sparse_numpy or sparse_gpu
            from ouroboros.acceleration.sparse_beam import SparseBeamSearch, SparseBeamConfig
            use_gpu = (strategy == "sparse_gpu")
            cfg = SparseBeamConfig(
                beam_width=self.cfg.beam_width,
                max_depth=self.cfg.max_depth,
                const_range=const_range,
                max_lag=self.cfg.max_lag,
                mcmc_iterations=self.cfg.mcmc_iterations,
                use_gpu=use_gpu,
                random_seed=self.cfg.random_seed,
            )
            return SparseBeamSearch(cfg).search(
                observations, alphabet_size, verbose=verbose
            )

    @staticmethod
    def benchmark(
        alphabet_sizes: List[int],
        stream_length: int = 500,
    ) -> dict:
        """
        Benchmark evaluation speed across alphabet sizes.
        Returns dict mapping alphabet_size → (time_seconds, strategy_used).
        """
        import time
        import random
        results = {}
        scaler = AlphabetScaler()

        for N in alphabet_sizes:
            rng = random.Random(42)
            # Generate random observations (no structure — worst case for compression)
            obs = [rng.randint(0, N - 1) for _ in range(stream_length)]

            strategy = scaler._choose_strategy(N, stream_length)
            start = time.time()
            scaler.search(obs, alphabet_size=N, verbose=False)
            elapsed = time.time() - start

            results[N] = {"time_seconds": elapsed, "strategy": strategy}
            print(f"  N={N:4d} ({strategy:12s}): {elapsed:.2f}s")

        return results
