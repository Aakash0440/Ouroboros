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
EOF
bashcat > ouroboros/environments/large_alphabet.py << 'EOF'
"""
Large-alphabet environments for testing scalability.

These environments operate over alphabets of size N > 100,
previously infeasible due to beam search scoring overhead.
With SparseBeamSearch (Day 27), they become tractable.
"""

from __future__ import annotations
from typing import List
from ouroboros.environments.base import Environment


class CRTLargeEnv(Environment):
    """
    Joint CRT environment with large alphabet.

    Interleaves two modular streams with moduli p and q.
    The joint alphabet has size p*q.

    At (p=13, q=17): alphabet_size=221
    At (p=13, q=11, r=7): alphabet_size=1001 (three-way CRT)

    obs[t] = (slope1 * t + int1) % p  for even t
    obs[t] = (slope2 * t + int2) % q  for odd t
    (expressed in the joint alphabet Z_{p*q})
    """

    def __init__(
        self,
        mod1: int = 13,
        slope1: int = 3,
        int1: int = 1,
        mod2: int = 17,
        slope2: int = 5,
        int2: int = 2,
        name: str = None,
        seed: int = 42,
    ):
        joint_mod = mod1 * mod2
        super().__init__(
            name=name or f"CRTLarge({mod1}×{mod2})",
            seed=seed,
        )
        self.mod1 = mod1
        self.slope1 = slope1
        self.int1 = int1
        self.mod2 = mod2
        self.slope2 = slope2
        self.int2 = int2
        self.joint_mod = joint_mod

    def generate(self, length: int, start: int = 0) -> List[int]:
        result = []
        for t in range(start, start + length):
            if t % 2 == 0:
                # Project sub-stream 1 into joint alphabet
                val1 = (self.slope1 * t + self.int1) % self.mod1
                # CRT lift: val1 mod mod1, 0 mod mod2
                # Simplification: encode as val1 * mod2
                joint_val = val1 * self.mod2
            else:
                # Project sub-stream 2 into joint alphabet
                val2 = (self.slope2 * t + self.int2) % self.mod2
                joint_val = val2 * self.mod1
            result.append(joint_val % self.joint_mod)
        return result

    @property
    def alphabet_size(self) -> int:
        return self.joint_mod

    @property
    def ground_truth_joint_mod(self) -> int:
        return self.joint_mod


class TripleCRTEnv(Environment):
    """
    Three-way CRT environment: alphabet = p*q*r.

    At (p=7, q=11, r=13): alphabet = 1001
    Requires SparseBeamSearch with GPU for practical runtime.
    """

    def __init__(
        self,
        mod1: int = 7,
        mod2: int = 11,
        mod3: int = 13,
        seed: int = 42,
    ):
        joint = mod1 * mod2 * mod3
        super().__init__(name=f"TripleCRT({mod1}×{mod2}×{mod3})", seed=seed)
        self.mod1, self.mod2, self.mod3 = mod1, mod2, mod3
        self.joint_mod = joint

    def generate(self, length: int, start: int = 0) -> List[int]:
        result = []
        for t in range(start, start + length):
            cycle = t % 3
            if cycle == 0:
                val = ((3 * t + 1) % self.mod1) * (self.mod2 * self.mod3)
            elif cycle == 1:
                val = ((5 * t + 2) % self.mod2) * (self.mod1 * self.mod3)
            else:
                val = ((7 * t + 3) % self.mod3) * (self.mod1 * self.mod2)
            result.append(val % self.joint_mod)
        return result

    @property
    def alphabet_size(self) -> int:
        return self.joint_mod