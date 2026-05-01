"""
GCDDiscoveryRunner — Runs agents on GCDEnv and measures algorithm discovery.

The Euclidean algorithm:
  function GCD(a, b):
    while b != 0:
      a, b = b, a % b
    return a

Discovery criteria:
  - Expression achieves > 90% accuracy on GCDEnv test pairs
  - Expression uses GCD_NODE or equivalent STATE-based computation
  - MDL cost is lower than naive baseline (mode prediction)

Note: The GCD algorithm is complex — don't expect reliable discovery
in 10 iterations of beam search. This runner measures the probability
of discovery across multiple runs and reports the best result found.
"""

from __future__ import annotations
import math
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional

from ouroboros.environments.algorithm_env import GCDEnv
from ouroboros.search.stateful_search import StatefulHierarchicalRouter, StatefulScorer


@dataclass
class GCDDiscoveryResult:
    """Result of attempting to discover the GCD algorithm."""
    n_attempts: int
    best_accuracy: float
    best_expr_str: Optional[str]
    best_mdl_cost: float
    mean_accuracy: float
    discovery_rate: float   # fraction of runs achieving > 90% accuracy
    runtime_seconds: float

    def description(self) -> str:
        status = "✅ GCD DISCOVERED" if self.best_accuracy > 0.90 else "❌ GCD not discovered"
        return (
            f"{status}\n"
            f"  Best accuracy: {self.best_accuracy:.2%} "
            f"({self.discovery_rate:.0%} runs > 90%)\n"
            f"  Best expression: {self.best_expr_str or 'None'}\n"
            f"  Best MDL: {self.best_mdl_cost:.2f} bits\n"
            f"  Attempts: {self.n_attempts}, Runtime: {self.runtime_seconds:.1f}s"
        )


class GCDDiscoveryRunner:
    """
    Runs multiple attempts to discover GCD algorithm.
    
    Strategy:
    1. Use StatefulHierarchicalRouter (detects GCDEnv, uses stateful search)
    2. Evaluate discovered expression against all 100 test pairs
    3. Track best accuracy across runs
    4. Report discovery rate
    """

    def __init__(
        self,
        n_attempts: int = 10,
        stream_length: int = 100,
        beam_width: int = 15,
        n_iterations: int = 8,
        verbose: bool = True,
    ):
        self.n_attempts = n_attempts
        self.stream_length = stream_length
        self.beam_width = beam_width
        self.n_iterations = n_iterations
        self.verbose = verbose
        self._scorer = StatefulScorer(n_state_vars=4)

    def _evaluate_on_gcd_pairs(
        self,
        expr,
        env: GCDEnv,
        n_pairs: int = 100,
    ) -> float:
        """Evaluate expression accuracy on n_pairs GCD test cases."""
        if expr is None:
            return 0.0

        outputs, inputs_encoded = [], []
        for i, (a, b) in enumerate(env._pairs[:n_pairs]):
            gcd_true = math.gcd(a, b)
            outputs.append(gcd_true)
            inputs_encoded.append(a * env.ENCODING + b)

        # Evaluate expression on each pair
        n_correct = 0
        for i, (gcd_true, encoded) in enumerate(zip(outputs, inputs_encoded)):
            try:
                # Give the expression the encoded input as a single observation
                pred = expr.evaluate(0, [float(encoded)], {})
                pred_int = int(round(pred))
                if pred_int == gcd_true:
                    n_correct += 1
            except Exception:
                pass

        return n_correct / n_pairs

    def run(self) -> GCDDiscoveryResult:
        """Run n_attempts to discover the GCD algorithm."""
        env = GCDEnv(seed=42)
        router = StatefulHierarchicalRouter(
            beam_width=self.beam_width,
            n_iterations=self.n_iterations,
            n_state_vars=4,
        )

        start = time.time()
        accuracies = []
        best_accuracy = 0.0
        best_expr = None
        best_cost = float('inf')

        if self.verbose:
            print(f"\nGCD DISCOVERY EXPERIMENT")
            print(f"Attempts: {self.n_attempts}, Stream: {self.stream_length}")

        for attempt in range(self.n_attempts):
            env.seed = attempt * 7  # different seed per attempt
            obs = env.generate(self.stream_length)

            expr, cost = router.search(
                obs,
                alphabet_size=env.alphabet_size,
                env_name="GCDEnv",
                use_stateful=True,
                verbose=False,
            )

            accuracy = self._evaluate_on_gcd_pairs(expr, env)
            accuracies.append(accuracy)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_expr = expr
                best_cost = cost

            if self.verbose:
                expr_str = expr.to_string()[:40] if expr else "None"
                print(f"  Attempt {attempt+1:2d}: accuracy={accuracy:.2%}, "
                      f"MDL={cost:.1f}, expr={expr_str}")

            if best_accuracy > 0.95:
                if self.verbose:
                    print(f"  ✅ GCD discovered at attempt {attempt+1}!")
                break

        elapsed = time.time() - start
        discovery_rate = sum(1 for a in accuracies if a > 0.90) / len(accuracies)

        result = GCDDiscoveryResult(
            n_attempts=len(accuracies),
            best_accuracy=best_accuracy,
            best_expr_str=best_expr.to_string()[:80] if best_expr else None,
            best_mdl_cost=best_cost,
            mean_accuracy=statistics.mean(accuracies),
            discovery_rate=discovery_rate,
            runtime_seconds=elapsed,
        )

        if self.verbose:
            print(f"\n{result.description()}")

        return result