"""
Long-range dependency discovery experiment.

Tests whether agents can find recurrences of order 2, 3, 4, 5
using the Berlekamp-Massey algorithm vs beam search.

Expected results:
  - BM finds order-2 (Fibonacci) and order-3 (Tribonacci) instantly
  - BM finds order-5 linear recurrences instantly
  - BM fails on nonlinear sequences (AutoregressiveEnv with nonlinear mixing)
  - Beam search with max_lag=20 handles cases BM misses
"""

import sys; sys.path.insert(0, '.')
import time
from ouroboros.environments.long_range import (
    TribonacciModEnv, LucasSequenceEnv, LinearRecurrenceEnv, AutoregressiveEnv,
)
from ouroboros.synthesis.long_range_beam import LongRangeBeamSearch, LongRangeBeamConfig


def run_experiment():
    print("LONG-RANGE DEPENDENCY DISCOVERY EXPERIMENT")
    print("=" * 60)

    envs = [
        ("Fibonacci(mod=7)",    None, 7),   # handled by existing FibonacciModEnv
        ("Tribonacci(mod=7)",   TribonacciModEnv(modulus=7), 7),
        ("Lucas(mod=11)",       LucasSequenceEnv(modulus=11), 11),
        ("LinearRec(order=4)",  LinearRecurrenceEnv([1,1,1,1], modulus=7, seeds=[0,1,1,2]), 7),
        ("LinearRec(order=5)",  LinearRecurrenceEnv([1,0,1,0,1], modulus=7), 7),
        ("AR(1,7)",             AutoregressiveEnv([(1,1),(7,1)], modulus=7), 7),
    ]

    cfg = LongRangeBeamConfig(
        max_lag=10,
        beam_width=20,
        use_bm_warmstart=True,
        bm_accuracy_threshold=0.95,
    )
    searcher = LongRangeBeamSearch(cfg)

    for env_name, env, modulus in envs:
        if env is None:
            from ouroboros.environments.fibonacci_mod import FibonacciModEnv
            env = FibonacciModEnv(modulus=7)

        seq = env.generate(300)

        start = time.time()
        result = searcher.search(seq, modulus=modulus, environment_name=env_name)
        elapsed = time.time() - start

        print(f"\n{env_name}:")
        print(f"  Method: {result.discovery_method}")
        print(f"  MDL cost: {result.best_mdl_cost:.2f} bits")
        print(f"  Time: {elapsed:.3f}s")
        if result.recurrence_axiom:
            ax = result.recurrence_axiom
            print(f"  Recurrence: {ax.expression_str}")
            print(f"  Fit error: {ax.fit_error:.6f}")
        if result.best_expr:
            print(f"  Expression: {result.best_expr.to_string()[:80]}")


if __name__ == '__main__':
    run_experiment()
    print("\n✅ Long-range experiment complete")