"""
Alphabet scaling benchmark.

Measures: how does search time scale with alphabet size N?
Compares standard BeamSearch vs SparseBeamSearch (NumPy) vs GPU.

Expected results (CPU only):
  N=7:   standard, ~0.1s
  N=13:  standard, ~0.3s
  N=77:  sparse_numpy, ~1.5s (vs ~15s with standard)
  N=143: sparse_numpy, ~3s
  N=221: sparse_gpu (or sparse_numpy), ~8s (vs minutes with standard)
"""

import sys; sys.path.insert(0, '.')
import time
import random

from ouroboros.acceleration.alphabet_scaler import AlphabetScaler


def run_benchmark():
    print("ALPHABET SCALING BENCHMARK")
    print("=" * 60)
    print(f"{'N':>6}  {'Strategy':>15}  {'Time(s)':>10}  {'Result':>10}")
    print("-" * 60)

    alphabet_sizes = [7, 13, 25, 50, 77, 143, 221]
    stream_length = 300
    scaler = AlphabetScaler()

    for N in alphabet_sizes:
        rng = random.Random(42)
        # Generate structured data: (3*t+1) % N
        obs = [(3*t + 1) % N for t in range(stream_length)]

        strategy = scaler._choose_strategy(N, stream_length)
        start = time.time()
        result = scaler.search(obs, alphabet_size=N, verbose=False)
        elapsed = time.time() - start

        result_str = result.to_string()[:20] if result else "None"
        print(f"  {N:4d}  {strategy:>15}  {elapsed:>10.2f}  {result_str:>20}")

    print("\n")

    # CRT large experiment
    print("CRT LARGE ALPHABET EXPERIMENT")
    print("=" * 60)
    from ouroboros.environments.large_alphabet import CRTLargeEnv

    for (p, q) in [(7, 11), (13, 17)]:
        env = CRTLargeEnv(mod1=p, mod2=q)
        obs = env.generate(300)
        N = env.alphabet_size
        print(f"\nCRT({p}×{q}), N={N}:")

        start = time.time()
        result = scaler.search(obs, alphabet_size=N, verbose=True)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.2f}s")
        if result:
            print(f"  Expression: {result.to_string()[:60]}")


if __name__ == '__main__':
    run_benchmark()
    print("\n✅ Alphabet scaling benchmark complete")