"""
Run the publication-quality full benchmark.

Time estimate:
  - Compression landmark: ~15 min (2000 stream × 10 seeds)
  - Moduli generalization: ~20 min (4 moduli × 10 seeds × 2000 stream)
  - Convergence rounds: ~15 min (20 rounds × 10 seeds)
  - CRT accuracy: ~20 min (1500 stream × 10 seeds)
  - OOD generalization: ~10 min (10 seeds)
  - Self-improvement: ~10 min (10 seeds)
  Total: ~90 minutes

Usage:
  python scripts/run_full_benchmark.py
  python scripts/run_full_benchmark.py --seeds 5  # faster, lower quality
"""

import sys; sys.path.insert(0, '.')
import argparse
from ouroboros.benchmark.full_runner import FullBenchmarkRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    n_seeds = 5 if args.fast else args.seeds

    runner = FullBenchmarkRunner(
        n_seeds=n_seeds,
        output_dir="results",
        verbose=True,
    )

    results = runner.run_all(save_intermediate=True)

    checks = runner.validate_and_report(
        results,
        comparison_path="results/benchmark_results.json",  # fast-mode results
    )

    runner.generate_updated_latex_tables(results)

    print(f"\n{'='*60}")
    n_pass = sum(1 for c in checks if c.passes_publication)
    if n_pass == len(checks):
        print("✅ ALL RESULTS PUBLICATION-QUALITY")
        print("   Update paper tables with results/full_latex_tables.tex")
    else:
        print(f"⚠️  {len(checks) - n_pass} results need more seeds")
        print("   Consider increasing to 15 seeds for failing experiments")


if __name__ == '__main__':
    main()
    