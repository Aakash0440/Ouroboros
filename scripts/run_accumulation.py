"""
Run the knowledge accumulation experiment.

Fast mode: 30 sessions × 3 environments × ~3s each ≈ 3 minutes
Full mode: 100 sessions × 5 environments × ~5s each ≈ 40 minutes

Usage:
  python scripts/run_accumulation.py --fast     # 30 sessions
  python scripts/run_accumulation.py --full     # 100 sessions
"""

import sys; sys.path.insert(0, '.')
import argparse
import json
from pathlib import Path
from ouroboros.knowledge.accumulation import AccumulationRunner
from ouroboros.knowledge.growth_tracker import KnowledgeGrowthTracker
from ouroboros.environments.modular import ModularArithmeticEnv
from ouroboros.environments.fibonacci_mod import FibonacciModEnv
from ouroboros.environments.noise import NoiseEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    if args.full:
        n_sessions = 100
        envs = [
            ModularArithmeticEnv(modulus=7, slope=3, intercept=1),
            ModularArithmeticEnv(modulus=11, slope=5, intercept=2),
            ModularArithmeticEnv(modulus=13, slope=7, intercept=3),
            FibonacciModEnv(modulus=7),
            NoiseEnv(alphabet_size=7),
        ]
        report_every = 20
    else:  # fast
        n_sessions = 30
        envs = [
            ModularArithmeticEnv(modulus=7, slope=3, intercept=1),
            ModularArithmeticEnv(modulus=11, slope=5, intercept=2),
            FibonacciModEnv(modulus=7),
        ]
        report_every = 10

    runner = AccumulationRunner(
        n_sessions=n_sessions,
        environments=envs,
        stream_length=80,
        beam_width=6,
        n_iterations=3,
        kb_path="results/accumulation_kb.json",
        verbose=True,
        report_every=report_every,
    )

    record = runner.run(experiment_id=f"accum_{'full' if args.full else 'fast'}")

    # Analyze growth
    tracker = KnowledgeGrowthTracker()
    analysis = tracker.analyze(record.sessions, verbose=True)

    # Generate growth curve data
    curve_data = tracker.generate_growth_curve_data(record.sessions, window=5)

    # Save results
    Path("results").mkdir(exist_ok=True)
    runner.save_record(record, "results/accumulation_record.json")

    with open("results/growth_curve.json", "w") as f:
        json.dump(curve_data, f, indent=2)
    print("Growth curve saved to results/growth_curve.json")


if __name__ == '__main__':
    main()