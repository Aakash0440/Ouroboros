"""
Full communication experiment runner.

Run: python experiments/run_communication_experiment.py --n_runs 10 --fast
"""

import sys; sys.path.insert(0, '.')
import argparse
from ouroboros.environments.modular import ModularArithmeticEnv
from experiments.communication_experiment import (
    run_solo_condition, run_comm_condition,
)
from experiments.statistical_tests import StatisticalTester


def make_env_factory(modulus=7, slope=3, intercept=1):
    def factory(seed):
        return ModularArithmeticEnv(modulus=modulus, slope=slope, intercept=intercept, seed=seed)
    return factory


def run_experiment(n_runs: int, n_agents: int, n_rounds: int, beam_width: int, stream_length: int):
    print(f"\nCOMMUNICATION EXPERIMENT")
    print(f"Runs per condition: {n_runs}")
    print(f"Agents: {n_agents}, Rounds: {n_rounds}, BeamWidth: {beam_width}")
    print(f"{'='*60}\n")

    env_factory = make_env_factory(modulus=7, slope=3, intercept=1)

    print("Running SOLO condition...")
    solo_runs = []
    for seed in range(n_runs):
        run = run_solo_condition(
            env_factory=env_factory, n_agents=n_agents,
            n_rounds=n_rounds, stream_length=stream_length,
            beam_width=beam_width, seed=seed,
        )
        solo_runs.append(run)
        print(f"  {run.description()}")

    print("\nRunning COMM condition...")
    comm_runs = []
    for seed in range(n_runs):
        run = run_comm_condition(
            env_factory=env_factory, n_agents=n_agents,
            n_rounds=n_rounds, stream_length=stream_length,
            beam_width=beam_width, seed=seed,
            hint_interval=2,
        )
        comm_runs.append(run)
        print(f"  {run.description()}")

    # Statistical analysis
    print("\nRunning statistical analysis...")
    tester = StatisticalTester()
    analysis = tester.analyze(solo_runs, comm_runs, n_rounds)
    print("\n" + analysis.summary())

    return analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--n_agents', type=int, default=6)
    parser.add_argument('--n_rounds', type=int, default=10)
    parser.add_argument('--beam_width', type=int, default=15)
    parser.add_argument('--stream_length', type=int, default=200)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    if args.fast:
        n_runs, n_rounds, beam_width, stream = 5, 6, 10, 150
    else:
        n_runs, n_rounds, beam_width, stream = args.n_runs, args.n_rounds, args.beam_width, args.stream_length

    analysis = run_experiment(n_runs=n_runs, n_agents=args.n_agents,
                              n_rounds=n_rounds, beam_width=beam_width,
                              stream_length=stream)
    print("\n✅ Communication experiment complete")


if __name__ == '__main__':
    main()
