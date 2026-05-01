"""Run the 100-session knowledge accumulation experiment."""
import sys; sys.path.insert(0, '.')
import argparse
from ouroboros.knowledge.experiment100 import KnowledgeAccumulationExperiment100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=30)
    parser.add_argument('--fast', action='store_true')
    args = parser.parse_args()

    n = 15 if args.fast else args.sessions
    exp = KnowledgeAccumulationExperiment100(
        n_sessions=n,
        stream_length=150 if args.fast else 200,
        beam_width=10 if args.fast else 12,
        n_iterations=4 if args.fast else 6,
        verbose=True,
        report_every=5 if args.fast else 10,
    )
    result = exp.run()
    print(f"\nSaved to results/experiment_100_result.json")

if __name__ == '__main__':
    main()