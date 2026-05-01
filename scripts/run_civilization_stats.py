"""
Run the civilization simulation with proper bootstrap statistics.

Fast:  5 runs × 16 agents × 30 rounds ≈ 15 minutes
Full: 10 runs × 64 agents × 200 rounds ≈ 4 hours
"""

import sys; sys.path.insert(0, '.')
import argparse, json
from pathlib import Path
from ouroboros.civilization.statistics import CivilizationStatisticalAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    if args.full:
        analyzer = CivilizationStatisticalAnalyzer(
            n_runs=10, n_agents=64, n_rounds=200,
            stream_length=300, beam_width=15, n_iterations=8, n_bootstrap=2000,
        )
    else:
        analyzer = CivilizationStatisticalAnalyzer(
            n_runs=5, n_agents=16, n_rounds=30,
            stream_length=150, beam_width=8, n_iterations=4, n_bootstrap=1000,
        )

    report = analyzer.run_full_analysis(verbose=True)

    Path("results").mkdir(exist_ok=True)
    data = {
        "observed_rho": report.spearman_result.observed_rho,
        "ci_lower": report.spearman_result.ci_lower,
        "ci_upper": report.spearman_result.ci_upper,
        "n_runs": report.n_runs,
        "n_bootstrap": report.spearman_result.n_bootstrap,
        "discovery_consistency": report.discovery_consistency,
        "is_significant": report.spearman_result.is_significant,
        "latex": report.spearman_result.latex_str(),
    }
    Path("results/civilization_stats.json").write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to results/civilization_stats.json")
    print(f"LaTeX: {report.spearman_result.latex_str()}")

if __name__ == '__main__':
    main()