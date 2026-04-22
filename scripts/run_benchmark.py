"""
Run the full OUROBOROS benchmark suite.

Usage:
  python scripts/run_benchmark.py --fast          # ~15 min, n_seeds=5
  python scripts/run_benchmark.py --full          # ~90 min, n_seeds=10
  python scripts/run_benchmark.py --quick         # ~3 min, n_seeds=3
"""

import sys; sys.path.insert(0, '.')
import argparse
import json
from pathlib import Path
from ouroboros.benchmark.runner import BenchmarkRunner
from ouroboros.benchmark.report import (
    generate_latex_table_1, generate_latex_table_2, generate_latex_table_3,
    generate_figures, PaperNumbersReport,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    if args.quick:
        n_seeds, fast = 3, True
    elif args.full:
        n_seeds, fast = 10, False
    else:  # --fast is default
        n_seeds, fast = 5, True

    runner = BenchmarkRunner(n_seeds=n_seeds, fast_mode=fast)
    results = runner.run_all()

    # Save raw results
    output_path = runner.save_results(results)
    print(f"\nRaw results saved to: {output_path}")

    # Load for report generation
    raw = {k: (v.__dict__ if hasattr(v, '__dict__') else v) for k, v in results.items()}
    # Convert ExperimentResult objects to dicts
    results_dict = {}
    for k, v in results.items():
        if hasattr(v, 'mean'):
            results_dict[k] = {
                'mean': v.mean, 'std': v.std, 'median': v.median,
                'min': v.min_val, 'max': v.max_val,
                'ci_95': v.ci_95(), 'n': v.n, 'values': v.values,
                'latex': v.latex_str(),
            }

    # Generate LaTeX tables
    output_dir = Path("results")
    tables_path = output_dir / "latex_tables.tex"
    tables = "\n\n".join([
        generate_latex_table_1(results_dict),
        generate_latex_table_2(results_dict),
        generate_latex_table_3(results_dict),
    ])
    tables_path.write_text(tables)
    print(f"LaTeX tables saved to: {tables_path}")

    # Generate figures
    figs = generate_figures(results_dict, output_dir)
    print(f"Figures saved: {[str(f) for f in figs]}")

    # Generate paper numbers report
    report = PaperNumbersReport(results_dict).generate()
    report_path = output_dir / "paper_numbers.md"
    report_path.write_text(report)
    print(f"Paper numbers report saved to: {report_path}")
    print("\n" + report[:800])


if __name__ == '__main__':
    main()