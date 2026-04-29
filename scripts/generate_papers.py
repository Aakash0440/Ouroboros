"""
Generate both papers from experimental results.

Usage: python scripts/generate_papers.py
"""

import sys; sys.path.insert(0, '.')
from pathlib import Path
from ouroboros.papers.paper_writer import ExperimentNumbers, generate_paper1, generate_paper2

def main():
    # Load real numbers if available
    benchmark_path = "results/benchmark_results.json"
    nums = ExperimentNumbers.from_benchmark_json(benchmark_path)

    # Try to load civilization results
    civ_path = "results/civilization_result.json"
    if Path(civ_path).exists():
        import json
        civ = json.loads(Path(civ_path).read_text())
        nums.civilization_order_corr = civ.get("order_correlation", nums.civilization_order_corr)
        nums.civilization_discovery_n = civ.get("total_discoveries", nums.civilization_discovery_n)

    print("Generating papers with numbers:")
    print(f"  Compression ratio: {nums.compression_ratio_mean:.4f} ± {nums.compression_ratio_ci:.4f}")
    print(f"  Convergence rounds: {nums.convergence_rounds_mean:.2f} ± {nums.convergence_rounds_ci:.2f}")
    print(f"  CRT success: {nums.crt_success_rate:.3f} ± {nums.crt_success_ci:.3f}")
    print(f"  Physics Hooke corr: {nums.physics_hookes_law_corr:.3f}")
    print(f"  Civilization corr: {nums.civilization_order_corr:.3f}")

    Path("results").mkdir(exist_ok=True)
    p1 = generate_paper1(nums, "results")
    p2 = generate_paper2(nums, "results")

    print(f"\n✅ Paper 1 written to: {p1}")
    print(f"✅ Paper 2 written to: {p2}")
    print("\nTo compile:")
    print("  cd results && pdflatex paper1_mathematical_emergence.tex")
    print("  cd results && pdflatex paper2_proof_market.tex")

if __name__ == '__main__':
    main()