# dataset_analysis.py
"""
Analyzes results from the 200-dataset study.
Produces the numbers you need for the paper:
  - What fraction of real scientific datasets have discoverable structure?
  - Which domains have the most structure?
  - What compression ratios are typical?
  - Which mathematical families are most common across domains?
"""

import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict


def analyze_results(results_file: str = "results/dataset_study/all_results.json"):
    with open(results_file) as f:
        results = json.load(f)

    n_total = len(results)
    print(f"\nDATASET STUDY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total datasets analyzed: {n_total}")

    # Filter out synthetic benchmarks for main analysis
    real = [r for r in results if "synth" not in r["dataset_id"]]
    synthetic = [r for r in results if "synth" in r["dataset_id"]]

    print(f"Real datasets: {len(real)}")
    print(f"Synthetic benchmarks: {len(synthetic)}")

    # ── Key claim 1: What fraction have discoverable structure? ──────────────
    structure_found = [r for r in real
                   if r["compression_ratio"] < 0.5]

    print(f"\n{'─'*60}")
    print(f"STRUCTURE DISCOVERY")
    print(f"{'─'*60}")
    print(f"Found structure (ratio < 0.5, beats baseline): "
          f"{len(structure_found)}/{len(real)} = "
          f"{len(structure_found)/max(len(real),1)*100:.1f}%")

    strong_structure = [r for r in real if r["compression_ratio"] < 0.1]
    print(f"Strong structure (ratio < 0.1): "
          f"{len(strong_structure)}/{len(real)} = "
          f"{len(strong_structure)/max(len(real),1)*100:.1f}%")

    # ── Key claim 2: By domain ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"STRUCTURE BY DOMAIN")
    print(f"{'─'*60}")
    by_domain = defaultdict(list)
    for r in real:
        by_domain[r["domain"]].append(r)

    domain_stats = []
    for domain, domain_results in sorted(by_domain.items()):
        n = len(domain_results)
        n_structure = sum(1 for r in domain_results
                  if r["compression_ratio"] < 0.5)
        avg_ratio = sum(r["compression_ratio"] for r in domain_results
                        if math.isfinite(r["compression_ratio"])) / max(n, 1)
        domain_stats.append((domain, n, n_structure, avg_ratio))

    domain_stats.sort(key=lambda x: -x[2]/max(x[1],1))
    for domain, n, n_structure, avg_ratio in domain_stats:
        pct = n_structure/max(n,1)*100
        print(f"  {domain:<20s}: {n_structure}/{n} ({pct:.0f}%) "
              f"avg_ratio={avg_ratio:.3f}")

    # ── Key claim 3: Mathematical families ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"MATHEMATICAL FAMILIES DISCOVERED")
    print(f"{'─'*60}")
    family_counts = Counter(r["math_family"] for r in real
                        if r.get("expression_str") is not None)
    for family, count in family_counts.most_common():
        pct = count / max(len([r for r in real if r.get("expression_str") is not None]), 1) * 100
        bar = "█" * int(pct / 3)
        print(f"  {family:<20s}: {count:3d} ({pct:.1f}%) {bar}")

    # ── Key claim 4: Compression ratio distribution ───────────────────────────
    print(f"\n{'─'*60}")
    print(f"COMPRESSION RATIO DISTRIBUTION")
    print(f"{'─'*60}")
    ratios = [r["compression_ratio"] for r in real
              if r.get("expression_str") is not None and math.isfinite(r["compression_ratio"])]

    if ratios:
        brackets = [
            (0.0, 0.01,  "< 0.01  (excellent)"),
            (0.01, 0.1,  "0.01–0.1 (strong)"),
            (0.1, 0.3,   "0.1–0.3  (moderate)"),
            (0.3, 0.5,   "0.3–0.5  (weak)"),
            (0.5, 1.0,   "0.5–1.0  (marginal)"),
            (1.0, 999.9, "> 1.0    (worse than baseline)"),
        ]
        for lo, hi, label in brackets:
            count = sum(1 for r in ratios if lo <= r < hi)
            pct = count / len(ratios) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

    # ── Key claim 5: Novelty ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"NOVELTY DETECTION")
    print(f"{'─'*60}")
    novel_findings = [r for r in real
                      if r["novelty_score"] > 0.5 and r["beats_trivial_baseline"]]
    print(f"Potentially novel (novelty > 0.5, beats baseline): {len(novel_findings)}")
    for r in novel_findings[:5]:
        print(f"  {r['name'][:40]}: score={r['novelty_score']:.3f}, "
              f"ratio={r['compression_ratio']:.4f}, "
              f"expr={str(r['expression_str'])[:40]}")

    # ── Synthetic validation ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"SYNTHETIC BENCHMARK VALIDATION")
    print(f"{'─'*60}")
    for r in synthetic:
        expected_max = 0.05 if "random" not in r["dataset_id"] else 1.5
        passed = r["compression_ratio"] < expected_max
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} {r['name'][:40]}: ratio={r['compression_ratio']:.4f}")

    # ── Paper-ready summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"PAPER-READY NUMBERS")
    print(f"{'='*60}")
    n_real = len(real)
    n_success = len(structure_found)
    n_strong = len(strong_structure)

    print(f"We analyzed {n_real} real scientific time series across "
          f"{len(by_domain)} domains.")
    print(f"OUROBOROS found compressible structure (ratio < 0.5) in "
          f"{n_success/max(n_real,1)*100:.1f}% of datasets.")
    print(f"Strong structure (ratio < 0.1) was found in "
          f"{n_strong/max(n_real,1)*100:.1f}% of datasets.")
    print(f"The most structure-rich domain was: "
          f"{domain_stats[0][0] if domain_stats else 'N/A'}")
    print(f"Potentially novel findings: {len(novel_findings)}")

    # Save analysis
    analysis = {
        "n_total": n_total,
        "n_real": n_real,
        "n_structure_found": n_success,
        "n_strong_structure": n_strong,
        "structure_rate": n_success / max(n_real, 1),
        "strong_structure_rate": n_strong / max(n_real, 1),
        "by_domain": {
            d: {"n": n, "n_structure": ns, "avg_ratio": ar}
            for d, n, ns, ar in domain_stats
        },
        "family_distribution": dict(family_counts.most_common()),
        "n_potentially_novel": len(novel_findings),
    }

    with open("results/dataset_study/analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"\nAnalysis saved to results/dataset_study/analysis.json")
    return analysis


if __name__ == "__main__":
    analyze_results()