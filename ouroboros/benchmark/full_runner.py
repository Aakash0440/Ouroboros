"""
FullBenchmarkRunner — Runs the complete benchmark at publication quality.

Changes from BenchmarkRunner (Day 28):
  - n_seeds = 10 (not 5)
  - fast_mode = False (full stream lengths, more rounds)
  - Saves intermediate results every 2 seeds (resumable if interrupted)
  - Compares against previous fast-mode results if available
  - Validates CI width: CI/mean < 0.15 for publication acceptance
  - Generates the complete updated LaTeX tables for both papers

Stream lengths in full mode:
  compression_landmark:    stream=2000 (vs 300 fast)
  moduli_generalization:   stream=2000
  convergence_rounds:      stream=1000, n_rounds=20 (vs 10 fast)
  crt_accuracy:            stream=1500
  ood_generalization:      stream=1000
  self_improvement_gain:   stream=1000
"""

from __future__ import annotations
import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from ouroboros.benchmark.runner import BenchmarkRunner, ExperimentResult


@dataclass
class PublicationCheck:
    """Checks whether an experiment result is publication-quality."""
    experiment_name: str
    mean: float
    ci_95: float
    n: int
    relative_ci: float       # ci_95 / mean — should be < 0.15
    passes_publication: bool
    recommendation: str

    def description(self) -> str:
        status = "✅ PUBLISHABLE" if self.passes_publication else "❌ NEEDS MORE SEEDS"
        return (
            f"{status} {self.experiment_name}: "
            f"{self.mean:.4f} ± {self.ci_95:.4f} "
            f"(relative CI: {self.relative_ci*100:.1f}%, n={self.n})"
        )


@dataclass
class BenchmarkComparison:
    """Compares fast-mode and full-mode benchmark results."""
    fast_results: Dict[str, Dict]
    full_results: Dict[str, Dict]

    def print_comparison(self) -> None:
        print("\nFAST-MODE vs FULL-MODE COMPARISON")
        print("=" * 70)
        print(f"{'Experiment':<35} {'Fast (n=5)':<18} {'Full (n=10)':<18} {'Δ CI'}")
        print("-" * 70)

        for key in self.full_results:
            if key not in self.fast_results:
                continue
            fast = self.fast_results[key]
            full = self.full_results[key]

            fast_str = f"{fast.get('mean', 0):.4f}±{fast.get('ci_95', 0):.4f}"
            full_str = f"{full.get('mean', 0):.4f}±{full.get('ci_95', 0):.4f}"
            fast_ci = fast.get('ci_95', 0)
            full_ci = full.get('ci_95', 0)
            delta = fast_ci - full_ci
            direction = "↓ better" if delta > 0 else "↑ worse"
            print(f"  {key[:33]:<35} {fast_str:<18} {full_str:<18} {delta:+.4f} {direction}")


class ConfidenceIntervalValidator:
    """Validates CI width for publication acceptability."""

    def __init__(self, relative_ci_threshold: float = 0.15):
        self.threshold = relative_ci_threshold

    def validate(self, result: ExperimentResult) -> PublicationCheck:
        """Check if CI is tight enough for publication."""
        ci = result.ci_95() if callable(getattr(result, 'ci_95', None)) else result.ci_95
        mean = result.mean
        rel_ci = ci / max(abs(mean), 1e-10)
        passes = rel_ci < self.threshold

        if passes:
            rec = "CI acceptable for publication"
        elif rel_ci < 0.30:
            rec = "CI borderline — consider n=15 seeds"
        else:
            rec = "CI too wide — run n=20 seeds"

        name = result.name if hasattr(result, 'name') else result.experiment_name

        return PublicationCheck(
            experiment_name=name,
            mean=mean,
            ci_95=ci,
            n=result.n,
            relative_ci=rel_ci,
            passes_publication=passes,
            recommendation=rec,
        )

    def validate_all(self, results: Dict[str, Any]) -> List[PublicationCheck]:
        checks = []
        for key, r in results.items():
            if hasattr(r, 'mean'):
                checks.append(self.validate(r))
            elif isinstance(r, list):
                for item in r:
                    if hasattr(item, 'mean'):
                        checks.append(self.validate(item))
        return checks


class FullBenchmarkRunner:
    """
    Publication-quality benchmark runner.

    Usage:
        runner = FullBenchmarkRunner(n_seeds=10, output_dir="results")
        results = runner.run_all()
        runner.validate_and_report(results)

    Speed vs fast mode
    ──────────────────
    Fast mode: n_seeds=5, short streams.  Full mode: n_seeds=10, long streams.

        Experiment              Fast stream   Full stream   Stream ×   Seeds ×   Total ×
        compression_landmark        300          2000          6.7×       2×      ~13×
        moduli_generalization       300          2000          6.7×       2×      ~13×
        convergence_rounds      10 rounds    20 rounds         2×        2×       ~4×
        crt_accuracy                500          1500          3.0×       2×       ~6×
        ood_generalization          300          1000          3.3×       2×       ~7×
        self_improvement_gain       300          1000          3.3×       2×       ~7×

    Weighted average across all 6 experiments: ~8–10× slower than fast mode.
    Typical wall-clock times:
        Fast mode  →  ~3–6 minutes
        Full mode  →  ~30–60 minutes
    """

    def __init__(
        self,
        n_seeds: int = 10,
        output_dir: str = "results",
        resume_from: Optional[str] = None,
        verbose: bool = True,
    ):
        self.n_seeds = n_seeds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        self.resume_from = resume_from

        self._runner = BenchmarkRunner(
            n_seeds=n_seeds,
            fast_mode=False,
            output_dir=output_dir,
            verbose=verbose,
        )
        self._validator = ConfidenceIntervalValidator(relative_ci_threshold=0.15)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def run_all(self, save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Run all 6 experiments at publication quality.
        Saves a checkpoint JSON after each experiment so a crash is resumable.
        """
        results: Dict[str, Any] = {}

        if self.resume_from and Path(self.resume_from).exists():
            print(f"Resuming from {self.resume_from} …")
            results = self._load_checkpoint(self.resume_from)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"FULL BENCHMARK — PUBLICATION MODE")
            print(f"Seeds: {self.n_seeds}, Fast mode: OFF")
            print(
                "Stream lengths: compression=2000, "
                "convergence=1000, CRT=1500"
            )
            print(f"{'='*60}\n")

        start = time.time()

        # [1] Compression landmark
        if "compression_landmark" not in results:
            print("[1/6] Compression Landmark (n=2000 stream, 10 seeds)...")
            r1 = self._runner.run_compression_landmark()
            results["compression_landmark"] = r1
            if save_intermediate:
                self._save_intermediate(results, "partial_1_compression.json")
            print(f"  → {r1.mean:.4f} ± {r1.ci_95():.4f}")
        else:
            print("[1/6] Compression Landmark … (loaded from checkpoint)")

        # [2] Moduli generalization
        if not any(k.startswith("moduli") for k in results):
            print("[2/6] Moduli Generalization (4 moduli × 10 seeds)...")
            r2_list = self._runner.run_moduli_generalization()
            for r2 in r2_list:
                results[r2.experiment_name] = r2
            if save_intermediate:
                self._save_intermediate(results, "partial_2_moduli.json")
        else:
            print("[2/6] Moduli Generalization … (loaded from checkpoint)")

        # [3] Convergence rounds
        if "convergence_rounds" not in results:
            print("[3/6] Convergence Rounds (n=20 rounds × 10 seeds)...")
            r3 = self._runner.run_convergence_rounds()
            results["convergence_rounds"] = r3
            if save_intermediate:
                self._save_intermediate(results, "partial_3_convergence.json")
            print(f"  → {r3.mean:.2f} ± {r3.ci_95():.2f} rounds")
        else:
            print("[3/6] Convergence Rounds … (loaded from checkpoint)")

        # [4] CRT accuracy
        if "crt_accuracy" not in results:
            print("[4/6] CRT Accuracy (n=1500 stream × 10 seeds)...")
            r4 = self._runner.run_crt_accuracy()
            results["crt_accuracy"] = r4
            if save_intermediate:
                self._save_intermediate(results, "partial_4_crt.json")
            print(f"  → {r4.mean:.3f} ± {r4.ci_95():.3f}")
        else:
            print("[4/6] CRT Accuracy … (loaded from checkpoint)")

        # [5] OOD generalization
        if "ood_generalization" not in results:
            print("[5/6] OOD Generalization (10 seeds)...")
            r5 = self._runner.run_ood_generalization()
            results["ood_generalization"] = r5
            if save_intermediate:
                self._save_intermediate(results, "partial_5_ood.json")
        else:
            print("[5/6] OOD Generalization … (loaded from checkpoint)")

        # [6] Self-improvement gain
        if "self_improvement_gain" not in results:
            print("[6/6] Self-Improvement Gain (10 seeds)...")
            r6 = self._runner.run_self_improvement_gain()
            results["self_improvement_gain"] = r6
        else:
            print("[6/6] Self-Improvement Gain … (loaded from checkpoint)")

        elapsed = time.time() - start
        print(f"\nFull benchmark complete in {elapsed/60:.1f} minutes")

        self._save_intermediate(results, "full_benchmark_results.json")
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_intermediate(self, results: Dict, filename: str) -> None:
        serializable = {}
        for key, r in results.items():
            if hasattr(r, 'mean'):
                ci = r.ci_95() if callable(getattr(r, 'ci_95', None)) else r.ci_95
                serializable[key] = {
                    "mean": r.mean,
                    "std": getattr(r, 'std', 0.0),
                    "ci_95": ci,
                    "n": r.n,
                    "values": list(getattr(r, 'values', [])),
                }
        (self.output_dir / filename).write_text(
            json.dumps(serializable, indent=2)
        )

    def _load_checkpoint(self, path: str) -> Dict:
        return json.loads(Path(path).read_text())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_and_report(
        self,
        results: Dict[str, Any],
        comparison_path: Optional[str] = None,
    ) -> List[PublicationCheck]:
        checks = self._validator.validate_all(results)

        print(f"\nPUBLICATION READINESS REPORT")
        print(f"{'='*60}")
        n_pass = sum(1 for c in checks if c.passes_publication)
        print(f"Results: {n_pass}/{len(checks)} pass publication CI threshold")
        print()
        for check in checks:
            print(f"  {check.description()}")

        if comparison_path and Path(comparison_path).exists():
            fast = json.loads(Path(comparison_path).read_text())
            full_serializable = {}
            for key, r in results.items():
                if hasattr(r, 'mean'):
                    ci = r.ci_95() if callable(getattr(r, 'ci_95', None)) else r.ci_95
                    full_serializable[key] = {"mean": r.mean, "ci_95": ci, "n": r.n}
            BenchmarkComparison(fast, full_serializable).print_comparison()

        return checks

    # ------------------------------------------------------------------
    # LaTeX
    # ------------------------------------------------------------------

    def generate_updated_latex_tables(self, results: Dict[str, Any]) -> str:
        from ouroboros.benchmark.report import (
            generate_latex_table_1,
            generate_latex_table_2,
            generate_latex_table_3,
        )
        serializable = {}
        for key, r in results.items():
            if hasattr(r, 'mean'):
                ci = r.ci_95() if callable(getattr(r, 'ci_95', None)) else r.ci_95
                serializable[key] = {
                    "mean": r.mean,
                    "std": getattr(r, 'std', 0.0),
                    "ci_95": ci,
                    "n": r.n,
                    "values": list(getattr(r, 'values', [])),
                    "latex": r.latex_str() if hasattr(r, 'latex_str') else
                             f"${r.mean:.4f} \\pm {ci:.4f}$",
                }

        tables = "\n\n".join([
            "% FULL-MODE RESULTS (n=10 seeds, publication quality)",
            "% Generated by FullBenchmarkRunner",
            f"% Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            generate_latex_table_1(serializable),
            generate_latex_table_2(serializable),
            generate_latex_table_3(serializable),
        ])

        output_path = self.output_dir / "full_latex_tables.tex"
        output_path.write_text(tables)
        print(f"\nUpdated LaTeX tables saved to: {output_path}")
        return tables