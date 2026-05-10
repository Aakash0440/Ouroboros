# dataset_runner.py
"""
Runs OUROBOROS on every downloaded dataset and records results.
Handles failures gracefully, resumes from checkpoints.

Import mapping (actual modules used):
  ouroboros.search.env_classifier      → EnvironmentClassifier (replaces SmartPreprocessor)
  ouroboros.search.hierarchical_router → HierarchicalSearchRouter, RouterConfig
  ouroboros.compression.mdl_engine     → MDLEngine (baseline + scoring)
  ouroboros.novelty.registry           → EmbeddingRegistry (wrapped in try/except)
  ouroboros.causal.do_calculus         → DoCalculusEngine (wrapped in try/except)
"""

import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetResult:
    """OUROBOROS result for one dataset."""
    dataset_id: str
    name: str
    domain: str
    source: str
    n_observations: int
    unit: str

    # Classification
    series_type: str = "UNKNOWN"
    detected_period: Optional[int] = None
    trend_slope: Optional[float] = None

    # Discovery
    expression_str: Optional[str] = None
    mdl_cost: float = float('inf')
    compression_ratio: float = 1.0
    math_family: str = "UNKNOWN"
    discovery_success: bool = False
    beats_trivial_baseline: bool = False
    baseline_mdl: float = float('inf')
    baseline_name: str = "none"

    # Novelty
    novelty_score: float = 0.0
    novelty_category: str = "unknown"

    # Causal
    causal_edges_found: int = 0

    # Meta
    runtime_seconds: float = 0.0
    error_message: str = ""


def _quantize(values: List[float], target_alpha: int = 20) -> tuple:
    """
    Map float values → integers in [0, target_alpha-1].
    Returns (int_obs, actual_alphabet_size).
    Handles constant sequences and edge cases.
    """
    min_v = min(values)
    max_v = max(values)
    range_v = max_v - min_v

    if range_v < 1e-9:
        # Constant sequence → all zeros, alphabet size 1
        return [0] * len(values), 1

    int_obs = [
        max(0, min(target_alpha - 1,
            int(round((v - min_v) / range_v * (target_alpha - 1)))))
        for v in values
    ]
    actual_alpha = max(set(int_obs)) + 1
    return int_obs, actual_alpha


def _compute_baseline_mdl(int_obs: List[int], alpha: int) -> tuple:
    """
    Compute MDL for three trivial baselines; return (best_cost, best_name).
    Baselines: repeat_last, predict_mean, predict_mode
    """
    from ouroboros.compression.mdl_engine import MDLEngine
    mdl = MDLEngine()
    n = len(int_obs)

    results = []

    # 1. Repeat-last: predict obs[t-1]
    preds_last = [int_obs[max(0, t - 1)] for t in range(n)]
    r = mdl.compute(preds_last, int_obs, node_count=1,
                    constant_count=0, alphabet_size=alpha)
    results.append((r.total_mdl_cost, "repeat_last"))

    # 2. Predict mean
    mean_val = int(round(sum(int_obs) / max(n, 1)))
    preds_mean = [mean_val] * n
    r = mdl.compute(preds_mean, int_obs, node_count=1,
                    constant_count=1, alphabet_size=alpha)
    results.append((r.total_mdl_cost, "predict_mean"))

    # 3. Predict mode
    from collections import Counter
    mode_val = Counter(int_obs).most_common(1)[0][0]
    preds_mode = [mode_val] * n
    r = mdl.compute(preds_mode, int_obs, node_count=1,
                    constant_count=1, alphabet_size=alpha)
    results.append((r.total_mdl_cost, "predict_mode"))

    return min(results, key=lambda x: x[0])


def _detect_period(values: List[float]) -> Optional[int]:
    """Simple FFT-based period detection. Returns None if no clear period."""
    try:
        from ouroboros.search.fft_period import detect_period
        return detect_period(values)
    except Exception:
        pass
    try:
        from ouroboros.search.fft_period_finder import find_period
        return find_period(values)
    except Exception:
        pass
    # Fallback: autocorrelation peak
    try:
        n = len(values)
        mean = sum(values) / n
        denom = sum((v - mean) ** 2 for v in values)
        if denom < 1e-9:
            return None
        best_lag, best_corr = None, 0.0
        for lag in range(2, min(n // 2, 50)):
            corr = sum((values[i] - mean) * (values[i - lag] - mean)
                       for i in range(lag, n)) / denom
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        return best_lag if best_corr > 0.5 else None
    except Exception:
        return None


def run_ouroboros_on_dataset(
    dataset: dict,
    beam_width: int = 15,
    n_iterations: int = 8,
    max_length: int = 200,
    target_alpha: int = 20,
) -> DatasetResult:
    """Run the full OUROBOROS pipeline on one dataset."""
    ds_id   = dataset["dataset_id"]
    values  = dataset["values"]
    name    = dataset["name"]
    domain  = dataset["domain"]

    result = DatasetResult(
        dataset_id=ds_id, name=name, domain=domain,
        source=dataset["source"], n_observations=len(values),
        unit=dataset.get("unit", ""),
    )

    if len(values) < 10:
        result.error_message = "Too few observations"
        return result

    start = time.time()

    try:
        # ── Step 1: Classify ─────────────────────────────────────────────────
        from ouroboros.search.env_classifier import EnvironmentClassifier, MathFamily

        classifier = EnvironmentClassifier()
        classification = classifier.classify(values[:max_length])

        result.series_type = classification.primary_family.name

        # Detect period if periodic
        if classification.primary_family == MathFamily.PERIODIC:
            result.detected_period = _detect_period(values[:max_length])

        # Detect trend slope (simple linear regression)
        try:
            n = min(len(values), max_length)
            xs = list(range(n))
            xm = (n - 1) / 2
            ym = sum(values[:n]) / n
            num = sum((xs[i] - xm) * (values[i] - ym) for i in range(n))
            den = sum((x - xm) ** 2 for x in xs)
            result.trend_slope = num / den if den > 1e-9 else 0.0
        except Exception:
            pass

        # Skip discovery if clearly random
        if classification.primary_family == MathFamily.RANDOM:
            result.runtime_seconds = time.time() - start
            return result

        # ── Step 2: Quantize ─────────────────────────────────────────────────
        truncated = values[:max_length]
        int_obs, alpha = _quantize(truncated, target_alpha=target_alpha)

        # ── Step 3: Baselines ────────────────────────────────────────────────
        result.baseline_mdl, result.baseline_name = _compute_baseline_mdl(int_obs, alpha)

        # Naive information content (no model)
        naive_bits = len(int_obs) * math.log2(max(alpha, 2))

        # ── Step 4: Search ───────────────────────────────────────────────────
        from ouroboros.search.hierarchical_router import (
            HierarchicalSearchRouter, RouterConfig
        )

        router = HierarchicalSearchRouter(RouterConfig(
            beam_width=beam_width,
            max_depth=5,
            n_iterations=n_iterations,
            random_seed=42,
            time_budget_seconds=30.0,
        ))
        discovery = router.search(int_obs, alphabet_size=alpha, verbose=False)

        if discovery.expr is None:
            result.error_message = "No expression found"
            result.runtime_seconds = time.time() - start
            return result

        result.expression_str = discovery.expr.to_string()
        result.mdl_cost       = (discovery.mdl_cost
                                  if math.isfinite(discovery.mdl_cost)
                                  else 9999.0)
        result.math_family    = discovery.math_family.name
        result.discovery_success = True

        # Baseline comparison (2-bit margin: only beat baseline if meaningfully better)
        result.beats_trivial_baseline = result.mdl_cost < (result.baseline_mdl - 2.0)

        # Compression ratio vs naive encoding
        result.compression_ratio = result.mdl_cost / max(naive_bits, 1.0)

        # ── Step 5: Novelty ──────────────────────────────────────────────────
        try:
            from ouroboros.novelty.registry import EmbeddingRegistry
            registry = EmbeddingRegistry()
            novelty_result = registry.query(discovery.expr)
            result.novelty_score    = novelty_result.novelty_score
            result.novelty_category = novelty_result.novelty_category
        except Exception:
            result.novelty_score = 0.5   # unknown

        # ── Step 6: Causal ───────────────────────────────────────────────────
        try:
            from ouroboros.causal.do_calculus import DoCalculusEngine
            if len(int_obs) >= 20:
                engine = DoCalculusEngine(granger_threshold=3.0, max_lag=3)
                deriv = [
                    int_obs[t] - int_obs[t - 1] if t > 0 else 0
                    for t in range(len(int_obs))
                ]
                graph = engine.discover(
                    {
                        "obs":   [float(v) for v in int_obs],
                        "deriv": [float(v) for v in deriv],
                    },
                    verbose=False,
                )
                result.causal_edges_found = graph.n_edges
        except Exception:
            pass

    except Exception as e:
        result.error_message = str(e)[:200]

    result.runtime_seconds = time.time() - start
    return result


def run_study(
    data_dir: str = "data/raw_datasets",
    results_dir: str = "results/dataset_study",
    resume: bool = True,
    max_datasets: int = None,
) -> list:
    """
    Run OUROBOROS on all datasets.
    Checkpoints every 5 datasets — safe to interrupt and restart.
    """
    data_path    = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    manifest_file = data_path / "manifest.json"
    if not manifest_file.exists():
        print("No manifest.json found. Run dataset_crawler.py first.")
        return []

    with open(manifest_file) as f:
        manifest = json.load(f)

    successful = [m for m in manifest if m["download_success"]]
    if max_datasets:
        successful = successful[:max_datasets]

    print(f"\nRunning OUROBOROS on {len(successful)} datasets")
    print(f"Results dir: {results_dir}")

    # Load checkpoint
    checkpoint_file = results_path / "checkpoint.json"
    completed_ids   = set()
    all_results     = []

    if resume and checkpoint_file.exists():
        with open(checkpoint_file) as f:
            ckpt = json.load(f)
        completed_ids = set(ckpt.get("completed", []))
        all_results   = ckpt.get("results", [])
        print(f"Resuming: {len(completed_ids)} already done")

    for i, entry in enumerate(successful):
        ds_id = entry["dataset_id"]
        if ds_id in completed_ids:
            continue

        ds_file = data_path / f"{ds_id}.json"
        if not ds_file.exists():
            continue

        with open(ds_file) as f:
            dataset = json.load(f)

        print(f"\n[{i+1}/{len(successful)}] {entry['name'][:55]}")
        print(f"  domain={entry['domain']}, n={entry['n_observations']}")

        result = run_ouroboros_on_dataset(dataset)

        status = "✓" if result.beats_trivial_baseline else "·"
        print(f"  {status} family={result.math_family}, "
              f"ratio={result.compression_ratio:.4f}, "
              f"series={result.series_type}, "
              f"time={result.runtime_seconds:.1f}s")
        if result.expression_str:
            print(f"  expr: {result.expression_str[:70]}")
        if result.error_message:
            print(f"  error: {result.error_message}")

        result_dict = {
            "dataset_id":           result.dataset_id,
            "name":                 result.name,
            "domain":               result.domain,
            "source":               result.source,
            "n_observations":       result.n_observations,
            "series_type":          result.series_type,
            "expression_str":       result.expression_str,
            "mdl_cost":             result.mdl_cost if math.isfinite(result.mdl_cost) else 9999.0,
            "compression_ratio":    result.compression_ratio,
            "math_family":          result.math_family,
            "beats_trivial_baseline": result.beats_trivial_baseline,
            "baseline_name":        result.baseline_name,
            "baseline_mdl":         result.baseline_mdl if math.isfinite(result.baseline_mdl) else 9999.0,
            "novelty_score":        result.novelty_score,
            "novelty_category":     result.novelty_category,
            "causal_edges_found":   result.causal_edges_found,
            "detected_period":      result.detected_period,
            "runtime_seconds":      result.runtime_seconds,
            "error_message":        result.error_message,
        }
        all_results.append(result_dict)
        completed_ids.add(ds_id)

        # Checkpoint every 5
        if len(completed_ids) % 5 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump({"completed": list(completed_ids),
                           "results": all_results}, f)

    # Final save
    with open(results_path / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"STUDY COMPLETE: {len(all_results)} datasets processed")
    print(f"Results: {results_dir}/all_results.json")

    return all_results