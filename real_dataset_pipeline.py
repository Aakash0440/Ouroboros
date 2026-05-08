"""
OUROBOROS Real Dataset Pipeline
Runs OUROBOROS on all registered public datasets, logs results.
"""
from __future__ import annotations
import json, time, math, sys
from pathlib import Path

from ouroboros.data.dataset_registry import DATASETS
from ouroboros.data.downloader import fetch
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.compression.mdl_engine import MDLEngine

from ouroboros.novelty.novelty_checker import NoveltyChecker
CHECKER = NoveltyChecker()

router = RouterConfig(beam_width=25, n_iterations=15, time_budget_seconds=30.0)
ROUTER = HierarchicalSearchRouter(router)
MDL    = MDLEngine()

results = []
print(f"{'#':<4} {'Dataset':<24} {'Domain':<14} {'Points':>6} {'Ratio':>7} {'Pass':>5}  Expression")
print("-" * 90)

for i, ds in enumerate(DATASETS):
    seq = fetch(ds, max_points=150)
    if seq is None:
        print(f"{i+1:<4} {ds['name']:<24} {'DOWNLOAD_FAIL':<14}")
        results.append({"name": ds["name"], "status": "download_fail"})
        continue

    alpha = max(seq) + 2
    # Baseline: mean prediction
    mean_val = int(round(sum(seq) / len(seq)))
    b = MDL.compute([mean_val]*len(seq), seq, node_count=1, constant_count=1, alphabet_size=alpha)

    t0 = time.time()
    r  = ROUTER.search(seq, alphabet_size=alpha, verbose=False)
    elapsed = time.time() - t0

    expr_str = r.expr.to_string() if r.expr else "None"
    ratio    = r.mdl_cost / b.total_mdl_cost if b.total_mdl_cost > 0 else 999
    passed   = ratio < 0.90

    record = {
        "name":    ds["name"],
        "domain":  ds["domain"],
        "points":  len(seq),
        "ratio":   round(ratio, 3),
        "passed":  passed,
        "expr":    expr_str,
        "time":    round(elapsed, 1),
        "status":  "ok",
    }
    results.append(record)

    # OEIS novelty check for passing results
    novelty = None
    if passed and r.expr is not None and expr_str not in ("obs[t-1]", "t", "None"):
        try:
            preds = [int(round(r.expr.evaluate(t, seq[:t]))) for t in range(min(12, len(seq)))]
            novelty = CHECKER.score(r.expr, seq, preds)
            record["novelty_score"] = novelty["novelty_score"]
            record["oeis_id"]       = novelty["oeis_id"]
            record["oeis_name"]     = novelty["oeis_name"]
        except Exception:
            pass

    status = "PASS" if passed else "FAIL"
    novelty_str = f"  novelty={novelty['novelty_score']:.2f} {novelty.get('oeis_id') or 'NEW'}" if novelty else ""
    print(f"{i+1:<4} {ds['name']:<24} {ds['domain']:<14} {len(seq):>6} {ratio:>7.3f} {status:>5}  {expr_str[:30]}{novelty_str}")
    sys.stdout.flush()

# Summary
ok      = [r for r in results if r.get("status") == "ok"]
passed  = [r for r in ok if r.get("passed")]
failed  = [r for r in ok if not r.get("passed")]

print()
print("=" * 90)
print(f"Datasets attempted : {len(DATASETS)}")
print(f"Successfully fetched: {len(ok)}")
print(f"Structure found    : {len(passed)}/{len(ok)} = {len(passed)/max(len(ok),1)*100:.1f}%")
print(f"Avg time/dataset   : {sum(r['time'] for r in ok)/max(len(ok),1):.1f}s")
print()
print("Top compressions:")
for r in sorted(ok, key=lambda x: x["ratio"])[:5]:
    print(f"  {r['name']:<24} ratio={r['ratio']:.3f}  {r['expr'][:50]}")

# Save results
out = Path("results")
out.mkdir(exist_ok=True)
with open(out / "real_dataset_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → results/real_dataset_results.json")