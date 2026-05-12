"""
Feynman benchmark: run OUROBOROS on ALL equations in FEYNMAN_ALL.
Reports recovery rate, MDL ratios, and expressions found.

Usage:
    python feynman_bench.py [--filter SUBSTR] [--workers N] [--out results.json]

Options:
    --filter SUBSTR   Only run equations whose name contains SUBSTR (e.g. "III")
    --workers N       Parallel workers (default: 1, set >1 with care)
    --out FILE        Write JSON results to FILE (default: feynman_results.json)
    --no-structural   Skip the structural benchmarks (linear_1, quad_1, etc.)
    --only-structural Run only the structural benchmarks
    --time-budget S   Per-equation time budget in seconds (default: 45)
"""
import math, time, sys, json, argparse, traceback
from feynman_data.equations import FEYNMAN_ALL
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.compression.mdl_engine import MDLEngine


# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="OUROBOROS × Feynman benchmark")
parser.add_argument("--filter",          default="",    help="Substring filter on equation name")
parser.add_argument("--workers",  type=int, default=1, help="Parallel workers")
parser.add_argument("--out",             default="feynman_results.json")
parser.add_argument("--no-structural",   action="store_true")
parser.add_argument("--only-structural", action="store_true")
parser.add_argument("--time-budget", type=float, default=45.0)
args = parser.parse_args()

STRUCTURAL_NAMES = {
    "linear_1","linear_2","quad_1","quad_2","cubic_1",
    "log_1","sqrt_1","exp_1","inv_1","inv2_1","sin_1","cos_1",
}

def should_run(name: str) -> bool:
    if args.filter and args.filter not in name:
        return False
    is_struct = name in STRUCTURAL_NAMES
    if args.no_structural and is_struct:
        return False
    if args.only_structural and not is_struct:
        return False
    return True

equations = [(n, e, f, np, a) for (n, e, f, np, a) in FEYNMAN_ALL if should_run(n)]
print(f"Running {len(equations)} equations  (time budget {args.time_budget}s each)\n")


# ── Solver setup ───────────────────────────────────────────────────────────
router = HierarchicalSearchRouter(RouterConfig(
    beam_width=100,
    max_depth=7,        # was 5 — gives room for nested expressions
    const_range=100,    # was 50 — wider constant search
    max_lag=5,          # was 10 — fewer lag candidates = more room for real expressions
    n_iterations=50,
    time_budget_seconds=args.time_budget,
))
mdl = MDLEngine()


# ── Benchmark loop ─────────────────────────────────────────────────────────
results = []
col_w = max(len(n) for n, *_ in equations) + 2

header = f"{'#':<4} {'Name':<{col_w}} {'Ratio':>6}  {'Pass':>5}  Expression"
sep    = "-" * (len(header) + 10)
print(header)
print(sep)

for i, (name, true_expr, fn, n_points, alpha) in enumerate(equations):
    row = {"name": name, "true": true_expr, "passed": False, "ratio": 999, "time": 0.0}
    try:
        # Generate & clip sequence
        seq = []
        for t in range(n_points):
            try:
                v = fn(t)
            except Exception:
                v = 0
            seq.append(max(-10_000, min(10_000, v)))

        alpha = max(alpha, max(seq) + 2)

        # Baseline: constant (mean) model
        mean_val = int(round(sum(seq) / len(seq)))
        baseline = mdl.compute(
            [mean_val] * len(seq), seq,
            node_count=1, constant_count=1, alphabet_size=alpha,
        )

        # Search
        t0 = time.time()
        result = router.search(seq, alphabet_size=alpha)
        elapsed = time.time() - t0

        expr_str = result.expr.to_string() if result.expr else "None"
        ratio    = (result.mdl_cost / baseline.total_mdl_cost
                    if baseline.total_mdl_cost > 0 else 999.0)
        passed   = ratio < 0.5

        row.update({"found": expr_str, "ratio": ratio, "passed": passed, "time": elapsed})
        status = "PASS" if passed else "FAIL"
        print(f"{i+1:<4} {name:<{col_w}} {ratio:>6.3f}  {status:>5}  {expr_str[:50]}")

    except Exception as exc:
        row["error"] = str(exc)
        print(f"{i+1:<4} {name:<{col_w}}  ERROR: {exc}")
        if "--verbose" in sys.argv:
            traceback.print_exc()

    results.append(row)
    sys.stdout.flush()


# ── Summary ────────────────────────────────────────────────────────────────
passed_list   = [r for r in results if r["passed"]]
failed_list   = [r for r in results if not r["passed"]]
total         = len(results)
n_passed      = len(passed_list)
times         = [r["time"] for r in results]
avg_t         = sum(times) / total if total else 0
med_t         = sorted(times)[total // 2] if total else 0

print()
print("=" * (len(header) + 10))
print(f"Recovery rate : {n_passed}/{total} = {n_passed/total*100:.1f}%")
print(f"Avg time/eq   : {avg_t:.1f}s   Median: {med_t:.1f}s")

if failed_list:
    print()
    print("Failed equations:")
    for r in failed_list:
        ratio_s = f"{r['ratio']:.3f}" if r["ratio"] < 900 else " err "
        found_s = r.get("found", r.get("error", "?"))[:55]
        print(f"  {r['name']:<{col_w}} ratio={ratio_s}  found={found_s}")

# ── Per-category breakdown ─────────────────────────────────────────────────
from collections import defaultdict
cats = defaultdict(lambda: {"pass": 0, "total": 0})
for r in results:
    cat = r["name"].split(".")[0] if "." in r["name"] else "structural"
    cats[cat]["total"] += 1
    if r["passed"]:
        cats[cat]["pass"] += 1

print()
print("Per-category:")
for cat, d in sorted(cats.items()):
    pct = d["pass"] / d["total"] * 100
    bar = "█" * d["pass"] + "░" * (d["total"] - d["pass"])
    print(f"  {cat:<12} {d['pass']:>3}/{d['total']:<3}  {pct:5.1f}%  {bar}")

# ── Write JSON ─────────────────────────────────────────────────────────────
with open(args.out, "w") as f:
    json.dump({
        "summary": {
            "total": total, "passed": n_passed,
            "rate": n_passed / total if total else 0,
            "avg_time": avg_t, "median_time": med_t,
        },
        "results": results,
    }, f, indent=2)
print(f"\nResults written to {args.out}")