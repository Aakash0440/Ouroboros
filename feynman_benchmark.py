"""
Feynman benchmark: run OUROBOROS on all single-variable equations.
Reports recovery rate, MDL ratios, and expressions found.
"""
import math, time, sys
from feynman_data.equations import FEYNMAN_1VAR
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.compression.mdl_engine import MDLEngine

router = HierarchicalSearchRouter(RouterConfig(
    beam_width=35,
    n_iterations=20,
    time_budget_seconds=45.0,
))
mdl = MDLEngine()

results = []
print(f"{'#':<4} {'Name':<14} {'Ratio':>6}  {'Pass':>5}  Expression")
print("-" * 75)

for i, (name, true_expr, fn, n_points, alpha) in enumerate(FEYNMAN_1VAR):
    try:
        seq = [fn(t) for t in range(n_points)]
        # clip any infinities
        seq = [max(-10000, min(10000, v)) for v in seq]
        alpha = max(alpha, max(seq) + 2)

        mean_val = int(round(sum(seq) / len(seq)))
        b = mdl.compute([mean_val]*len(seq), seq,
                        node_count=1, constant_count=1, alphabet_size=alpha)

        t0 = time.time()
        r  = router.search(seq, alphabet_size=alpha)
        elapsed = time.time() - t0

        expr_str = r.expr.to_string() if r.expr else 'None'
        ratio    = r.mdl_cost / b.total_mdl_cost if b.total_mdl_cost > 0 else 999
        passed   = ratio < 0.5

        results.append({
            'name': name, 'true': true_expr, 'found': expr_str,
            'ratio': ratio, 'passed': passed, 'time': elapsed,
        })
        status = 'PASS' if passed else 'FAIL'
        print(f"{i+1:<4} {name:<14} {ratio:>6.3f}  {status:>5}  {expr_str[:40]}")
        sys.stdout.flush()

    except Exception as e:
        print(f"{i+1:<4} {name:<14}  ERROR: {e}")
        results.append({'name': name, 'passed': False, 'ratio': 999, 'time': 0})

passed  = sum(1 for r in results if r['passed'])
total   = len(results)
avg_t   = sum(r['time'] for r in results) / total

print()
print("=" * 75)
print(f"Recovery rate : {passed}/{total} = {passed/total*100:.1f}%")
print(f"Avg time/eq   : {avg_t:.1f}s")
print()
print("Failed equations:")
for r in results:
    if not r['passed']:
        print(f"  {r['name']:<14} ratio={r['ratio']:.3f}  found={r.get('found','?')[:50]}")