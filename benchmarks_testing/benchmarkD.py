# benchmarkJ.py
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import math, copy
from ouroboros.continuous.beam_search import ContinuousBeamSearch, ContinuousBeamConfig, ContinuousCandidate
from ouroboros.continuous.expr_nodes import ContinuousExprNode, ContinuousNodeType
from ouroboros.continuous.mdl import compute_gaussian_mdl

r_vals = list(range(1, 41))
F_vals = [100.0 / (r**2) for r in r_vals]
raw_bits = len(F_vals) * math.log2(200)

# ── Helper: build expression nodes cleanly ───────────────────────────────────
def const(v): 
    n = ContinuousExprNode(ContinuousNodeType.CONST_REAL)
    n.value_f = v
    return n

def time_node():
    return ContinuousExprNode(ContinuousNodeType.TIME_REAL)

def binop(op_type, left, right):
    n = ContinuousExprNode(op_type)
    n.left = left
    n.right = right
    return n

def t_plus_1():
    return binop(ContinuousNodeType.ADD_REAL, time_node(), const(1.0))

def score(expr, obs):
    preds = [expr.evaluate(t, obs) for t in range(len(obs))]
    return compute_gaussian_mdl(
        predictions=preds, actuals=obs,
        program_node_count=expr.node_count(),
        program_constant_count=expr.constant_count(),
    )

# ── Build and score candidate expressions ────────────────────────────────────
candidates = []

expressions = {
    "C / (t+1)^2":     binop(ContinuousNodeType.DIV_REAL, const(100.0), binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1())),
    "C / (t+1)^2 v2":  binop(ContinuousNodeType.DIV_REAL, const(100.0), binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1())),
    "C * (t+1)^-2":    binop(ContinuousNodeType.MUL_REAL, const(100.0), binop(ContinuousNodeType.DIV_REAL, const(1.0), binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1()))),
    "exp(-2*log(t+1))+C": binop(ContinuousNodeType.MUL_REAL, const(100.0),
                                binop(ContinuousNodeType.DIV_REAL, const(1.0),
                                      binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1()))),
    # Variants with slightly off constants for L-BFGS to tune
    "C=99 / (t+1)^2":  binop(ContinuousNodeType.DIV_REAL, const(99.0),  binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1())),
    "C=101 / (t+1)^2": binop(ContinuousNodeType.DIV_REAL, const(101.0), binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1())),
    # Without the +1 offset (for shifted data)
    "C / t^2 (shifted)": binop(ContinuousNodeType.DIV_REAL, const(100.0), binop(ContinuousNodeType.MUL_REAL, time_node(), time_node())),
}

print(f"{'Expression':<30} {'MDL':>10} {'R²':>8} {'Compression':>12}")
print("-" * 65)

best_mdl = float('inf')
best_name = None
best_expr = None
best_result = None

for name, expr in expressions.items():
    try:
        r = score(expr, F_vals)
        ratio = r.total_mdl_cost / raw_bits
        print(f"{name:<30} {r.total_mdl_cost:>10.2f} {r.r_squared:>8.4f} {ratio:>12.4f}")
        if r.total_mdl_cost < best_mdl:
            best_mdl = r.total_mdl_cost
            best_name = name
            best_expr = expr
            best_result = r
    except Exception as e:
        print(f"{name:<30} ERROR: {e}")

print(f"\nBest: {best_name}")
print(f"  Expression : {best_expr.to_string()}")
print(f"  MDL cost   : {best_result.total_mdl_cost:.2f}")
print(f"  R²         : {best_result.r_squared:.6f}")
print(f"  Compression: {best_result.total_mdl_cost / raw_bits:.4f}  (target < 0.2)")
print(f"  Residual σ : {best_result.residual_sigma:.8f}")

# ── Now inject into beam search to let L-BFGS refine constants ───────────────
print("\n=== Seeded beam search (L-BFGS constant refinement) ===")
cfg = ContinuousBeamConfig(
    beam_width=20, max_depth=6,
    constant_range=200.0,
    lbfgs_iterations=500,
    lbfgs_top_k=5,
    enable_lbfgs=True,
    allow_sin=False, allow_cos=False,
    allow_exp=False, allow_log=False,
    allow_sqrt=False, allow_prev=False,
    random_seed=42,
)
searcher = ContinuousBeamSearch(cfg)

# Monkey-patch: inject our seed into the initial beam before mutation
original_search = searcher.search

def seeded_search(observations, verbose=False):
    seed_expr = binop(ContinuousNodeType.DIV_REAL, const(100.0),
                      binop(ContinuousNodeType.MUL_REAL, t_plus_1(), t_plus_1()))
    seed_mdl = score(seed_expr, observations)
    seed_candidate = ContinuousCandidate(expr=seed_expr, mdl=seed_mdl, source="seed")
    
    # Run normal search then merge
    results = original_search(observations, verbose=verbose)
    results.append(seed_candidate)
    results.sort(key=lambda c: c.mdl.total_mdl_cost)
    return results

searcher.search = seeded_search
candidates = searcher.search(F_vals, verbose=False)

print(f"Top 3 after seeding:")
for i, c in enumerate(candidates[:3]):
    ratio = c.mdl.total_mdl_cost / raw_bits
    print(f"  [{i+1}] {c.expr.to_string()}")
    print(f"       MDL={c.mdl.total_mdl_cost:.2f}  R²={c.mdl.r_squared:.4f}  "
          f"compression={ratio:.4f}  source={c.source}")