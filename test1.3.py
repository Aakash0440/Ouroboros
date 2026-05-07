# pip install pysr sympy
import math, sys
from types import SimpleNamespace
sys.path.insert(0, '.')

from pysr import PySRRegressor
import numpy as np

from ouroboros.search.hierarchical_router import HierarchicalSearchRouter
from ouroboros.core.sequence_env import SequenceEnvironment
from ouroboros.core.phase1_runner import Phase1Runner

# ── Monkey-patch: fast capped search (same as test suite) ─────────────────────
MAX_STEPS     = 2000
EVAL_INTERVAL = 500

def quick_search(self, sequence, alphabet_size=None, context=None, **kwargs):
    alpha    = alphabet_size or (max(sequence) - min(sequence) + 2)
    raw_bits = len(sequence) * math.log2(max(alpha, 2))

    env     = SequenceEnvironment(sequence, alphabet_size=alphabet_size)
    runner  = Phase1Runner(env, "sequence_env")
    results = runner.run(
        stream_length=MAX_STEPS,
        eval_interval=EVAL_INTERVAL,
        verbose=False,
    )

    ratio = results.best_ratio
    expr  = results.discovery_expression or "literal"

    return SimpleNamespace(
        compression_ratio=ratio,
        mdl_cost=ratio * raw_bits,
        expr=expr,
        program_mdl=0.0,
        data_mdl=ratio * raw_bits,
    )

HierarchicalSearchRouter.search = quick_search


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_expr_to_str(expr):
    if expr is None:
        return "None"
    if hasattr(expr, "to_string"):
        return expr.to_string()
    return str(expr)


def compare(sequence_name, X_vals, y_vals):
    print(f"\n{'='*60}")
    print(f"  {sequence_name}")
    print('='*60)

    # ── PySR ──────────────────────────────────────────────────────────────────
    X = np.array(X_vals).reshape(-1, 1)
    y = np.array(y_vals, dtype=float)

    model = PySRRegressor(
        niterations=30,
        binary_operators=["+", "*", "-", "/"],
        unary_operators=["sin", "cos", "exp", "log"],
        verbosity=0,
    )
    model.fit(X, y)
    pysr_best = model.get_best()

    # ── OUROBOROS ─────────────────────────────────────────────────────────────
    scale   = 100
    obs_int = [int(round(v * scale)) for v in y_vals]
    alpha   = max(abs(min(obs_int)), abs(max(obs_int))) + 2

    try:
        result    = HierarchicalSearchRouter().search(obs_int, alphabet_size=int(alpha))
        ouro_expr = safe_expr_to_str(result.expr)
        ouro_mdl  = result.mdl_cost
        ouro_cr   = result.compression_ratio
    except Exception as e:
        ouro_expr = f"ERROR: {e}"
        ouro_mdl  = float('inf')
        ouro_cr   = float('inf')

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\nPySR:")
    print(f"  Equation : {pysr_best['equation']}")
    print(f"  Score    : {pysr_best['score']:.6f}")

    print(f"\nOUROBOROS:")
    print(f"  Expression         : {ouro_expr}")
    print(f"  MDL cost (bits)    : {ouro_mdl:.2f}")
    print(f"  Compression ratio  : {ouro_cr:.4f}")

    print("\nNotes:")
    print("  • PySR   — optimizes numeric fit (MSE-like)")
    print("  • OUROBOROS — optimizes compression (MDL)")


# ── Benchmark cases ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    compare(
        "Hooke-like Oscillation  [10·cos(0.3t)]",
        X_vals=[t * 0.1 for t in range(100)],
        y_vals=[10 * math.cos(0.3 * t) for t in range(100)],
    )

    compare(
        "Linear Decay  [10 - 0.5t]",
        X_vals=list(range(50)),
        y_vals=[10 - 0.5 * t for t in range(50)],
    )

    compare(
        "Quadratic  [t²]",
        X_vals=list(range(40)),
        y_vals=[t ** 2 for t in range(40)],
    )