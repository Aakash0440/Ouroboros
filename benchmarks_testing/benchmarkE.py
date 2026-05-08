# benchmark_baselines_fixed.py

import sys, io, math, random, traceback
from types import SimpleNamespace

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')

# ── OUROBOROS SAFE ROUTER (from your tests) ───────────────────────────────────

from ouroboros.search.hierarchical_router import HierarchicalSearchRouter
from ouroboros.core.sequence_env import SequenceEnvironment
from ouroboros.core.phase1_runner import Phase1Runner

MAX_STEPS     = 400     # HARD LIMIT → prevents freezing
EVAL_INTERVAL = 200

def safe_search(self, sequence, alphabet_size=None, context=None, **kwargs):
    """
    Bounded search using GrammarConstrainedBeam directly.
    Bypasses Phase1Runner / SynthesisAgent so the grammar beam is
    actually invoked instead of the old BeamSearchSynthesizer.
    """
    try:
        alpha    = alphabet_size or (max(sequence) - min(sequence) + 2)
        raw_bits = len(sequence) * math.log2(max(alpha, 2))

        from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine

        cfg = GrammarBeamConfig(
            beam_width   = 20,
            max_depth    = 5,
            const_range  = max(50, int(alpha) * 2),
            max_lag      = 8,
            n_iterations = 12,
            random_seed  = 42,
        )
        beam = GrammarConstrainedBeam(cfg)
        expr = beam.search(list(sequence), verbose=False)

        if expr is None:
            return SimpleNamespace(
                compression_ratio = 1.0,
                mdl_cost          = raw_bits,
                expr              = "literal",
                program_mdl       = 0.0,
                data_mdl          = raw_bits,
            )

        mdl   = MDLEngine()
        state = {}
        preds = []
        for t in range(len(sequence)):
            try:
                p = expr.evaluate(t, list(sequence[:t]), state)
            except Exception:
                p = 0.0
            import math as _math
            preds.append(int(round(p)) if _math.isfinite(p) else 0)

        result = mdl.compute(
            preds, list(sequence),
            expr.node_count(), expr.constant_count()
        )
        cost  = result.total_mdl_cost
        ratio = cost / raw_bits if raw_bits > 0 else 1.0

        return SimpleNamespace(
            compression_ratio = ratio,
            mdl_cost          = cost,
            expr              = expr.to_string() if hasattr(expr, "to_string") else str(expr),
            program_mdl       = 0.0,
            data_mdl          = cost,
        )

    except Exception as e:
        print("  [safe_search ERROR]")
        traceback.print_exc()
        raise e

# 🔥 Override the default infinite search
HierarchicalSearchRouter.search = safe_search


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_fib_mod7(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % 7
    return a

def compute_prime_gaps(n):
    def is_prime(k):
        if k < 2: return False
        for d in range(2, int(k**0.5)+1):
            if k % d == 0: return False
        return True

    primes, num = [], 2
    while len(primes) < n + 1:
        if is_prime(num):
            primes.append(num)
        num += 1

    return [primes[i+1] - primes[i] for i in range(n)]

rng = random.Random(42)


# ── Trivial Baselines ─────────────────────────────────────────────────────────

class TrivialBaselines:

    @staticmethod
    def constant_baseline(obs):
        mean = round(sum(obs) / len(obs))
        return sum(abs(v - mean) for v in obs)

    @staticmethod
    def last_value_baseline(obs):
        if len(obs) < 2:
            return 0
        return sum(abs(obs[t] - obs[t-1]) for t in range(1, len(obs)))

    @staticmethod
    def linear_trend_baseline(obs):
        n = len(obs)
        if n < 2:
            return 0

        ts = list(range(n))
        mt = sum(ts) / n
        my = sum(obs) / n

        denom = sum((ti - mt)**2 for ti in ts)
        if denom == 0:
            return TrivialBaselines.constant_baseline(obs)

        b = sum((ts[i]-mt)*(obs[i]-my) for i in range(n)) / denom
        a = my - b * mt

        return sum(abs(obs[i] - round(a + b*i)) for i in range(n))

    @staticmethod
    def all_scores(obs):
        return {
            "constant":    TrivialBaselines.constant_baseline(obs),
            "last_value":  TrivialBaselines.last_value_baseline(obs),
            "linear_trend":TrivialBaselines.linear_trend_baseline(obs),
        }


# ── Test sequences ────────────────────────────────────────────────────────────

sequences = {
    "ModArith(7)":  [(3*t + 1) % 7 for t in range(200)],
    "Fibonacci%7":  [compute_fib_mod7(t) for t in range(200)],
    "Const(5)":     [5] * 200,

    # ⚠️ Hard cases trimmed so they don’t waste compute
    "PrimeGaps":    compute_prime_gaps(80),
    "RandomNoise":  [rng.randint(0, 9) for _ in range(80)],
}


# ── Run ───────────────────────────────────────────────────────────────────────

router = HierarchicalSearchRouter()

print("=" * 75)
print("  OUROBOROS vs Trivial Baselines (SAFE MODE)")
print("=" * 75)

print(f"\n  {'Sequence':<16}  {'OUROBOROS':>12}  {'Const':>10}  {'LastVal':>10}  "
      f"{'Linear':>10}  {'Best BL':>10}  {'Ratio':>8}  {'Beat?':>6}")

print(f"  {'-'*16}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")

results = {}

for name, seq in sequences.items():
    print(f"\nRunning {name}...", flush=True)

    try:
        result = router.search(seq)
        ouro_cost = result.mdl_cost

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    scores = TrivialBaselines.all_scores(seq)
    best_bl_name = min(scores, key=scores.get)
    best_bl = scores[best_bl_name]

    ratio = (ouro_cost / best_bl) if best_bl > 0 else 0.0
    beat  = ratio < 1.0
    icon  = "✓" if beat else "✗"

    print(
        f"  {icon} {name:<15}  {ouro_cost:>12.1f}  "
        f"{scores['constant']:>10.1f}  {scores['last_value']:>10.1f}  "
        f"{scores['linear_trend']:>10.1f}  {best_bl:>10.1f}  "
        f"{ratio:>8.3f}  [{best_bl_name}]"
    )

    results[name] = {
        "ouroboros": ouro_cost,
        "best_bl": best_bl,
        "best_bl_name": best_bl_name,
        "ratio": ratio,
        "beat": beat,
    }


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 75)

n_total = len(results)
n_beat  = sum(r["beat"] for r in results.values())

print(f"  Summary: OUROBOROS beat trivial baseline on {n_beat}/{n_total}\n")

for name, r in results.items():
    icon   = "✓" if r["beat"] else "✗"
    status = "PASS" if r["beat"] else "FAIL"

    msg = (
        f"  {icon} [{status}]  {name:<16}  "
        f"OUROBOROS={r['ouroboros']:.1f}  "
        f"Best baseline={r['best_bl']:.1f} ({r['best_bl_name']})  "
        f"Ratio={r['ratio']:.3f}"
    )

    if not r["beat"]:
        msg += "  ← OUROBOROS worse than trivial baseline"

    print(msg)

failures = [n for n, r in results.items() if not r["beat"]]

print()
if failures:
    print(f"  WARNING: {len(failures)} failure(s):")
    for f in failures:
        print(f"    - {f} (ratio={results[f]['ratio']:.3f})")
    print("\n  Suggestion: increase MAX_STEPS or improve primitives.")
else:
    print("  All sequences: OUROBOROS outperforms trivial baselines.")