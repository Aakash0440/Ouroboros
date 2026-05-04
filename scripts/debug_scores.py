# scripts/debug_scores.py
import math, statistics
from ouroboros.compression.mdl_engine import MDLEngine
from ouroboros.physics.law_signature import _test_free_fall
from ouroboros.environments.physics import FreeFallEnv

# ── Debug 1: MDL ratio breakdown ──────────────────────────────────────────────
mdl = MDLEngine()
for mod, slope, intercept, name in [(7, 3, 1, "ModArith(7)"), (11, 5, 2, "ModArith(11)")]:
    obs   = [(slope*t + intercept) % mod for t in range(150)]
    preds = obs[:]  # perfect prediction
    baseline = 150 * math.log2(mod)
    r = mdl.compute(preds, list(obs), node_count=5, constant_count=3)
    print(f"{name}: cost={r.total_mdl_cost:.4f}, baseline={baseline:.4f}, "
          f"ratio={r.total_mdl_cost/baseline:.4f}")
    print(f"  program_bits={r.program_bits:.4f}, error_bits={r.error_bits:.4f}")

# ── Debug 2: Free Fall cv value ───────────────────────────────────────────────
env = FreeFallEnv(h0=100, g=9.8, scale=0.05)
obs = [float(v) for v in env.generate(100)]
print(f"\nFreeFall first 5 values: {obs[:5]}")
print(f"FreeFall last 5 values:  {obs[-5:]}")
result = _test_free_fall(obs, threshold=0.75)
print(f"Free fall: passed={result.passed}, conf={result.confidence:.4f}, "
      f"cv={result.key_value:.6f}, threshold={result.threshold}")