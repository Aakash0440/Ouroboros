"""
Physics law discovery experiment.

Demonstrates OUROBOROS with 60 nodes discovering physical laws from data.

Experiment 1: Spring-Mass System
  Data: discrete position measurements x(t) = A*cos(ωt)
  Discovery target: DERIV2(x) correlates with x (Hooke's Law)
  
Experiment 2: Radioactive Decay
  Data: count(t) = N₀ * exp(-λt) discretized
  Discovery target: DERIV(count)/count ≈ constant (decay rate)
  
Experiment 3: Free Fall
  Data: y(t) = h - (g/2)t² discretized
  Discovery target: DERIV2(y) ≈ constant (constant acceleration)
"""

import sys; sys.path.insert(0, '.')
import time
from ouroboros.environments.physics import SpringMassEnv, RadioactiveDecayEnv, FreeFallEnv
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig


def run_physics_discovery():
    print("PHYSICS LAW DISCOVERY EXPERIMENT")
    print("=" * 60)

    envs = [
        ("Spring-Mass", SpringMassEnv(amplitude=10, omega=0.3), 50),
        ("Radioactive Decay", RadioactiveDecayEnv(n0=100, decay_rate=0.05), 200),
        ("Free Fall", FreeFallEnv(h0=50, g=9.8, scale=0.2), 50),
    ]

    cfg = RouterConfig(beam_width=15, max_depth=4, n_iterations=8, random_seed=42)
    router = HierarchicalSearchRouter(cfg)

    for env_name, env, length in envs:
        print(f"\n── {env_name} ──")
        print(f"  Ground truth: {env.ground_truth_rule()}")
        print(f"  Target law:   {env.discovered_law()}")

        obs = env.generate(length)
        print(f"  Sequence[:10]: {obs[:10]}")

        start = time.time()
        result = router.search(obs, alphabet_size=env.alphabet_size, verbose=False)
        elapsed = time.time() - start

        print(f"  Classified as: {result.math_family.name} ({result.classification_confidence:.2f})")
        print(f"  MDL cost: {result.mdl_cost:.2f} bits")
        print(f"  Expression: {result.expr.to_string()[:80] if result.expr else 'None'}")
        print(f"  Time: {elapsed:.2f}s")


if __name__ == '__main__':
    run_physics_discovery()
    print("\n✅ Physics discovery experiment complete")