"""
GCD Discovery Experiment.

Can OUROBOROS discover that PrimeCountEnv = CUMSUM(ISPRIME(TIME))?
Can it discover Collatz stopping times have IF-based structure?

The key question: does the hierarchical router, when presented with
a number-theoretic sequence, correctly classify it and assign high
weight to NUMBER and LOGICAL category nodes?

Expected results:
  PrimeCountEnv: classifier → NUMBER_THEORETIC → ISPRIME+CUMSUM seeds
  CollatzEnv: classifier → MIXED (complex) → IF+THRESHOLD nodes explored
  GCDEnv: classifier → NUMBER_THEORETIC → GCD_NODE gets high weight
"""

import sys; sys.path.insert(0, '.')
import time
from ouroboros.environments.algorithm_env import (
    GCDEnv, PrimeCountEnv, CollatzEnv, FibonacciDirectEnv,
)
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.search.fft_period_finder import FFTPeriodFinder


def analyze_environment(env, stream_length: int = 200):
    print(f"\n── {env.name} ──")
    obs = env.generate(stream_length)
    print(f"  Sample: {obs[:10]}")
    print(f"  Alphabet size: {env.alphabet_size}")
    print(f"  Ground truth: {env.ground_truth_algorithm()}")

    # FFT period check
    finder = FFTPeriodFinder()
    period = finder.find_dominant_period([float(v) for v in obs], verbose=True)
    if period and period.is_significant:
        print(f"  FFT period: {period.period} steps (confidence={period.confidence:.2f})")
    else:
        print(f"  FFT: no significant period detected")

    # Hierarchical search
    cfg = RouterConfig(beam_width=12, max_depth=5, n_iterations=8, random_seed=42)
    router = HierarchicalSearchRouter(cfg)

    start = time.time()
    result = router.search(obs, alphabet_size=env.alphabet_size, verbose=False)
    elapsed = time.time() - start

    print(f"  Classified: {result.math_family.name} ({result.classification_confidence:.2f})")
    print(f"  Categories searched: {[c.name for c in result.categories_searched]}")
    print(f"  MDL cost: {result.mdl_cost:.2f} bits")
    print(f"  Time: {elapsed:.2f}s")
    if result.expr:
        print(f"  Expression: {result.expr.to_string()[:80]}")

    return result


def run_prime_count_experiment():
    """
    PrimeCountEnv: π(t) = CUMSUM(ISPRIME(TIME))
    An agent with CUMSUM and ISPRIME should find this in a few iterations.
    """
    print("\nPRIME COUNT EXPERIMENT")
    print("  Target: obs[t] = #{p ≤ t : p is prime}")
    print("  Key nodes needed: ISPRIME + CUMSUM")
    env = PrimeCountEnv()
    obs = env.generate(200)

    # Direct verification: does CUMSUM(ISPRIME(TIME)) match?
    from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
    from ouroboros.synthesis.expr_node import NodeType, ExprNode

    def const_e(v):
        n = ExtExprNode.__new__(ExtExprNode)
        n.node_type = NodeType.CONST; n.value = float(v)
        n.lag = 1; n.state_key = 0; n.window = 10
        n.left = n.right = n.third = None; n._cache = {}
        return n

    def time_e():
        n = ExtExprNode.__new__(ExtExprNode)
        n.node_type = NodeType.TIME; n.value = 0.0
        n.lag = 1; n.state_key = 0; n.window = 10
        n.left = n.right = n.third = None; n._cache = {}
        return n

    # Build CUMSUM(ISPRIME(TIME))
    isprime_expr = ExtExprNode(ExtNodeType.ISPRIME, left=time_e())
    cumsum_expr = ExtExprNode(ExtNodeType.CUMSUM, left=isprime_expr)

    predictions = [int(round(cumsum_expr.evaluate(t, []))) for t in range(100)]
    n_correct = sum(1 for p, o in zip(predictions, obs[:100]) if p == o)
    accuracy = n_correct / 100

    print(f"\n  Direct test: CUMSUM(ISPRIME(TIME))")
    print(f"  Predictions[:10]: {predictions[:10]}")
    print(f"  Observations[:10]: {obs[:10]}")
    print(f"  Accuracy: {n_correct}/100 = {accuracy:.2%}")
    print(f"  ✅ Formula verified!" if accuracy > 0.95 else "  ❌ Formula needs refinement")


if __name__ == '__main__':
    print("ALGORITHM ENVIRONMENT DISCOVERY EXPERIMENT")
    print("=" * 60)

    run_prime_count_experiment()

    for env_cls in [GCDEnv, FibonacciDirectEnv, CollatzEnv]:
        env = env_cls()
        analyze_environment(env, stream_length=150)

    print("\n✅ Algorithm discovery experiment complete")