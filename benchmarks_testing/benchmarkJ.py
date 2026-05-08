import math
import sys


def test_canonical_sequence():
    """
    (3t+1)%7 is the canonical OUROBOROS test case.
    It must achieve compression ratio < 0.01 consistently.
    This test fails if it does not.
    """
    from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig

    obs = [(3*t + 1) % 7 for t in range(200)]

    results = []
    for seed in range(10):
        router = HierarchicalSearchRouter(RouterConfig(
            beam_width=20, max_depth=4, n_iterations=10,
            random_seed=seed
        ))
        result = router.search(obs, alphabet_size=8)
        baseline = 200 * math.log2(8)
        ratio = result.mdl_cost / baseline
        results.append(ratio)

    mean_ratio = sum(results) / len(results)
    success_rate = sum(1 for r in results if r < 0.01) / len(results)

    print(f"Canonical sequence (3t+1)%7 across 10 seeds:")
    print(f"  Mean compression ratio : {mean_ratio:.4f}")
    print(f"  Success rate (< 0.01)  : {success_rate:.1%}")
    print(f"  Individual ratios      : {[round(r, 4) for r in results]}")

    assert mean_ratio < 0.01, (
        f"REGRESSION: canonical sequence failing, mean ratio={mean_ratio:.4f}"
    )
    assert success_rate >= 0.8, (
        f"REGRESSION: success rate {success_rate:.1%} too low"
    )
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("  Canonical Sequence Regression Test")
    print("=" * 60)
    try:
        test_canonical_sequence()
        print("\nOverall: PASS")
        sys.exit(0)
    except AssertionError as e:
        print(f"\nOverall: FAIL\n  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nOverall: ERROR\n  {type(e).__name__}: {e}")
        sys.exit(2)