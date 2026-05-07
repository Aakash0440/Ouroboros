from collections import Counter
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig

obs = [(3*t + 1) % 7 for t in range(200)]

top_exprs = []
for seed in range(20):
    r = HierarchicalSearchRouter(RouterConfig(
        beam_width=20, n_iterations=10, random_seed=seed
    ))
    result = r.search(obs)
    if result.expr:
        top_exprs.append(result.math_family.name)

family_counts = Counter(top_exprs)
found = len(top_exprs)

print("Family distribution across 20 seeds:")
for family, count in family_counts.most_common():
    bar = "█" * count
    print(f"  {family:<20s} {count:2d}/20 {bar}")

if found < 20:
    print(f"\n  WARNING: {20 - found} seeds returned no expression (result.expr was None/empty)")

dominant = family_counts.most_common(1)
if dominant:
    top_family, top_count = dominant[0]
    number_theor_ok = top_family == "NUMBER_THEOR" and top_count > 14
    no_bad_families = family_counts.get("MIXED", 0) + family_counts.get("STATISTICAL", 0) < 4

    print(f"\nDiagnostics:")
    print(f"  Top family      : {top_family} ({top_count}/20)")
    print(f"  NUMBER_THEOR >14: {'PASS' if number_theor_ok else 'FAIL -- classifier may be weak'}")
    print(f"  MIXED/STAT < 4  : {'PASS' if no_bad_families else 'FAIL -- classifier routing to wrong family'}")
    print(f"  Overall         : {'PASS' if number_theor_ok and no_bad_families else 'FAIL'}")
else:
    print("\n  FAIL -- no expressions found across all seeds")