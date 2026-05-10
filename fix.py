"""
seed_survival.py — Check if growth seeds survive into beam at iteration 0
and what kills them during mutation.
"""
import math
from feynman_data.equations import FEYNMAN_1VAR
from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
from ouroboros.compression.mdl_engine import MDLEngine

mdl = MDLEngine()

CHECK = ["exp_1", "inv_1", "quad_1", "I.9.18", "I.12.2", "inv2_1"]

for name, true_expr_str, fn, n_points, alpha in FEYNMAN_1VAR:
    if name not in CHECK:
        continue

    seq = [fn(t) for t in range(n_points)]
    seq = [max(-10000, min(10000, v)) for v in seq]
    alpha = max(alpha, max(seq) + 2)

    beam = GrammarConstrainedBeam(GrammarBeamConfig(
        beam_width=25, n_iterations=15, random_seed=42
    ))
    beam._alphabet_size = alpha

    # Build seeds manually
    growth_seeds = beam._seed_growth_templates(seq)
    mod_seeds    = beam._seed_modular_templates(seq)
    rec_seeds    = beam._seed_recurrence_templates(seq)

    print(f"\n── {name} (alpha={alpha}, n={n_points}) ──")
    print(f"  growth_seeds count: {len(growth_seeds)}")
    print(f"  mod_seeds count:    {len(mod_seeds)}")
    print(f"  rec_seeds count:    {len(rec_seeds)}")

    if growth_seeds:
        print(f"  TOP 5 growth seeds:")
        for s in growth_seeds[:5]:
            print(f"    {s.mdl_cost:8.2f}  {s.expr.to_string()[:60]}")

    if mod_seeds:
        print(f"  TOP 3 mod seeds:")
        for s in mod_seeds[:3]:
            print(f"    {s.mdl_cost:8.2f}  {s.expr.to_string()[:60]}")

    # Simulate the slot reservation logic
    growth_slots = growth_seeds[:max(8, 25 // 3)]
    mod_slots    = mod_seeds[:max(4, 25 // 6)]
    rec_slots    = rec_seeds[:max(4, 25 // 6)]

    from ouroboros.search.grammar_beam import GrammarBeamCandidate
    import random
    rng = random.Random(42)
    init_exprs = [beam._random_expr() for _ in range(25 * 5)]
    init_costs = [beam._score(e, seq) for e in init_exprs]
    random_slots = sorted(
        [GrammarBeamCandidate(e, c) for e, c in zip(init_exprs, init_costs)]
    )[:max(4, 25 // 6)]

    all_c = growth_slots + mod_slots + rec_slots + random_slots
    seen = set()
    deduped = []
    for c in sorted(all_c):
        key = round(c.mdl_cost, 1)
        if key not in seen:
            seen.add(key)
            deduped.append(c)
    initial_beam = deduped[:25]

    print(f"  INITIAL BEAM top 5 (after slot reservation):")
    for c in initial_beam[:5]:
        print(f"    {c.mdl_cost:8.2f}  {c.expr.to_string()[:60]}")

    # Check: is a good growth seed in the initial beam?
    best_growth_cost = growth_seeds[0].mdl_cost if growth_seeds else float('inf')
    best_growth_in_beam = any(
        abs(c.mdl_cost - best_growth_cost) < 0.2
        for c in initial_beam
    )
    print(f"  Best growth seed cost: {best_growth_cost:.2f}")
    print(f"  Best growth seed IN initial beam: {best_growth_in_beam}")

    # Now simulate 1 iteration of mutation to see if it survives
    best_before = initial_beam[0].mdl_cost
    new_candidates = list(initial_beam)
    for cand in initial_beam:
        for _ in range(2):
            mutated = beam._mutate_grammar(cand.expr)
            cost = beam._score(mutated, seq)
            new_candidates.append(GrammarBeamCandidate(mutated, cost))
    new_candidates.sort()
    beam_after_1 = new_candidates[:25]
    best_after = beam_after_1[0].mdl_cost

    print(f"  Best cost BEFORE iter 1: {best_before:.2f}")
    print(f"  Best cost AFTER  iter 1: {best_after:.2f}")
    print(f"  After iter 1 top expr:   {beam_after_1[0].expr.to_string()[:60]}")
    improved = best_after < best_before - 1.0
    print(f"  Improved: {improved}")