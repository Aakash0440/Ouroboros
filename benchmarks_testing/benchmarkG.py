import sys
sys.path.insert(0, '.')

from ouroboros.grammar.math_grammar import MathGrammar
from ouroboros.nodes.extended_nodes import ExtNodeType, NODE_SPECS

grammar = MathGrammar(strict=True)

# Only ExtNodeType parents that have explicit grammar rules
test_parents = [
    ExtNodeType.DERIV,
    ExtNodeType.FFT_AMP,
    ExtNodeType.BOOL_AND,
    ExtNodeType.CUMSUM,
    ExtNodeType.MEAN_WIN,
    ExtNodeType.THRESHOLD,
    ExtNodeType.ISPRIME,
]

total_types = len(NODE_SPECS)
total_allowed = 0

print(f"Total node types in grammar : {total_types}")
print()
print(f"{'Node':<16} {'Arg0 allowed':>14} {'% of total':>12}  {'Constraint'}")
print('-' * 60)

for parent in test_parents:
    allowed = grammar.allowed_child_node_types(parent, 0)
    pct = len(allowed) / total_types * 100
    constraint = "tight" if pct < 40 else ("medium" if pct < 70 else "loose")
    total_allowed += len(allowed)
    print(f"{parent.name:<16} {len(allowed):>8}/{total_types:<5} {pct:>8.1f}%   {constraint}")

avg_branching = total_allowed / len(test_parents)
reduction     = total_types / avg_branching

print(f"\nAverage branching factor : {avg_branching:.2f}")
print(f"Unconstrained branching  : {total_types}")
print(f"Reduction                : {reduction:.1f}x")
print(f"Effective branching (all): {grammar.effective_branching_factor(0):.2f}")
print(f"Search space @ depth 5   : {grammar.search_space_size(5):.3e}")
print(f"\nRESULT: {'PASS' if reduction >= 2.0 else 'FAIL -- minimal constraint'}")
