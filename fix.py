"""
Patch: extended_nodes.py

ExtExprNode._eval only handles ExtNodeType members.
But _seed_modular_templates builds nodes with old NodeType (CONST, TIME, MOD, MUL, ADD)
from program_synthesis — these fall through to `return PENALTY`.

Fix: insert handlers for the old NodeType arithmetic ops right before `return PENALTY`.
"""

path = "ouroboros/nodes/extended_nodes.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

old_tail = "        return PENALTY\n\n    def node_count"

new_tail = """\
        # ── Old NodeType arithmetic (ADD, SUB, MUL, DIV, MOD, POW, EQ, LT, IF) ──
        # These come from ouroboros.synthesis.expr_node / program_synthesis.
        # _seed_modular_templates builds trees with these types, so they must
        # be handled here or they silently return PENALTY.
        name = nt.name if hasattr(nt, 'name') else ''
        if name == 'ADD':
            return L() + R()
        if name == 'SUB':
            return L() - R()
        if name == 'MUL':
            return L() * R()
        if name == 'DIV':
            r = R()
            return L() / r if abs(r) > EPS else PENALTY
        if name == 'MOD':
            r = R()
            ri = int(round(r))
            return float(int(round(L())) % ri) if ri != 0 else PENALTY
        if name == 'POW':
            base, exp = L(), R()
            if abs(base) > 1e4 or abs(exp) > 20:
                return PENALTY
            try:
                return float(base ** exp)
            except Exception:
                return PENALTY
        if name == 'EQ':
            return 1.0 if abs(L() - R()) < 0.5 else 0.0
        if name == 'LT':
            return 1.0 if L() < R() else 0.0
        if name == 'IF':
            return L() if (self.right and R() != 0) else (T() if self.third else 0.0)
        if name == 'ABS':
            return abs(L())
        if name == 'SIN':
            return math.sin(L())
        if name == 'COS':
            return math.cos(L())
        if name == 'EXP':
            v = L()
            return math.exp(min(v, 700)) if v < 700 else PENALTY
        if name == 'LOG':
            v = abs(L())
            return math.log(v) if v > EPS else PENALTY
        if name == 'SQRT':
            v = L()
            return math.sqrt(v) if v >= 0 else PENALTY
        if name == 'PRIME':
            n = int(abs(L()))
            if n < 2: return 0.0
            if n == 2: return 1.0
            if n % 2 == 0: return 0.0
            for i in range(3, min(int(n**0.5)+1, 1000), 2):
                if n % i == 0: return 0.0
            return 1.0

        return PENALTY

    def node_count"""

assert old_tail in src, "ERROR: could not find 'return PENALTY' tail — file may have changed"
src = src.replace(old_tail, new_tail, 1)  # only replace the last one before node_count

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("Patched extended_nodes.py")

# Verify: manually build (3*t+1)%7 with old NodeType and check evaluate
import importlib
import ouroboros.nodes.extended_nodes as en
importlib.reload(en)
ExtExprNode = en.ExtExprNode

from ouroboros.synthesis.expr_node import NodeType

def make(nt, val=0.0, left=None, right=None, third=None):
    n = ExtExprNode.__new__(ExtExprNode)
    n.node_type = nt; n.value = float(val); n.lag = 1
    n.state_key = 0; n.window = 10
    n.left = left; n.right = right; n.third = third; n._cache = {}
    return n

obs = [(3*t+1)%7 for t in range(20)]
t_node = make(NodeType.TIME)
c3     = make(NodeType.CONST, 3)
c1     = make(NodeType.CONST, 1)
c7     = make(NodeType.CONST, 7)
mul    = make(NodeType.MUL, left=c3, right=t_node)
add    = make(NodeType.ADD, left=mul, right=c1)
mod    = make(NodeType.MOD, left=add, right=c7)

print("\nEvaluate (3*t+1)%7 check:")
state = {}
all_ok = True
for t in range(10):
    got = mod.evaluate(t, obs, state)
    exp = (3*t+1)%7
    ok = abs(got - exp) < 0.5
    if not ok:
        all_ok = False
    print(f"  t={t}  got={got}  expected={exp}  {'OK' if ok else 'FAIL'}")

print(f"\nNode eval fix: {'PASS' if all_ok else 'FAIL'}")

# Now check seed scoring
if all_ok:
    from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
    import importlib as il
    import ouroboros.search.grammar_beam as gb
    il.reload(gb)
    GrammarConstrainedBeam = gb.GrammarConstrainedBeam
    GrammarBeamConfig = gb.GrammarBeamConfig

    b = GrammarConstrainedBeam(GrammarBeamConfig(random_seed=0))
    obs200 = [(3*t+1)%7 for t in range(200)]
    seeds = b._seed_modular_templates(obs200)
    print(f"\nTop 3 seeds after fix:")
    for s in seeds[:3]:
        print(f"  cost={s.mdl_cost:.4f}  expr={s.expr.to_string()}")
    print(f"\nSeed fix: {'PASS' if seeds[0].mdl_cost < 10 else 'FAIL -- seeds still broken'}")