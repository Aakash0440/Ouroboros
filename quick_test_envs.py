# quick_test_envs.py — run interactively, not committed
import sys; sys.path.insert(0, '.')
from ouroboros.continuous.environments import (
    SineEnv, PolynomialEnv, DampedOscillatorEnv, LogisticMapEnv
)

s = SineEnv()
seq = s.generate(20)
print("SineEnv:", [f"{v:.4f}" for v in seq[:8]])
# Expected: [0.0000, 0.1411, 0.2817, ..., pattern of sin(t/7)]

p = PolynomialEnv()
seq = p.generate(10)
print("PolyEnv:", [f"{v:.3f}" for v in seq])
# Expected: [1.0, -0.5, 1.0, 4.5, 9.0, 14.5, 21.0, ...]
# (1.0 - 2t + 0.5t²) at t=0..9

d = DampedOscillatorEnv()
seq = d.generate(15)
print("DampedOsc:", [f"{v:.4f}" for v in seq])
# Expected: oscillation decaying toward 0

l35 = LogisticMapEnv(r=3.5)
print("Logistic r=3.5:", [f"{v:.4f}" for v in l35.generate(20)])
# Expected: 4-cycle pattern settling

l39 = LogisticMapEnv(r=3.9)
print("Logistic r=3.9:", [f"{v:.4f}" for v in l39.generate(20)])
# Expected: chaotic, no visible pattern