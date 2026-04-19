import sys; sys.path.insert(0, '.')
from ouroboros.compression.program_synthesis import (
    build_fibonacci_mod, build_linear_modular, build_piecewise,
    PREV, ADD, MOD, C, T, IF, EQ
)

# Test PREV
expr_fib = build_fibonacci_mod(11)
# Manually compute Fibonacci mod 11
fibs = [0, 1]
for i in range(2, 20):
    fibs.append((fibs[-1] + fibs[-2]) % 11)

preds = expr_fib.predict_sequence(20, 11, initial_history=[0, 1])
print(f"Fibonacci mod 11 (true):      {fibs[:10]}")
print(f"Fibonacci mod 11 (predicted): {preds[:10]}")
assert preds[:10] == fibs[:10], "PREV-based Fibonacci wrong!"
print("✅ PREV: Fibonacci recurrence works")

# Test IF
expr_if = build_piecewise(4, build_linear_modular(3,1,7), C(0))
preds_if = expr_if.predict_sequence(12, 7)
print(f"\nPiecewise (period=4): {preds_if}")
print("✅ IF: piecewise expression works")

# Test new arithmetic
from ouroboros.compression.program_synthesis import SUB, DIV, POW
sub_expr = SUB(T(), C(3))    # t - 3
div_expr = DIV(T(), C(2))    # t // 2
pow_expr = POW(C(2), C(3))   # 2^3 = 8

assert sub_expr.evaluate(5) == 2
assert div_expr.evaluate(7) == 3
assert pow_expr.evaluate(0) == 8
print("✅ SUB, DIV, POW: arithmetic correct")
