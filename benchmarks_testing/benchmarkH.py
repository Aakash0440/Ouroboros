from ouroboros.compression.mdl_engine import MDLEngine

mdl = MDLEngine()

print("=" * 60)
print("  MDL Engine Calibration Test")
print("=" * 60)

const_seq = [5] * 100
perfect_result = mdl.compute(predictions=[5]*100, actuals=const_seq, node_count=1, constant_count=1)
print(f"\nTest 1 - Perfect fit (CONST=5 on [5]*100)")
print(f"  Total MDL  : {perfect_result.total_mdl_cost:.4f}")
print(f"  Program    : {perfect_result.program_bits:.4f}")
print(f"  Data       : {perfect_result.error_bits:.4f}")
data_ok = perfect_result.error_bits < 0.1
print(f"  RESULT     : {'PASS' if data_ok else 'FAIL -- error_bits > 0.1'}")

near_perfect = [5]*99 + [6]
near_result = mdl.compute(predictions=[5]*100, actuals=near_perfect, node_count=1, constant_count=1)
print(f"\nTest 2 - Near-perfect fit ([5]*99 + [6])")
print(f"  Total MDL  : {near_result.total_mdl_cost:.4f}")
print(f"  Program    : {near_result.program_bits:.4f}")
print(f"  Data       : {near_result.error_bits:.4f}")
ordering_ok = near_result.total_mdl_cost > perfect_result.total_mdl_cost
print(f"  RESULT     : {'PASS' if ordering_ok else 'FAIL -- ordering violation'}")

print(f"\nTest 3 - Monotonicity (increasing error count)")
prev_mdl = None
monotone = True
for n_wrong in [0, 1, 5, 10, 25, 50]:
    obs = [5]*(100-n_wrong) + [6]*n_wrong
    result = mdl.compute(predictions=[5]*100, actuals=obs, node_count=1, constant_count=1)
    flag = ""
    if prev_mdl is not None and result.total_mdl_cost < prev_mdl:
        monotone = False
        flag = " <- VIOLATION"
    print(f"  {n_wrong:>3} wrong  ->  MDL={result.total_mdl_cost:8.4f}{flag}")
    prev_mdl = result.total_mdl_cost
print(f"  RESULT     : {'PASS' if monotone else 'FAIL -- not monotone'}")

print(f"\nTest 4 - Complexity penalty (increasing node_count)")
prev_prog = None
penalty_ok = True
for n in [1, 2, 5, 10, 20]:
    result = mdl.compute(predictions=[5]*100, actuals=const_seq, node_count=n, constant_count=n)
    flag = ""
    if prev_prog is not None and result.program_bits < prev_prog:
        penalty_ok = False
        flag = " <- VIOLATION"
    print(f"  node_count={n:<3}  ->  program_bits={result.program_bits:8.4f}{flag}")
    prev_prog = result.program_bits
print(f"  RESULT     : {'PASS' if penalty_ok else 'FAIL -- not monotone'}")

print(f"\n{'='*60}")
all_pass = data_ok and ordering_ok and monotone and penalty_ok
print(f"  Overall: {'ALL PASS' if all_pass else 'SOME TESTS FAILED'}")
print('='*60)
