"""Run the definitive OUROBOROS final benchmark."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from ouroboros.benchmark.final_benchmark import FinalBenchmark

def main():
    bench = FinalBenchmark(n_seeds=3, stream_length=150, verbose=True)
    results = bench.run_full()
    r = results["ouroboros_v13"]
    print(f"\n{'='*60}")
    print(f"FINAL SCORE: {r.total_points:.1f}/{r.max_points} ({r.percentage:.1f}%)")
    if r.percentage >= 95:
        print("TARGET ACHIEVED: 95+/100")
    elif r.percentage >= 90:
        print("EXCELLENT: 90+/100")
    else:
        print(f"  {r.percentage:.1f}/100 -- continue improving")

if __name__ == '__main__':
    main()