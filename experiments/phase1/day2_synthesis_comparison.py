# experiments/phase1/day2_synthesis_comparison.py

"""
Compare n-gram (Day 1) vs. symbolic synthesis (Day 2) on all 5 environments.

This generates the BEFORE/AFTER comparison for the paper.

Expected results:
    Environment         | Day 1 (n-gram) | Day 2 (symbolic) | Improvement
    ─────────────────────────────────────────────────────────────────────
    Binary Repeat       |   0.040        |   0.003          |  13x
    Modular Arith       |   0.280        |   0.008          |  35x  ← KEY
    Fibonacci Mod 11    |   0.610        |   0.510          |   1.2x
    Prime Sequence      |   0.870        |   0.850          |   1.0x  (as expected)
    Noise               |   0.960        |   0.970          |  ~1.0x  (sanity check)

The Modular Arith row is the headline result.
35x improvement = agents discovered the algebraic rule.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ouroboros.environmentss import (
    BinaryRepeatEnv, ModularArithmeticEnv, FibonacciModEnv,
    PrimeSequenceEnv, NoiseEnv,
)
from ouroboros.agents.base_agent import BaseAgent
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.compression.mdl import naive_bits
from rich.console import Console
from rich.table import Table

console = Console()

ENVS = [
    ('Binary Repeat',     BinaryRepeatEnv(42),             2),
    ('Modular Arith',     ModularArithmeticEnv(7,3,1,42),  7),
    ('Fibonacci mod 11',  FibonacciModEnv(11, 42),          11),
    ('Prime Sequence',    PrimeSequenceEnv(42),              2),
    ('Noise',             NoiseEnv(4, 42),                   4),
]


def best_ratio(AgentClass, env, alpha, stream_len=600, **kwargs):
    """Run 4 agents, return best compression ratio achieved."""
    env.reset(stream_len)
    stream = env.peek_all()
    best = 1.0
    for i in range(4):
        agent = AgentClass(agent_id=i, alphabet_size=alpha, seed=42+i*7, **kwargs)
        agent.set_history(list(stream))
        agent.search_and_update()
        r = agent.measure_compression_ratio()
        if r < best:
            best = r
    return best


def main():
    console.print("\n[bold]DAY 1 vs DAY 2: N-GRAM vs SYMBOLIC SYNTHESIS[/bold]\n")

    table = Table(title="Compression Ratio: N-gram vs Symbolic")
    table.add_column("Environment",   style="cyan",    width=22)
    table.add_column("Day 1 (n-gram)",style="yellow",  width=15)
    table.add_column("Day 2 (synth)", style="green",   width=15)
    table.add_column("Improvement",                    width=13)
    table.add_column("Note",          style="dim",     width=20)

    for name, env, alpha in ENVS:
        console.print(f"  Testing {name}...", end='')

        ngram_r = best_ratio(
            BaseAgent, env, alpha, stream_len=600,
            max_context_length=6
        )

        env.reset(600)  # Reset stream for fair comparison
        synth_r = best_ratio(
            SynthesisAgent, env, alpha, stream_len=600,
            beam_width=25, max_depth=3, const_range=14,
            use_mcmc=True, mcmc_iters=100
        )

        improvement = ngram_r / synth_r if synth_r > 0 else 1.0
        improvement_str = f"{improvement:.1f}x" if improvement > 1.05 else "~1.0x"

        note = ""
        if 'Modular' in name and improvement > 5:
            note = "← KEY RESULT"
        elif 'Noise' in name and improvement < 1.2:
            note = "✅ sanity check"
        elif 'Prime' in name:
            note = "expected — no rule"

        table.add_row(name, f"{ngram_r:.4f}", f"{synth_r:.4f}",
                      improvement_str, note)
        console.print(f" ngram={ngram_r:.4f} synth={synth_r:.4f}")

    console.print()
    console.print(table)
    console.print()
    console.print("[bold green]Day 2 complete.[/bold green]")
    console.print("Day 3: When 4+ agents agree on the same expression,")
    console.print("       it becomes a proto-axiom. That's where the proof")
    console.print("       market seeds will come from.")


if __name__ == '__main__':
    main()