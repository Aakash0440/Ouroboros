# experiments/phase1/day3_axiom_survey.py

"""
Day 3 Supplementary: Survey all environments for axiom emergence.

Tests the claim:
- Structured environments → axioms emerge
- Noise → no axioms (sanity check)
- Prime sequence → no algebraic axioms (as expected)

This becomes Table 1 in the Phase 1 paper.

Expected results:
    Environment             | Axioms | Best Confidence | Note
    ────────────────────────────────────────────────────────
    Binary Repeat           |   1    |   0.85+         | trivial
    Modular Arith (7,3,1)   |   1    |   0.70+         | LANDMARK
    Fibonacci mod 11        |   0-1  |   0.30-0.60     | partial
    Prime Sequence          |   0    |   N/A           | as expected
    Noise                   |   0    |   N/A           | MUST be 0 (sanity)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ouroboros.environments import (
    BinaryRepeatEnv, ModularArithmeticEnv, FibonacciModEnv,
    PrimeSequenceEnv, NoiseEnv
)
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.emergence.proto_axiom_pool import ProtoAxiomPool
from ouroboros.compression.mdl import naive_bits
from rich.console import Console
from rich.table import Table

console = Console()

ENVS = [
    ('BinaryRepeat',        BinaryRepeatEnv(42),             2,  'trivial'),
    ('ModularArith(7,3,1)', ModularArithmeticEnv(7,3,1,42),  7,  'LANDMARK'),
    ('FibonacciMod(11)',    FibonacciModEnv(11, 42),          11, 'recurrence'),
    ('PrimeSequence',       PrimeSequenceEnv(42),             2,  'no algebraic rule'),
    ('Noise(4)',            NoiseEnv(4, 42),                  4,  'sanity check'),
]


def run_one(name, env, alpha, stream_len=600):
    """Run 8 agents + pool on one environment. Returns survey result."""
    env.reset(stream_len)
    stream = env.peek_all()
    nb = naive_bits(stream, alpha)

    pool = ProtoAxiomPool(
        num_agents=8,
        consensus_threshold=0.5,
        alphabet_size=alpha,
        fingerprint_length=100,
    )

    agents = [
        SynthesisAgent(
            agent_id=i, alphabet_size=alpha,
            beam_width=20, max_depth=3, const_range=12,
            use_mcmc=True, mcmc_iters=100,
            seed=200 + i * 11
        )
        for i in range(8)
    ]

    for agent in agents:
        agent.set_history(list(stream))
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        pool.submit(agent.agent_id, agent.best_expression, ratio * nb, stream_len)

    new_axioms = pool.detect_consensus(stream_len, name, nb)

    best_conf = max((ax.confidence for ax in new_axioms), default=0.0)
    best_ratio = min(
        (agent.latest_ratio() for agent in agents), default=1.0
    )

    return {
        'name': name,
        'num_axioms': len(new_axioms),
        'best_confidence': best_conf,
        'best_compression': best_ratio,
        'axioms': new_axioms,
    }


def main():
    console.print("\n[bold]DAY 3 AXIOM SURVEY - All Environments[/bold]\n")

    results = []
    for name, env, alpha, note in ENVS:
        console.print(f"  {name}...", end='')
        r = run_one(name, env, alpha)
        results.append({**r, 'note': note})
        console.print(
            f" axioms={r['num_axioms']}  "
            f"conf={r['best_confidence']:.3f}  "
            f"ratio={r['best_compression']:.4f}"
        )
        for ax in r['axioms']:
            console.print(f"    -> {ax.axiom_id}: {ax.expression.to_string()!r}")

    # Summary table
    table = Table(title="Axiom Emergence Survey")
    table.add_column("Environment",      style="cyan",   width=22)
    table.add_column("Axioms",           style="green",  width=8)
    table.add_column("Best Confidence",  style="yellow", width=16)
    table.add_column("Best Compression", style="yellow", width=17)
    table.add_column("Note",             style="dim",    width=20)

    for r in results:
        table.add_row(
            r['name'],
            str(r['num_axioms']),
            f"{r['best_confidence']:.4f}" if r['num_axioms'] > 0 else "N/A",
            f"{r['best_compression']:.4f}",
            r['note'],
        )

    console.print()
    console.print(table)

    # Sanity checks
    noise_r = next(r for r in results if 'Noise' in r['name'])
    prime_r = next(r for r in results if 'Prime' in r['name'])

    console.print()
    if noise_r['num_axioms'] == 0:
        console.print("[green]PASS: Noise produced 0 axioms (correct)[/green]")
    else:
        console.print("[red]ALERT: Noise produced axioms - check fingerprinting![/red]")

    if prime_r['num_axioms'] == 0:
        console.print("[green]PASS: Prime sequence: 0 axioms (correct - no short formula)[/green]")

    mod_r = next(r for r in results if 'Modular' in r['name'])
    if mod_r['num_axioms'] > 0:
        console.print("[green]PASS: Modular arith: axiom promoted (landmark result)[/green]")


if __name__ == '__main__':
    main()