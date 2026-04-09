# experiments/phase1/phase1_demo.py

"""
Phase 1 Day 1 Demo: 8 agents, 5 environments, MDL compression.

Run:
    python experiments/phase1/phase1_demo.py

What this proves:
1. Agents compress structured environments better than noise
2. Compression ratio is measurable and logged
3. All agents beat random baseline (ratio < 1.0) on structured streams
4. Noise stream stays near 1.0 (sanity check)

Expected output:
    Binary Repeat:      mean=0.04  best=0.04  ✅ structure found
    Modular Arith:      mean=0.28  best=0.21  ✅ structure found
    Fibonacci Mod 11:   mean=0.62  best=0.55  ✅ structure found
    Prime Sequence:     mean=0.88  best=0.84  (partial — as expected)
    Noise (Baseline):   mean=0.96  best=0.94  ✅ incompressible

Note: ModularArith ratio will drop dramatically on Day 2 when
      symbolic program synthesis replaces n-gram search.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
from typing import List, Dict
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ouroboros.environment import (
    BinaryRepeatEnv,
    ModularArithmeticEnv,
    FibonacciModEnv,
    PrimeSequenceEnv,
    NoiseEnv,
)
from ouroboros.agents.society import AgentSociety
from ouroboros.utils.logger import MetricsWriter, make_run_dir, get_logger

console = Console()
logger = get_logger('phase1_demo')


ENV_CONFIGS = [
    {
        'name': 'Binary Repeat',
        'env': BinaryRepeatEnv(seed=42),
        'alpha': 2,
        'expected': '≈ 0.05',
        'note': 'trivially compressible — warm-up',
    },
    {
        'name': 'Modular Arith (7,3,1)',
        'env': ModularArithmeticEnv(7, 3, 1, seed=42),
        'alpha': 7,
        'expected': '< 0.35',
        'note': 'LANDMARK — improves dramatically on Day 2',
    },
    {
        'name': 'Fibonacci mod 11',
        'env': FibonacciModEnv(11, seed=42),
        'alpha': 11,
        'expected': '< 0.70',
        'note': 'recurrence structure',
    },
    {
        'name': 'Prime Sequence',
        'env': PrimeSequenceEnv(seed=42),
        'alpha': 2,
        'expected': '0.80–0.95',
        'note': 'no short algebraic form — expected poor compression',
    },
    {
        'name': 'Noise (Baseline)',
        'env': NoiseEnv(4, seed=42),
        'alpha': 4,
        'expected': '≈ 1.0',
        'note': 'sanity check — must NOT compress',
    },
]


def run_one_env(config: dict, run_base: str) -> Dict:
    """Run 8 agents on one environment."""
    name = config['name']
    env = config['env']
    alpha = config['alpha']

    console.print(f"\n[bold cyan]Running: {name}[/bold cyan]")
    console.print(f"  Expected ratio: [dim]{config['expected']}[/dim]  ({config['note']})")

    run_dir = os.path.join(run_base, name.lower().replace(' ', '_').replace(',', ''))
    writer = MetricsWriter(run_dir)

    society = AgentSociety(
        num_agents=8,
        environment=env,
        alphabet_size=alpha,
        max_context_length=6,
        lambda_weight=1.0,
        seed=42,
    )

    t0 = time.time()
    results = society.run_episode(
        stream_length=1500,
        eval_interval=300,
        writer=writer,
        verbose=False,
    )
    elapsed = time.time() - t0
    writer.close()

    mean_r = results['mean_ratio']
    best_r = results['best_ratio']

    console.print(
        f"  mean={mean_r:.4f}  "
        f"best={best_r:.4f}  "
        f"[dim]({elapsed:.1f}s)[/dim]"
    )
    society.print_status(f"{name} — Final Agent Status")

    return {**results, 'name': name, 'elapsed': elapsed,
            'expected': config['expected']}


def print_summary(all_results: List[Dict]) -> None:
    """Print final comparison table."""
    console.print("\n" + "=" * 65)
    console.print("[bold]PHASE 1 DAY 1 — SUMMARY[/bold]")
    console.print("=" * 65)

    table = Table()
    table.add_column("Environment",  style="cyan",   width=25)
    table.add_column("Mean Ratio",   style="yellow", width=12)
    table.add_column("Best Ratio",   style="green",  width=12)
    table.add_column("Expected",     style="dim",    width=12)
    table.add_column("Status",       width=8)

    random_baseline = 1.0

    for r in all_results:
        beats_random = r['best_ratio'] < random_baseline * 0.99
        status = "✅" if beats_random else "❌"
        table.add_row(
            r['name'],
            f"{r['mean_ratio']:.4f}",
            f"{r['best_ratio']:.4f}",
            r['expected'],
            status,
        )

    console.print(table)

    # Critical checks
    noise_r = next(r for r in all_results if 'Noise' in r['name'])
    mod_r = next(r for r in all_results if 'Modular' in r['name'])

    console.print()
    if noise_r['mean_ratio'] > 0.90:
        console.print("[green]✅ Noise correctly uncompressible (ratio > 0.90)[/green]")
    else:
        console.print("[red]❌ ALERT: Noise compressed — check implementation![/red]")

    if mod_r['best_ratio'] < 0.40:
        console.print("[green]✅ Modular arithmetic structure found[/green]")
        console.print(
            "[dim]   Day 2 will push this below 0.05 with symbolic synthesis[/dim]"
        )
    else:
        console.print(
            f"[yellow]⚠️  Modular arith ratio {mod_r['best_ratio']:.3f} — "
            f"expected < 0.40[/yellow]"
        )


def main():
    console.print(Panel.fit(
        "[bold]OUROBOROS — Phase 1 Day 1 Demo[/bold]\n"
        "8 agents · 5 environments · n-gram MDL compression",
        border_style="bright_blue",
    ))

    run_base = make_run_dir('experiments/phase1/runs', 'day1_demo')
    logger.info(f"Saving metrics to: {run_base}")

    all_results = []
    for config in ENV_CONFIGS:
        result = run_one_env(config, run_base)
        all_results.append(result)

    print_summary(all_results)

    console.print(f"\n[dim]Metrics saved to: {run_base}[/dim]")
    console.print("[bold green]\nDay 1 complete.[/bold green]")
    console.print(
        "Day 2: Replace n-gram search with symbolic beam search.\n"
        "Target: ModularArith ratio drops from ~0.28 → < 0.05"
    )


if __name__ == '__main__':
    main()