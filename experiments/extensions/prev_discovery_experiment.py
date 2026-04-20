"""
PREV DISCOVERY EXPERIMENT — Day 13 Core.

8 SynthesisAgents on FibonacciModEnv(11).

Pre-extension: best ratio ~0.40 (no recurrence primitive)
Post-extension: target ratio < 0.01 (PREV enables recurrence discovery)

Expected: at least one agent finds "prev(1) + prev(2)) mod 11"
This is the PREV analog of the Day 2 modular arithmetic landmark.

Run:
    python experiments/extensions/prev_discovery_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.environmentss.structured import FibonacciModEnv, RecurrenceEnv
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.compression.program_synthesis import build_fibonacci_mod
from ouroboros.compression.mdl import naive_bits
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
Path('experiments/extensions').mkdir(parents=True, exist_ok=True)

MODULUS = 11
STREAM_LEN = 1000


def main():
    console.print(Panel.fit(
        "[bold cyan]PREV DISCOVERY EXPERIMENT[/bold cyan]\n"
        f"FibonacciModEnv({MODULUS}) · 8 SynthesisAgents\n"
        "Testing: Do PREV nodes enable recurrence discovery?",
        border_style="bright_cyan"
    ))

    env = FibonacciModEnv(modulus=MODULUS, seed=42)
    env.reset(STREAM_LEN)
    stream = env.peek_all()

    # Show what the target expression looks like
    target = build_fibonacci_mod(MODULUS)
    target_preds = target.predict_sequence(20, MODULUS, initial_history=[0, 1])
    console.print(f"\n  Target expression: {target.to_string()!r}")
    console.print(f"  Target predictions (first 10): {target_preds[:10]}")
    console.print(f"  True stream (first 10):        {stream[:10]}")
    assert target_preds[:10] == stream[:10], "Target expression is wrong"
    console.print("  ✅ Target expression verified\n")

    # Create agents with PREV enabled
    agents = [
        SynthesisAgent(
            agent_id=i,
            alphabet_size=MODULUS,
            beam_width=25,
            max_depth=3,
            const_range=MODULUS * 2,
            mcmc_iterations=200,
            seed=42 + i * 11
        )
        for i in range(8)
    ]

    # Run search
    for agent in agents:
        agent.observe(stream)
        agent.search_and_update()
        agent.measure_compression_ratio()

    # Report results
    table = Table(title="PREV Discovery Results")
    table.add_column("Agent", style="cyan", justify="center")
    table.add_column("Expression", style="yellow")
    table.add_column("Ratio", style="bold", justify="right")
    table.add_column("Has PREV?", justify="center")
    table.add_column("Correct?", justify="center")

    any_found = False
    for agent in agents:
        ratio = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
        expr = agent.best_expression
        expr_str = expr.to_string() if expr else 'none'
        has_prev = expr.has_prev() if expr else False

        # Check correctness
        if expr:
            preds = expr.predict_sequence(100, MODULUS, initial_history=[0, 1] if expr.has_prev() else None)
            correct = preds[:100] == stream[:100]
            if correct:
                any_found = True
        else:
            correct = False

        color = "green" if ratio < 0.05 else ("yellow" if ratio < 0.30 else "dim")
        table.add_row(
            str(agent.agent_id),
            expr_str[:40],
            f"[{color}]{ratio:.4f}[/{color}]",
            "✅" if has_prev else "—",
            "✅" if correct else "—"
        )

    console.print(table)

    # Comparison: ratio before vs after PREV extension
    best_ratio = min(
        a.compression_ratios[-1] for a in agents if a.compression_ratios
    )
    old_best = 0.40   # Pre-extension baseline from Day 1-12

    console.print()
    if any_found:
        console.print(Panel.fit(
            f"[bold green]✅ RECURRENCE DISCOVERY CONFIRMED[/bold green]\n\n"
            f"Expression found: prev(1) + prev(2)) mod {MODULUS}\n"
            f"Best ratio: {best_ratio:.4f} (was ~{old_best} without PREV)\n"
            f"Improvement: {old_best/max(best_ratio,0.001):.0f}×\n\n"
            f"PREV primitive enables recurrence relation discovery.",
            border_style="bright_green"
        ))
    else:
        console.print(
            f"[yellow]Best ratio so far: {best_ratio:.4f} (baseline: {old_best})\n"
            f"PREV nodes are being explored but exact Fibonacci not found yet.\n"
            f"Try increasing beam_width to 35 or mcmc_iterations to 300.[/yellow]"
        )


if __name__ == '__main__':
    main()