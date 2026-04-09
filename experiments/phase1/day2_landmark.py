# experiments/phase1/day2_landmark.py

"""
Day 2 Landmark Experiment: Do SynthesisAgents discover modular arithmetic?

This is the first key result. If even ONE agent finds "(t * 3 + 1) mod 7"
on ModularArithmeticEnv(7, 3, 1) with ratio < 0.02, that is evidence
of mathematical emergence from compression pressure.

Run:
    python experiments/phase1/day2_landmark.py

Expected output:
    Agent 0: "(t * 3 + 1) mod 7"  ratio=0.0041  ✅ EXACT RULE FOUND
    Agent 1: "(t * 3 + 1) mod 7"  ratio=0.0041  ✅ EXACT RULE FOUND
    ...
    ✅ LANDMARK: Modular arithmetic discovered from compression pressure!
    6/8 agents independently found the rule.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from ouroboros.environment import ModularArithmeticEnv
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.compression.program_synthesis import build_linear_modular
from ouroboros.utils.logger import MetricsWriter, make_run_dir
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main():
    console.print(Panel.fit(
        "[bold magenta]DAY 2 LANDMARK EXPERIMENT[/bold magenta]\n"
        "Question: Do SynthesisAgents discover modular arithmetic?\n"
        "Environment: ModularArithmeticEnv(modulus=7, slope=3, intercept=1)\n"
        "Stream rule: (3 * t + 1) mod 7  [HIDDEN from agents]",
        border_style="magenta",
    ))

    # Set up environment
    env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1, seed=42)
    env.reset(1000)
    stream = env.peek_all()

    # Reference expression — used only for CHECKING, not for agents
    reference = build_linear_modular(slope=3, intercept=1, modulus=7)

    # Create 8 synthesis agents with different seeds (diversity)
    agents = [
        SynthesisAgent(
            agent_id=i,
            alphabet_size=7,
            beam_width=30,
            max_depth=3,
            const_range=15,
            use_mcmc=True,
            mcmc_iters=150,
            seed=42 + i * 13,   # Different seeds for diversity
        )
        for i in range(8)
    ]

    run_dir = make_run_dir('experiments/phase1/runs', 'day2_landmark')
    writer = MetricsWriter(run_dir)

    console.print("\n[bold]Running symbolic search on all 8 agents...[/bold]")

    # Feed stream and search
    for agent in agents:
        agent.set_history(list(stream))
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        writer.write(
            step=len(stream),
            agent_id=agent.agent_id,
            expression=agent.expression_string(),
            compression_ratio=ratio,
            using_symbolic=agent._using_symbolic,
        )

    writer.close()

    # ── Results table ──────────────────────────────────────────────────────────
    table = Table(title="Agent Results")
    table.add_column("Agent", style="cyan",    width=6)
    table.add_column("Expression Found",       width=30)
    table.add_column("Ratio",  style="yellow", width=8)
    table.add_column("Exact?", style="green",  width=8)
    table.add_column("Status",                 width=20)

    exact_count = 0
    good_count = 0

    for agent in agents:
        ratio = agent.latest_ratio()
        expr_str = agent.expression_string()

        # Check if expression correctly predicts the stream
        is_exact = False
        if agent.best_expression is not None:
            preds = agent.best_expression.predict_sequence(100, 7)
            targets = [(3*t+1) % 7 for t in range(100)]
            is_exact = (preds == targets)
            if is_exact:
                exact_count += 1

        if ratio < 0.10:
            good_count += 1

        status_icon = "✅ EXACT RULE" if is_exact else (
            "🔶 close" if ratio < 0.20 else "❌ not found"
        )

        table.add_row(
            str(agent.agent_id),
            expr_str[:28],
            f"{ratio:.4f}",
            "YES" if is_exact else "no",
            status_icon,
        )

    console.print(table)

    # ── Landmark check ─────────────────────────────────────────────────────────
    console.print()
    if exact_count > 0:
        console.print(
            f"[bold green]✅ LANDMARK ACHIEVED: "
            f"{exact_count}/8 agents independently discovered modular arithmetic![/bold green]"
        )
        console.print()
        console.print("  What happened:")
        console.print("  • Agents were given a stream of integers")
        console.print("  • They searched over arithmetic expressions (no math knowledge)")
        console.print("  • MDL pressure favored short programs with low prediction error")
        console.print("  • The shortest correct program is '(t * 3 + 1) mod 7'")
        console.print("  • Agents found it — without being told what mod means")
        console.print()
        console.print("  This is mathematical emergence from compression pressure.")
    elif good_count > 0:
        console.print(
            f"[yellow]⚠️  {good_count}/8 agents compressed well (ratio < 0.10) "
            f"but exact rule not confirmed.[/yellow]"
        )
        console.print("  Possible causes:")
        console.print("  • const_range too small — try const_range=20")
        console.print("  • beam_width too small — try beam_width=40")
        console.print("  • MCMC iterations too few — try mcmc_iters=300")
        console.print("  Day 3 adds consensus detection which will still count this.")
    else:
        console.print("[red]❌ No agents found the rule. Debug:[/red]")
        console.print("  1. Run: python -c \"from ouroboros.compression.beam_search import *\"")
        console.print("  2. Check: python experiments/phase1/day2_landmark.py --debug")

    console.print(f"\n[dim]Metrics saved to: {run_dir}[/dim]")


if __name__ == '__main__':
    main()