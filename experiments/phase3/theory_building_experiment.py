"""
THEORY BUILDING EXPERIMENT — Day 10 Core.

5 TheoryAgents on ModularArithmeticEnv(7, 3, 1) for 20 rounds.
Tracks theory richness over time.

Expected behavior:
    Round 1–5:   Agents build initial scale-1 axioms
    Round 6–12:  Scale-4 and scale-16 axioms added
    Round 13–20: Cross-scale consistency increases
    Final:       Complete theory, richness > 0.40

Run:
    python experiments/phase3/theory_building_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.core.phase3_runner import Phase3Runner
from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    console.print(Panel.fit(
        "[bold cyan]THEORY BUILDING EXPERIMENT[/bold cyan]\n"
        "5 TheoryAgents · ModularArith(7,3,1) · 20 rounds\n"
        "Measuring: theory richness over time",
        border_style="bright_cyan"
    ))

    runner = Phase3Runner.for_modular_arithmetic(
        7, 3, 1, num_agents=5,
        scales=[1, 4, 16],
        run_dir='experiments/phase3/runs/theory_001'
    )

    results = runner.run(num_rounds=20, verbose=True)
    runner.save_results('experiments/phase3/results/theory_001.json')

    console.print("\n[bold]Final Theory Summary (best agent):[/bold]")
    if runner._agents:
        best_agent = max(runner._agents, key=lambda a: a.theory.richness_score())
        console.print(best_agent.theory_summary())

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Mean final richness: {results.mean_final_richness:.4f}")
    console.print(f"  Best final richness: {results.best_final_richness:.4f}")
    console.print(f"  Richness over rounds: "
                  f"{[round(r, 3) for r in results.per_round_richness[:10]]}...")

    if results.best_final_richness > 0.20:
        console.print("[bold green]✅ Theory building: richness > 0.20[/bold green]")
    else:
        console.print("[yellow]⚠️  Low richness — increase num_rounds or beam_width[/yellow]")


if __name__ == '__main__':
    main()