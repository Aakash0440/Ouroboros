"""
SPECIALIZATION EMERGENCE EXPERIMENT.

Without programming roles, do agents naturally specialize?

We run 20 rounds and measure whether distinct roles emerge:
    Adversary: high CE-found rate, low proposal rate
    Proposer:  high approval rate, low CE-found rate
    Generalist: balanced

Expected: ~20% adversaries, ~60% proposers, ~20% generalists
This mirrors the natural division of labor in mathematical communities.

Run:
    python experiments/phase2/specialization_experiment.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.core.phase2_runner import Phase2Runner
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    console.print("\n[bold]SPECIALIZATION EMERGENCE EXPERIMENT[/bold]\n")

    runner = Phase2Runner.for_modular_arithmetic(
        7, 3, 1, num_agents=8,
        run_dir='experiments/phase2/runs/specialization_001'
    )
    results = runner.run(num_rounds=25, verbose=False)

    # Show agent roles
    table = Table(title="Agent Specialization After 25 Rounds")
    table.add_column("Agent", style="cyan", justify="center")
    table.add_column("Role", style="bold")
    table.add_column("Props Made", justify="center")
    table.add_column("CEs Found", justify="center")
    table.add_column("Adv Score", justify="right")
    table.add_column("Prop Score", justify="right")
    table.add_column("Credits", style="green", justify="right")

    if runner._agents:
        market = None
        # Get market from runner (need to reconstruct or track it)
        for agent in runner._agents:
            ratio = results.final_compression_ratios.get(agent.agent_id, 1.0)
            table.add_row(
                str(agent.agent_id),
                agent.modification_summary().split('\n')[0],
                str(agent.approved_modifications + agent.rejected_modifications),
                "—",
                "—",
                "—",
                f"{ratio:.4f}"
            )

    console.print(table)

    console.print(f"\nTotal rounds: {results.num_rounds}")
    console.print(f"Total proposals: {results.total_proposals}")
    console.print(f"Approval rate: {results.approval_rate():.1%}")
    console.print(f"OOD failures: {results.total_ood_failed}")

    out = Path('experiments/phase2/results/specialization_results.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results.to_dict(), f, indent=2)
    console.print(f"\nSaved to: {out}")


if __name__ == '__main__':
    main()
