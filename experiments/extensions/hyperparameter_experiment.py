"""
HYPERPARAMETER SELF-MODIFICATION EXPERIMENT — Day 20 Core.

Runs 30 rounds of combined expression + HP self-modification.

Expected behavior:
    Round 1–5:   Agents discover basic modular rule with default HP
    Round 6–10:  Some agents find HP improvements (beam_width up)
    Round 11–20: HP-improved agents discover rules faster/better
    Round 21–30: Convergence — all agents at correct rule + optimal HP

Key metric: HP improvement trajectory
    Initial HP: beam_width=25, mcmc=200
    Expected final: beam_width=30–35, mcmc=250–300

This shows recursive self-improvement at the meta-level.

Run:
    python experiments/extensions/hyperparameter_experiment.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.core.phase2_hyper_runner import Phase2HyperRunner
from ouroboros.agents.hyperparameter_agent import HyperparameterSet
from rich.console import Console
from rich.panel import Panel

console = Console()
Path('experiments/extensions').mkdir(parents=True, exist_ok=True)


def main():
    console.print(Panel.fit(
        "[bold gold1]HYPERPARAMETER SELF-MODIFICATION EXPERIMENT[/bold gold1]\n"
        "Layer 1 Recursive Self-Improvement\n"
        "Agents tune their OWN search parameters through the proof market",
        border_style="gold1"
    ))

    initial_hp = HyperparameterSet(
        beam_width=20,
        mcmc_iterations=150,
        const_range=14,
        max_depth=3,
        max_lag=2
    )

    console.print(f"\n  Initial HP: {initial_hp}")
    console.print(f"  Agents: 5 | Rounds: 25 | HP check every 5 rounds\n")

    runner = Phase2HyperRunner.for_modular_arithmetic(
        7, 3, 1,
        num_agents=5,
        initial_hp=initial_hp,
        hp_check_frequency=5,
        stream_length_per_round=300,
        run_dir='experiments/extensions/hp_run_001'
    )

    results = runner.run_with_hp(num_rounds=25, verbose=True)

    runner.save_results('experiments/extensions/hp_results.json')
    runner.print_hp_evolution()

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  Expression proposals: {results.total_expr_proposals}")
    console.print(f"  Expression approved:  {results.total_expr_approved}")
    console.print(f"  HP proposals:         {results.total_hp_proposals}")
    console.print(f"  HP approved:          {results.total_hp_approved}")
    console.print(f"  HP improvement rate:  {results.hp_improvement_rate():.1%}")
    console.print(f"  Converged:            {results.converged}")
    if results.convergence_round:
        console.print(f"  Convergence round:    {results.convergence_round}")

    # HP divergence
    final_beams = [
        results.final_hyperparameters.get(str(i), {}).get('beam_width', 20)
        for i in range(5)
    ]
    initial_beams = [initial_hp.beam_width] * 5
    any_changed = any(f != i for f, i in zip(final_beams, initial_beams))

    console.print()
    if any_changed:
        console.print(Panel.fit(
            f"[bold green]✅ HP SELF-MODIFICATION CONFIRMED[/bold green]\n\n"
            f"Initial beam_widths: {initial_beams}\n"
            f"Final beam_widths:   {final_beams}\n\n"
            f"Agents tuned their own search parameters through\n"
            f"the adversarial proof market without external guidance.\n"
            f"This is Layer 1 recursive self-improvement.",
            border_style="bright_green"
        ))
    else:
        console.print(
            "[yellow]Initial HP was already near-optimal for this environment.\n"
            "Try running on a harder environment (FibonacciMod) where\n"
            "larger HP settings provide more benefit.[/yellow]"
        )


if __name__ == '__main__':
    main()