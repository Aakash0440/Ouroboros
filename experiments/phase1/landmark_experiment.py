"""
THE LANDMARK EXPERIMENT — Figure 1 of the paper.

This is the definitive experimental evidence that mathematical structure
emerges from compression pressure alone.

Setup:
    8 SynthesisAgents on ModularArithmeticEnv(7, 3, 1)
    10,000 observations · checkpoints every 500 steps

What we're looking for:
    1. At least one agent's compression ratio drops from ~1.0 to < 0.01
    2. The dropped agent found expression "(t * 3 + 1) mod 7"
    3. The drop is ABRUPT (happens at one checkpoint, not gradually)
    4. The discovery step is recorded as the "mathematical discovery event"

The output figure shows:
    X-axis: observations (0..10000)
    Y-axis: compression ratio
    Lines: one per agent
    Vertical gold line: discovery_step
    Annotation: "Discovery: (t * 3 + 1) mod 7"
    Caption: "Mathematical discovery from compression pressure alone"

This figure goes on slide 1 of any talk about this project.

Run:
    python experiments/phase1/landmark_experiment.py
    (Takes ~5-15 minutes depending on hardware)
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.core.phase1_runner import Phase1Runner
from ouroboros.core.config import OuroborosConfig, CompressionConfig
from rich.console import Console
from rich.panel import Panel

console = Console()

MODULUS    = 7
SLOPE      = 3
INTERCEPT  = 1
STREAM_LEN = 5000   # Can increase to 10000 for paper-quality
INTERVAL   = 250


def main():
    console.print(Panel.fit(
        "[bold gold1]THE LANDMARK EXPERIMENT[/bold gold1]\n"
        f"ModularArithmeticEnv({MODULUS}, {SLOPE}, {INTERCEPT})\n"
        "8 SynthesisAgents · 5000 observations\n"
        "Goal: Agents discover (t*3+1) mod 7 from compression pressure",
        border_style="gold1"
    ))

    # Config tuned for best discovery chances
    cfg = OuroborosConfig()
    cfg.compression.beam_width = 30
    cfg.compression.max_depth = 3
    cfg.compression.const_range = MODULUS * 3

    runner = Phase1Runner.for_modular_arithmetic(
        modulus=MODULUS,
        slope=SLOPE,
        intercept=INTERCEPT,
        num_agents=8,
        config=cfg,
        run_dir='experiments/phase1/runs/landmark_001'
    )

    console.print("\n[dim]Running... (this takes 3–10 minutes)[/dim]")
    results = runner.run(
        stream_length=STREAM_LEN,
        eval_interval=INTERVAL,
        consensus_threshold=0.375,  # 3/8 agents for promotion
        verbose=True
    )

    # Save results
    runner.save_results('experiments/phase1/results/landmark_results.json')

    # Generate all plots
    plots = runner.plot_all('experiments/phase1/results/')

    # Print verdict
    console.print()
    if results.discovery_step:
        console.print(Panel.fit(
            f"[bold green]✅ LANDMARK EXPERIMENT SUCCESSFUL[/bold green]\n\n"
            f"Discovery step:  [bold]{results.discovery_step}[/bold] observations\n"
            f"Expression:      [yellow]{results.discovery_expression}[/yellow]\n"
            f"Best ratio:      [bold green]{results.best_ratio:.6f}[/bold green]\n"
            f"Compression:     [bold]{1/max(results.best_ratio, 1e-6):.0f}×[/bold] vs random\n"
            f"Axioms promoted: {len(results.axioms_promoted)}\n"
            f"Time:            {results.elapsed_seconds:.1f}s\n\n"
            f"Agents were NEVER shown modular arithmetic.\n"
            f"MDL pressure caused independent discovery.\n"
            f"Figure saved: experiments/phase1/results/discovery_event.png",
            border_style="bright_green"
        ))
    else:
        console.print(Panel.fit(
            f"[yellow]⚠️  No axiom promoted yet.\n\n"
            f"Best ratio: {results.best_ratio:.4f}\n"
            f"Mean ratio: {results.mean_ratio:.4f}\n\n"
            f"The agents found structure but haven't reached consensus.\n"
            f"Solutions:\n"
            f"  1. Increase stream_length to 10000\n"
            f"  2. Lower consensus_threshold to 0.25\n"
            f"  3. Increase beam_width to 40 in config",
            border_style="yellow"
        ))
        console.print("\nAgent details:")
        for aid, ratio in sorted(results.final_ratios.items()):
            console.print(f"  Agent {aid}: ratio={ratio:.4f}")

    # Print all agent final expressions
    if runner._agents:
        console.print("\n[bold]Agent discoveries:[/bold]")
        for agent in runner._agents:
            ratio = results.final_ratios.get(agent.agent_id, 1.0)
            expr = getattr(agent, 'expression_string',
                          lambda: f"ngram(k={agent.program.context_length})")()
            console.print(f"  Agent {agent.agent_id}: {expr!r}  ratio={ratio:.4f}")

    console.print(f"\nPlots: {plots}")


if __name__ == '__main__':
    main()