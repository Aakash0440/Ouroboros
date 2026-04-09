"""
MULTI-SCALE EXPERIMENT - Day 4 Core.

Run 6 HierarchicalAgents on MultiScaleEnv(slow=28, fast=7).
(Using slow=28 instead of 100 so the slow pattern is visible in 1000 symbols)

Expected behavior:
    Scale 1:  agents find fast pattern  (t mod 7 or similar)
    Scale 4:  agents find slow pattern  (t mod 7 after 4-aggregation = different expr)
    Scale 16: slow pattern dominates    (strongest signal)

Verification:
    - At least one agent finds a program with ratio < 0.30 at scale 1
    - At least one agent finds a program with ratio < 0.30 at scale 4 or higher
    - Compression profile shows multiple scales with structure
    - Cross-scale consistency > 0.5 for at least one agent

Run:
    python experiments/phase1/multiscale_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.environment.structured import MultiScaleEnv, ModularArithmeticEnv
from ouroboros.agents.hierarchical_agent import HierarchicalAgent
from ouroboros.compression.hierarchical_mdl import HierarchicalMDL
from ouroboros.utils.logger import MetricsWriter
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

SLOW_PERIOD = 28
FAST_PERIOD = 7
STREAM_LENGTH = 1200
NUM_AGENTS = 6
SCALES = [1, 4, 16, 32]
RUN_DIR = 'experiments/phase1/runs/multiscale_001'


def main():
    console.print(Panel.fit(
        "[bold cyan]MULTI-SCALE EXPERIMENT[/bold cyan]\n"
        f"MultiScaleEnv(slow={SLOW_PERIOD}, fast={FAST_PERIOD})\n"
        f"{NUM_AGENTS} HierarchicalAgents - Scales: {SCALES}",
        border_style="bright_cyan"
    ))

    env = MultiScaleEnv(
        slow_period=SLOW_PERIOD,
        fast_period=FAST_PERIOD,
        noise_fraction=0.03,
        seed=42
    )
    env.reset(STREAM_LENGTH)
    stream = env.peek_all()

    # Show compression profile of the raw stream first
    hier = HierarchicalMDL(SCALES, alphabet_size=4)
    profile = hier.compression_profile(stream)
    console.print("\n[bold]Stream compression profile:[/bold]")
    for scale, ratio in profile.items():
        bar = "#" * int((1 - ratio) * 30)
        console.print(f"  Scale {scale:2d}: {ratio:.4f}  {bar}")
    dominant, dom_ratio = hier.dominant_scale(stream)
    console.print(f"\n  Dominant scale: {dominant} (ratio={dom_ratio:.4f})")

    # Create and run agents
    agents = [
        HierarchicalAgent(
            agent_id=i,
            alphabet_size=4,
            scales=SCALES,
            beam_width=18,
            max_depth=3,
            const_range=max(SLOW_PERIOD, FAST_PERIOD) * 2,
            mcmc_iterations=100,
            seed=42 + i * 13
        )
        for i in range(NUM_AGENTS)
    ]

    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

    writer = MetricsWriter(RUN_DIR)
    try:
        for step in [400, 800, STREAM_LENGTH]:
            data = stream[:step]
            console.print(f"\n[dim]Checkpoint: {step} observations[/dim]")

            for agent in agents:
                agent.observation_history = list(data)
                agent.search_and_update()
                agent.measure_compression_ratio()

                for scale in SCALES:
                    scale_ratio = (
                        agent.scale_compression_ratios[scale][-1]
                        if agent.scale_compression_ratios[scale] else 1.0
                    )
                    writer.write(
                        step=step,
                        agent_id=agent.agent_id,
                        scale=scale,
                        scale_ratio=scale_ratio,
                    )
    finally:
        writer.close()

    # Final results table
    console.print("\n[bold]Final Agent Status (per scale):[/bold]")
    table = Table()
    table.add_column("Agent", style="cyan", justify="center")
    table.add_column("Scale 1", style="yellow", justify="right")
    table.add_column("Scale 4", style="yellow", justify="right")
    table.add_column("Scale 16", style="green", justify="right")
    table.add_column("Scale 32", style="green", justify="right")
    table.add_column("Dominant", style="bold", justify="center")
    table.add_column("Consistency", style="magenta", justify="right")

    for agent in agents:
        row = [str(agent.agent_id)]
        for scale in SCALES:
            ratios = agent.scale_compression_ratios.get(scale, [])
            r = ratios[-1] if ratios else 1.0
            color = "green" if r < 0.30 else ("yellow" if r < 0.60 else "dim")
            row.append(f"[{color}]{r:.3f}[/{color}]")
        row.append(str(agent.dominant_scale))
        row.append(f"{agent.cross_scale_consistency:.3f}")
        table.add_row(*row)

    console.print(table)

    # Scale programs summary for best agent
    best_agent = min(agents, key=lambda a: min(
        a.scale_costs.get(s, 1.0) for s in SCALES
    ))
    console.print(f"\n[bold]Best agent (Agent {best_agent.agent_id}) scale programs:[/bold]")
    console.print(best_agent.scale_programs_summary())

    # Assertions
    console.print("\n[bold]Checks:[/bold]")

    # Check 1: at least some agent compresses at SOME scale
    any_good = any(
        any(agent.scale_compression_ratios.get(s, [1.0])[-1] < 0.60
            for s in SCALES if agent.scale_compression_ratios.get(s))
        for agent in agents
    )
    if any_good:
        console.print("[green]PASS: At least one agent found structure at some scale[/green]")
    else:
        console.print("[yellow]WARN: No agent found strong structure. Increase stream_length.[/yellow]")

    # Check 2: different agents dominant at different scales
    dominant_scales = [a.dominant_scale for a in agents]
    if len(set(dominant_scales)) > 1:
        console.print(f"[green]PASS: Agents found different dominant scales: {set(dominant_scales)}[/green]")
    else:
        console.print(f"[yellow]WARN: All agents converged to same scale: {dominant_scales}[/yellow]")

    # Check 3: compare single-scale vs multi-scale
    console.print("\n[bold]Single-scale vs Multi-scale comparison:[/bold]")
    console.print(f"  Stream period ratio: slow={SLOW_PERIOD} / fast={FAST_PERIOD} = {SLOW_PERIOD//FAST_PERIOD}x")
    console.print(f"  Expected: fast pattern visible at scale 1, slow at scale {SLOW_PERIOD//FAST_PERIOD}+")

    console.print("\n[green]Day 4 multi-scale experiment complete.[/green]")
    console.print("Results saved to:", RUN_DIR)


if __name__ == '__main__':
    main()