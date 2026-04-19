"""
IF DISCOVERY EXPERIMENT — Day 13 second experiment.

8 SynthesisAgents on PiecewiseModEnv(switch_period=10).
Tests: do IF nodes enable piecewise rule discovery?

Run:
    python experiments/extensions/if_discovery_experiment.py
"""

import sys
sys.path.insert(0, '.')
from ouroboros.environment.structured import PiecewiseModEnv
from ouroboros.agents.synthesis_agent import SynthesisAgent
from rich.console import Console

console = Console()


def main():
    console.print("\n[bold cyan]IF DISCOVERY EXPERIMENT[/bold cyan]")
    console.print("PiecewiseModEnv(switch_period=10) · 8 agents\n")

    env = PiecewiseModEnv(switch_period=10, mod1=5, slope1=2,
                          intercept1=1, mod2=7, slope2=3, intercept2=2)
    env.reset(800)
    stream = env.peek_all()

    agents = [
        SynthesisAgent(
            i, alphabet_size=7,
            beam_width=20, max_depth=3, const_range=14,
            mcmc_iterations=150, seed=42 + i * 7
        )
        for i in range(8)
    ]

    for agent in agents:
        agent.observe(stream)
        agent.search_and_update()
        agent.measure_compression_ratio()

    best_ratio = min(
        a.compression_ratios[-1] for a in agents if a.compression_ratios
    )
    any_if = any(
        a.best_expression and a.best_expression.has_prev() is False and
        'IF' in (a.best_expression.to_string() if a.best_expression else '')
        for a in agents
    )

    console.print(f"Best compression ratio: {best_ratio:.4f}")
    console.print(f"IF expressions found: {'✅' if any_if else 'Partial'}")

    for agent in agents:
        r = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
        expr = agent.expression_string()
        console.print(f"  Agent {agent.agent_id}: {expr[:40]!r}  ratio={r:.4f}")


if __name__ == '__main__':
    main()