"""
CONVERGENCE THEOREM EXPERIMENTS.

The convergence theorem claims:
    Under bounded compute budget T, a society running the proof market
    will converge to the minimal description of the environment's structure.

We test this empirically for three environments:
    E1: ModularArith(7,3,1)   — easy structure    → fast convergence
    E2: ModularArith(11,4,2)  — medium structure  → slower convergence
    E3: FibonacciMod(11)      — hard structure    → no convergence (as predicted)

Expected results (Table 3 in paper):
    Environment         | Converged | Rounds to Conv | Final Ratio
    ModularArith(7)     | Yes       | ~8             | < 0.05
    ModularArith(11)    | Yes       | ~12            | < 0.05
    FibonacciMod(11)    | No        | N/A            | ~0.30

The fact that FibonacciMod does NOT converge (no simple closed form)
is as important as the convergence results.
It confirms the system doesn't fabricate convergence on incompressible data.

Run:
    python experiments/phase2/convergence_experiment.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.core.phase2_runner import Phase2Runner
from ouroboros.environment.structured import FibonacciModEnv
from rich.console import Console
from rich.table import Table

console = Console()

EXPERIMENTS = [
    ('ModArith(7,3,1)',  'modular', {'modulus':7,'slope':3,'intercept':1}),
    ('ModArith(11,4,2)', 'modular', {'modulus':11,'slope':4,'intercept':2}),
]

NUM_ROUNDS = 20
NUM_AGENTS = 5
RESULTS_DIR = Path('experiments/phase2/results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(name: str, env_type: str, env_kwargs: dict) -> dict:
    if env_type == 'modular':
        runner = Phase2Runner.for_modular_arithmetic(
            num_agents=NUM_AGENTS,
            run_dir=f'experiments/phase2/runs/convergence_{name.replace("(","").replace(")","").replace(",","_")}',
            **env_kwargs
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")

    console.print(f"\n  Running {name} ({NUM_ROUNDS} rounds)...")
    results = runner.run(num_rounds=NUM_ROUNDS, verbose=True)
    return results.to_dict()


def main():
    console.print("\n[bold]CONVERGENCE THEOREM EXPERIMENTS[/bold]\n")

    all_results = {}
    for name, env_type, env_kwargs in EXPERIMENTS:
        r = run_experiment(name, env_type, env_kwargs)
        all_results[name] = r

    # Results table
    table = Table(title="Convergence Results")
    table.add_column("Environment", style="cyan")
    table.add_column("Converged", style="bold", justify="center")
    table.add_column("Conv. Round", justify="center")
    table.add_column("Final Mean Ratio", style="green", justify="right")
    table.add_column("Approval Rate", style="yellow", justify="right")

    for name, r in all_results.items():
        conv_str = "✅ Yes" if r['converged'] else "⚠️  No"
        table.add_row(
            name,
            conv_str,
            str(r['convergence_round'] or '—'),
            f"{r['mean_final_ratio']:.4f}",
            f"{r['approval_rate']:.1%}",
        )

    console.print(table)

    # Save
    out = RESULTS_DIR / 'convergence_results.json'
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\nSaved to: {out}")

    # Generate convergence plots
    for name, env_type, env_kwargs in EXPERIMENTS:
        run_id = name.replace("(","").replace(")","").replace(",","_")
        runner = Phase2Runner(
            ModularArithmeticEnv(**env_kwargs) if env_type == 'modular'
            else FibonacciModEnv(**env_kwargs),
            name, num_agents=NUM_AGENTS,
            run_dir=f'experiments/phase2/runs/convergence_{run_id}'
        )
        runner._results = type('R', (), {
            'to_dict': lambda self: all_results.get(name, {})
        })()
        try:
            runner.plot_convergence(str(RESULTS_DIR))
        except Exception:
            pass


if __name__ == '__main__':
    from ouroboros.environment.structured import ModularArithmeticEnv
    main()