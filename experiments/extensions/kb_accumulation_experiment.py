"""
KNOWLEDGE BASE ACCUMULATION EXPERIMENT — Day 14 Core.

Runs 5 successive Phase 1 experiments, each saving to the same KB.
Shows that later runs benefit from earlier discoveries.

Expected behavior:
    Run 1: searches from scratch, takes ~N steps to find rule
    Run 2: loads Run 1's axiom as prior, finds rule faster
    Run 3: confirms prior twice, confidence increases
    Run 5: axiom is multi-confirmed (times_confirmed >= 5)

Metrics:
    - Discovery step per run (should decrease)
    - KB axiom count (should increase)
    - Mean confidence of re-confirmed axioms (should increase)

Run:
    python experiments/extensions/kb_accumulation_experiment.py
"""

import sys, os, json
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.core.phase1_runner import Phase1Runner
from ouroboros.core.knowledge_base import KnowledgeBase
from ouroboros.core.config import OuroborosConfig
from rich.console import Console
from rich.table import Table

console = Console()

KB_PATH = 'experiments/extensions/accumulation_test.db'
Path('experiments/extensions').mkdir(parents=True, exist_ok=True)

# Remove old DB to start fresh
if os.path.exists(KB_PATH):
    os.remove(KB_PATH)
    console.print(f"[dim]Removed old KB: {KB_PATH}[/dim]")


def main():
    console.print("\n[bold cyan]KB ACCUMULATION EXPERIMENT[/bold cyan]")
    console.print("5 successive runs on ModArith(7,3,1) — does KB help?\n")

    cfg = OuroborosConfig()
    cfg.compression.beam_width = 20
    cfg.compression.const_range = 14

    run_stats = []

    for run_num in range(1, 6):
        console.print(f"[bold]Run {run_num}/5:[/bold]")

        # Different seeds each run — simulating different researchers
        runner = Phase1Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=5, config=cfg,
            run_dir=f'experiments/extensions/kb_run_{run_num}',
            seed=42 + run_num * 100   # Different seed each run
        )

        results = runner.run_with_kb(
            stream_length=1000,
            eval_interval=200,
            consensus_threshold=0.40,
            kb_path=KB_PATH,
            verbose=False
        )

        # Check KB state after this run
        kb = KnowledgeBase(KB_PATH)
        stats = kb.statistics()
        kb.close()

        run_stats.append({
            'run': run_num,
            'best_ratio': results.best_ratio,
            'discovery_step': results.discovery_step,
            'kb_total_axioms': stats['total_axioms'],
            'kb_multi_confirmed': stats['multi_confirmed_axioms'],
        })

        console.print(
            f"  best_ratio={results.best_ratio:.4f}  "
            f"discovery={results.discovery_step or 'none'}  "
            f"kb_axioms={stats['total_axioms']}  "
            f"confirmed_3x={stats['multi_confirmed_axioms']}"
        )

    # Results table
    table = Table(title="KB Accumulation Results")
    table.add_column("Run", style="cyan", justify="center")
    table.add_column("Best Ratio", style="bold green", justify="right")
    table.add_column("Discovery Step", justify="right")
    table.add_column("KB Axioms", style="yellow", justify="center")
    table.add_column("Confirmed 3x", style="magenta", justify="center")

    for s in run_stats:
        table.add_row(
            str(s['run']),
            f"{s['best_ratio']:.4f}",
            str(s['discovery_step'] or '—'),
            str(s['kb_total_axioms']),
            str(s['kb_multi_confirmed'])
        )

    console.print(table)

    # Final KB state
    kb = KnowledgeBase(KB_PATH)
    console.print("\n[bold]Final Knowledge Base:[/bold]")
    console.print(kb.summary())

    all_axioms = kb.load_all_axioms(min_confidence=0.1)
    for ax in all_axioms[:5]:
        console.print(f"  {ax}")

    kb.close()

    # Verdict
    final_confirmed = run_stats[-1]['kb_multi_confirmed']
    if final_confirmed >= 1:
        console.print(
            f"\n[bold green]✅ KB accumulation working: "
            f"{final_confirmed} axiom(s) confirmed across multiple runs[/bold green]"
        )
    else:
        console.print(
            "[yellow]⚠️  No multi-confirmed axioms yet. "
            "Run more iterations or lower consensus threshold.[/yellow]"
        )


if __name__ == '__main__':
    main()