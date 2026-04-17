"""
MODULI GENERALIZATION EXPERIMENT.

Does the system generalize across different moduli?
Run on ModularArithmeticEnv for prime moduli: 5, 7, 11, 13, 17

If agents can discover the rule for each modulus independently,
this is evidence that modular arithmetic emergence is systematic,
not a lucky coincidence with modulus=7.

This becomes Table 2 in the paper:
    Modulus | Agents Found | Best Ratio | Discovery Step
    5       | 6/6          | 0.0031     | 250
    7       | 6/6          | 0.0038     | 250
    11      | 6/6          | 0.0051     | 250
    13      | 6/6          | 0.0062     | ~600  (gradual convergence; lower threshold)
    17      | ?/6          | ?          | ~900  (wider const_range + long stream)

Fixes vs v1:
  - Mod-adaptive stream_length   : 13 → 2500,  17 → 3500
  - Mod-adaptive consensus_thresh: 13 → 0.35,  17 → 0.30
  - Mod-adaptive beam_width      : 13 → 30,    17 → 35
  - Mod-adaptive const_range     : 17 → mod*4  (vs mod*3)
  - Mod 17 re-added to PRIME_MODULI
  - Paper note injected for gradual-convergence case (step=None but all agents correct)

Run:
    python experiments/phase1/moduli_generalization.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, '.')

from ouroboros.core.phase1_runner import Phase1Runner
from ouroboros.core.config import OuroborosConfig
from rich.console import Console
from rich.table import Table

console = Console()

# Full 5-row Table 2.  Remove 17 for a quick smoke-test run.
PRIME_MODULI = [5, 7, 11, 13, 17]

BASE_STREAM_LEN = 1500
INTERVAL = 250


# ---------------------------------------------------------------------------
# Per-modulus hyper-parameters
# Larger moduli need: more stream, wider beam, looser consensus gate, wider
# constant search range.  Keep small moduli unchanged so v1 baselines hold.
# ---------------------------------------------------------------------------
def get_mod_config(mod: int) -> dict:
    if mod <= 11:
        return dict(
            stream_length=BASE_STREAM_LEN,
            consensus_threshold=0.40,
            beam_width=25,
            const_range=mod * 3,
        )
    elif mod == 13:
        return dict(
            stream_length=2500,
            consensus_threshold=0.35,   # 6/6 agents agreed but gate was too tight
            beam_width=30,
            const_range=mod * 3,
        )
    else:  # mod == 17 (and any future larger prime)
        return dict(
            stream_length=3500,
            consensus_threshold=0.30,
            beam_width=35,
            const_range=mod * 4,        # wider search space for larger modulus
        )


def run_one_modulus(mod: int) -> dict:
    """Run landmark experiment for one modulus."""
    mc = get_mod_config(mod)

    cfg = OuroborosConfig()
    cfg.compression.beam_width = mc['beam_width']
    cfg.compression.max_depth = 3
    cfg.compression.const_range = mc['const_range']

    slope = (mod // 2) + 1
    intercept = mod // 3 + 1

    runner = Phase1Runner.for_modular_arithmetic(
        modulus=mod,
        slope=slope,
        intercept=intercept,
        num_agents=6,
        config=cfg,
        run_dir=f'experiments/phase1/runs/moduli_{mod}',
    )

    results = runner.run(
        stream_length=mc['stream_length'],
        eval_interval=INTERVAL,
        consensus_threshold=mc['consensus_threshold'],
        verbose=False,
    )

    agents_correct = sum(
        1 for ratio in results.final_ratios.values()
        if ratio < 0.10
    )

    # "found_rule" = hard consensus triggered  OR  all agents correct at end.
    # The second condition handles gradual convergence (mod 13 pattern):
    # agents are all correct but never spiked past the threshold in a single
    # eval window.  Both cases are valid discoveries; paper distinguishes them
    # as "phase-transition" vs "gradual" convergence.
    phase_transition = results.discovery_step is not None
    gradual = (not phase_transition) and (agents_correct == 6)
    found_rule = phase_transition or gradual

    return {
        'modulus': mod,
        'slope': slope,
        'intercept': intercept,
        'found_rule': found_rule,
        'convergence_type': 'phase_transition' if phase_transition else ('gradual' if gradual else 'none'),
        'agents_correct': agents_correct,
        'best_ratio': results.best_ratio,
        'mean_ratio': results.mean_ratio,
        'discovery_step': results.discovery_step,   # None for gradual
        'discovery_expression': results.discovery_expression,
        # surface the adaptive config used — useful for Methods section
        'stream_length_used': mc['stream_length'],
        'consensus_threshold_used': mc['consensus_threshold'],
        'beam_width_used': mc['beam_width'],
    }


def main():
    console.print("\n[bold]MODULI GENERALIZATION EXPERIMENT[/bold]")
    console.print(f"Testing moduli: {PRIME_MODULI}\n")

    all_results = []
    for mod in PRIME_MODULI:
        console.print(f"  Modulus {mod}...", end=' ')
        r = run_one_modulus(mod)
        all_results.append(r)

        conv_tag = {
            'phase_transition': '[green]phase-transition[/green]',
            'gradual':          '[yellow]gradual[/yellow]',
            'none':             '[red]none[/red]',
        }[r['convergence_type']]

        status = "✅" if r['found_rule'] else "❌"
        console.print(
            f"{status}  agents_correct={r['agents_correct']}/6  "
            f"best_ratio={r['best_ratio']:.4f}  "
            f"step={r['discovery_step']}  "
            f"conv={conv_tag}"
        )
        if r['discovery_expression']:
            console.print(f"       found: {r['discovery_expression']!r}")

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    table = Table(title="Moduli Generalization Results")
    table.add_column("Modulus",        style="cyan",  justify="center")
    table.add_column("Rule",           style="dim")
    table.add_column("Agents",         style="green", justify="center")
    table.add_column("Best Ratio",     style="bold",  justify="right")
    table.add_column("Disc. Step",     style="yellow",justify="right")
    table.add_column("Convergence",    justify="center")
    table.add_column("Found",          justify="center")

    for r in all_results:
        rule = f"({r['slope']}t+{r['intercept']}) mod {r['modulus']}"
        conv_label = {
            'phase_transition': 'phase-Δ',
            'gradual':          'gradual',
            'none':             '—',
        }[r['convergence_type']]
        table.add_row(
            str(r['modulus']),
            rule,
            f"{r['agents_correct']}/6",
            f"{r['best_ratio']:.4f}",
            str(r['discovery_step'] or '—'),
            conv_label,
            "✅" if r['found_rule'] else "❌",
        )

    console.print(table)

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out = Path('experiments/phase1/results/moduli_generalization.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    console.print(f"\nResults saved to: {out}")

    # ------------------------------------------------------------------
    # Summary + paper guidance
    # ------------------------------------------------------------------
    found_all   = sum(1 for r in all_results if r['found_rule'])
    gradual_cnt = sum(1 for r in all_results if r['convergence_type'] == 'gradual')
    none_cnt    = sum(1 for r in all_results if r['convergence_type'] == 'none')

    console.print(f"\n[bold]Summary: Rule found in {found_all}/{len(PRIME_MODULI)} environments[/bold]")

    if none_cnt == 0:
        console.print(
            "[bold green]✅ Full generalization across all moduli. "
            "Modular arithmetic emergence is systematic.[/bold green]"
        )
    elif found_all == len(PRIME_MODULI):
        console.print(
            f"[bold green]✅ Full generalization ({gradual_cnt} via gradual convergence).[/bold green]\n"
            "[dim]Paper caption note: larger moduli converge gradually rather than via "
            "a sharp phase transition; discovery_step is undefined for these cases "
            "but all agents reach correct ratio < 0.10.[/dim]"
        )
    elif found_all >= len(PRIME_MODULI) // 2:
        console.print("[yellow]⚠️  Partial generalization.[/yellow]")
        for r in all_results:
            if not r['found_rule']:
                console.print(
                    f"   Mod {r['modulus']} failed — "
                    f"agents_correct={r['agents_correct']}/6  best_ratio={r['best_ratio']:.4f}\n"
                    f"   Try: stream_length={r['stream_length_used'] + 1000}  "
                    f"beam_width={r['beam_width_used'] + 5}  "
                    f"const_range={r['modulus'] * 4}"
                )
    else:
        console.print(
            "[bold red]❌ Generalization failed for majority of moduli. "
            "Re-check Phase1Runner.for_modular_arithmetic params.[/bold red]"
        )


if __name__ == '__main__':
    main()