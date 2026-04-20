"""
CRT Analysis — deep dive into the landmark results.

Generates:
    1. Side-by-side comparison of individual vs joint compression
    2. CRT accuracy over search iterations
    3. Mathematical explanation of what was found

Run after crt_landmark_experiment.py:
    python experiments/phase3/crt_analysis.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.emergence.crt_detector import (
    crt_solution, gcd, verify_crt_structure
)
from ouroboros.compression.program_synthesis import (
    build_linear_modular, BeamSearchSynthesizer
)
from ouroboros.compression.mdl import compression_ratio, naive_bits
from ouroboros.environments.structured import ModularArithmeticEnv
from ouroboros.environments.joint_environment import JointEnvironment
from rich.console import Console
from rich.panel import Panel

console = Console()

MOD1, SLOPE1, INT1 = 7, 3, 1
MOD2, SLOPE2, INT2 = 11, 5, 2


def demonstrate_crt_manually():
    """Show what the CRT joint expression SHOULD look like."""
    console.print("\n[bold]CRT Mathematical Demonstration:[/bold]")

    if gcd(MOD1, MOD2) != 1:
        console.print("[red]gcd != 1, CRT doesn't apply[/red]")
        return

    joint_mod = MOD1 * MOD2
    console.print(f"  gcd({MOD1}, {MOD2}) = {gcd(MOD1, MOD2)} ← coprime ✅")
    console.print(f"  Joint modulus: {MOD1} × {MOD2} = {joint_mod}")

    # Show CRT encoding for first 10 timesteps
    console.print(f"\n  CRT encoding for t=0..9:")
    console.print(f"  {'t':>4} | {'mod7':>6} | {'mod11':>6} | {'joint(CRT)':>12}")
    console.print("  " + "-" * 34)

    for t in range(10):
        a1 = (SLOPE1 * t + INT1) % MOD1
        a2 = (SLOPE2 * t + INT2) % MOD2
        x = crt_solution(a1, MOD1, a2, MOD2)
        console.print(f"  {t:>4} | {a1:>6} | {a2:>6} | {x:>12}")

    console.print(f"\n  The joint sequence is (slope*t + intercept) mod {joint_mod}")
    console.print(f"  where slope and intercept are determined by CRT.")
    console.print(f"\n  An agent that finds this expression has derived CRT.")


def verify_known_crt_expression():
    """Verify that a manually-constructed CRT expression works."""
    console.print("\n[bold]Verifying manually-constructed CRT expression:[/bold]")

    # Build the CRT joint stream
    joint_mod = MOD1 * MOD2
    joint_stream_true = []
    for t in range(200):
        a1 = (SLOPE1 * t + INT1) % MOD1
        a2 = (SLOPE2 * t + INT2) % MOD2
        x = crt_solution(a1, MOD1, a2, MOD2)
        joint_stream_true.append(x)

    # Compress it — is it compressible?
    ratio_true = compression_ratio(joint_stream_true, joint_mod)
    naive_mod = compression_ratio(
        [i % joint_mod for i in range(200)], joint_mod
    )

    console.print(f"  True CRT joint stream: ratio={ratio_true:.4f}")
    console.print(f"  Linear (t mod {joint_mod}) stream: ratio={naive_mod:.4f}")
    console.print(f"  Random stream: ratio≈0.97")

    # Try to find the CRT expression by beam search
    console.print(f"\n  Searching for CRT expression (beam_width=30)...")
    synth = BeamSearchSynthesizer(
        beam_width=30, max_depth=3,
        const_range=joint_mod * 2,
        alphabet_size=joint_mod
    )
    expr, cost = synth.search(joint_stream_true[:200], verbose=False)
    nb = naive_bits(joint_stream_true[:200], joint_mod)
    found_ratio = cost / nb if nb > 0 else 1.0

    console.print(f"  Best expression found: {expr.to_string()!r}")
    console.print(f"  Compression ratio:     {found_ratio:.4f}")

    # Check if it's a CRT expression
    preds = expr.predict_sequence(100, joint_mod)
    mod1_acc = sum(
        preds[t] % MOD1 == (SLOPE1*t+INT1) % MOD1
        for t in range(100)
    ) / 100
    mod2_acc = sum(
        preds[t] % MOD2 == (SLOPE2*t+INT2) % MOD2
        for t in range(100)
    ) / 100

    console.print(f"\n  CRT verification:")
    console.print(f"    mod-{MOD1} accuracy: {mod1_acc:.1%}")
    console.print(f"    mod-{MOD2} accuracy: {mod2_acc:.1%}")

    if mod1_acc > 0.90 and mod2_acc > 0.90:
        console.print(f"\n  [bold green]✅ CRT expression FOUND by beam search![/bold green]")
        console.print(f"  This is the expression that SHOULD emerge from compression.")
    else:
        console.print(f"\n  [yellow]⚠️  CRT not fully captured. Partial structure found.[/yellow]")
        console.print(f"  Increase beam_width or search iterations for full recovery.")


def main():
    console.print(Panel.fit(
        "[bold]CRT ANALYSIS[/bold]\n"
        "Deep dive into landmark experiment results",
        border_style="bright_cyan"
    ))

    demonstrate_crt_manually()
    verify_known_crt_expression()

    # Load and display experiment results if they exist
    results_path = Path('experiments/phase3/results/crt_landmark_results.json')
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        console.print("\n[bold]Landmark Experiment Results:[/bold]")
        p2 = results.get('phase2', {})
        if p2:
            console.print(f"  CRT found: {p2.get('found_crt', False)}")
            console.print(f"  Best CRT accuracy: {p2.get('best_crt_accuracy', 0):.1%}")
            console.print(f"  Best expression: {p2.get('best_expression', 'N/A')}")


if __name__ == '__main__':
    main()