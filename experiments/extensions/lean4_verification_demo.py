"""
LEAN4 VERIFICATION DEMO — Day 17 Core.

Demonstrates the full Lean4 bridge workflow:

Case 1: Perfect expression → Lean4 PROVES it (or empirical NONE)
Case 2: Wrong expression → Lean4 REFUTES it (or empirical CE)
Case 3: FormalProofMarket round with Lean4

If Lean4 is not installed, all cases use empirical fallback.
The demo reports which path was taken and why.

Run:
    python experiments/extensions/lean4_verification_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.proof_market.lean4_bridge import (
    Lean4Translator, Lean4Runner, FormalProofMarket, VerificationResult
)
from ouroboros.compression.program_synthesis import (
    build_linear_modular, C, MOD, T
)
from rich.console import Console
from rich.panel import Panel

console = Console()
Path('experiments/extensions').mkdir(parents=True, exist_ok=True)


def demonstrate_translator():
    """Show what expressions look like in Lean4."""
    console.print("\n[bold cyan]LEAN4 TRANSLATION DEMO[/bold cyan]")
    translator = Lean4Translator()

    examples = [
        ("Constant",            C(3)),
        ("Timestep",            T()),
        ("Linear",              MOD(T(), C(7))),
        ("Modular arithmetic",  build_linear_modular(3, 1, 7)),
    ]

    for name, expr in examples:
        lean_str = translator.expr_to_lean4(expr)
        console.print(f"  {name}: {expr.to_string()!r} → [yellow]{lean_str}[/yellow]")

    # Show full verification script
    expr = build_linear_modular(3, 1, 7)
    stream = [(3*t+1)%7 for t in range(8)]
    script = translator.build_verification_script(expr, stream, 7)
    console.print("\n  Full verification script for (3t+1) mod 7:")
    for line in script.split('\n'):
        console.print(f"    [dim]{line}[/dim]")


def demonstrate_runner():
    """Show runner behavior — with or without Lean4."""
    console.print("\n[bold cyan]LEAN4 RUNNER DEMO[/bold cyan]")
    runner = Lean4Runner(lean_executable='lean', timeout_seconds=20)

    available = runner.is_available()
    if available:
        console.print("  [green]✅ Lean4 is installed![/green]")
        console.print("  Running formal verification...")

        translator = Lean4Translator()
        expr = build_linear_modular(3, 1, 7)
        stream = [(3*t+1)%7 for t in range(20)]
        script = translator.build_verification_script(expr, stream, 7)

        report = runner.run_script(script)
        console.print(f"  Result: [bold]{report.result.name}[/bold]")
        console.print(f"  Elapsed: {report.elapsed_seconds:.2f}s")
        if report.lean4_output:
            console.print(f"  Output: {report.lean4_output[:200]}")
    else:
        console.print("  [yellow]⚠️  Lean4 not installed — using empirical fallback[/yellow]")
        console.print("  Install: curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")


def demonstrate_formal_market():
    """Show FormalProofMarket with both cases."""
    console.print("\n[bold cyan]FORMAL PROOF MARKET DEMO[/bold cyan]")

    fpm = FormalProofMarket(num_agents=6, lean_timeout=20)
    test_seq = [(3*t+1)%7 for t in range(100)]

    # Case 1: Good expression
    console.print("\n  Case 1: Proposing correct expression (3t+1) mod 7")
    good_expr = build_linear_modular(3, 1, 7)
    report1 = fpm.verify_formally(good_expr, test_seq, alphabet_size=7)
    status = "✅ APPROVED" if report1.approved else "❌ REJECTED"
    console.print(f"  Result: {status}")
    console.print(f"  Verification: {report1.result.name} via {report1.method}")

    # Case 2: Bad expression
    console.print("\n  Case 2: Proposing wrong expression C(3)")
    bad_expr = C(3)
    report2 = fpm.verify_formally(bad_expr, test_seq, alphabet_size=7)
    status2 = "✅ APPROVED" if report2.approved else "❌ REJECTED"
    console.print(f"  Result: {status2}")
    console.print(f"  Verification: {report2.result.name} via {report2.method}")
    if report2.counterexample:
        console.print(f"  Counterexample: {report2.counterexample}")

    # Stats
    console.print(f"\n  Stats: {fpm.formal_stats()}")

    if report1.approved and not report2.approved:
        console.print(Panel.fit(
            "[bold green]✅ FORMAL PROOF MARKET WORKING[/bold green]\n\n"
            "Correct expression: APPROVED\n"
            "Wrong expression: REJECTED\n\n"
            f"Method: {report1.method}\n"
            f"(Lean4 {'active' if fpm.runner.is_available() else 'not installed — empirical fallback'})",
            border_style="bright_green"
        ))
    else:
        console.print("[yellow]Unexpected results — check output above[/yellow]")


def main():
    console.print(Panel.fit(
        "[bold]LEAN4 VERIFICATION DEMO[/bold]\n"
        "Formal proof market with Lean4 bridge",
        border_style="bright_cyan"
    ))
    demonstrate_translator()
    demonstrate_runner()
    demonstrate_formal_market()


if __name__ == '__main__':
    main()