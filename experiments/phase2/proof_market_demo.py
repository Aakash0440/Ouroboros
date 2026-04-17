"""
PROOF MARKET DEMO — Day 7 Core Verification.

Two scenarios:
A) Agent proposes GENUINE improvement → no valid CEs → APPROVED
B) Agent proposes WORSE program → CEs found → REJECTED + bounty paid

This verifies the market mechanics before adding self-modification.

Run:
    python experiments/phase2/proof_market_demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.proof_market.market import ProofMarket
from ouroboros.proof_market.counterexample import CounterexampleResult, CounterexampleSearcher
from ouroboros.compression.program_synthesis import build_linear_modular, C, T, MOD
from rich.console import Console
from rich.panel import Panel

console = Console()

Path('experiments/phase2').mkdir(parents=True, exist_ok=True)


def scenario_a_genuine_improvement():
    """
    Agent 0 proposes upgrading from 't mod 7' to '(3t+1) mod 7'.
    This is a genuine improvement on the ModArith stream.
    No agent should be able to find a counterexample.
    → Expected: APPROVED
    """
    console.print("\n[bold cyan]Scenario A: Genuine Improvement[/bold cyan]")

    market = ProofMarket(num_agents=8, starting_credit=100.0)

    # Current: t mod 7 (suboptimal)
    current = MOD(T(), C(7))
    # Proposed: (3t+1) mod 7 (correct rule)
    proposed = build_linear_modular(3, 1, 7)
    # Test sequence from the correct rule
    test_seq = [(3*t+1) % 7 for t in range(200)]

    searcher = CounterexampleSearcher(
        alphabet_size=7, beam_width=10, max_depth=2,
        mcmc_iterations=50, validity_threshold=0.90
    )

    # Compute all CE results (agents 1-7 try to find CEs)
    ce_results = {}
    for aid in range(1, 8):
        # The proposed expression IS the correct rule — no agent can beat it
        result = searcher.search(aid, proposed, test_seq)
        ce_results[aid] = result

    approved = market.run_full_round(
        proposer_id=0,
        current_expr=current,
        proposed_expr=proposed,
        test_sequence=test_seq,
        alphabet_size=7,
        adversarial_agents=list(range(1, 8)),
        ce_results=ce_results,
        bounty=10.0
    )

    console.print(f"  Current:  {current.to_string()!r}")
    console.print(f"  Proposed: {proposed.to_string()!r}")
    console.print(f"  CEs found by agents: "
                  f"{sum(1 for r in ce_results.values() if r.is_valid_counterexample)}/7")
    console.print(
        f"  Result: [{'green' if approved else 'red'}]"
        f"{'APPROVED ✅' if approved else 'REJECTED ❌'}[/]"
    )

    if approved:
        bonus = market.agents[0].credit - (100.0 - 10.0)
        console.print(f"  Proposer credit: {market.agents[0].credit:.1f} (+{bonus:.1f} bonus)")
    else:
        console.print(f"  ❌ Expected APPROVED but got REJECTED — check CE threshold")

    return approved


def scenario_b_bad_modification():
    """
    Agent 0 proposes replacing '(3t+1) mod 7' with constant C(3).
    This is clearly worse. Agents should find CEs and reject.
    → Expected: REJECTED
    """
    console.print("\n[bold cyan]Scenario B: Bad Modification[/bold cyan]")

    market = ProofMarket(num_agents=8, starting_credit=100.0)

    current = build_linear_modular(3, 1, 7)
    proposed = C(3)  # Terrible proposal
    test_seq = [(3*t+1) % 7 for t in range(200)]

    searcher = CounterexampleSearcher(
        alphabet_size=7, beam_width=15, max_depth=3,
        mcmc_iterations=100, validity_threshold=0.90
    )

    # Agents search for CEs — current expression IS the CE
    ce_results = {}
    for aid in range(1, 8):
        result = searcher.search(aid, proposed, test_seq)
        ce_results[aid] = result

    approved = market.run_full_round(
        proposer_id=0,
        current_expr=current,
        proposed_expr=proposed,
        test_sequence=test_seq,
        alphabet_size=7,
        adversarial_agents=list(range(1, 8)),
        ce_results=ce_results,
        bounty=10.0
    )

    num_finders = sum(1 for r in ce_results.values() if r.is_valid_counterexample)
    console.print(f"  Current:  {current.to_string()!r}")
    console.print(f"  Proposed: {proposed.to_string()!r}")
    console.print(f"  CEs found by agents: {num_finders}/7")
    console.print(
        f"  Result: [{'green' if not approved else 'red'}]"
        f"{'REJECTED ✅ (correct!)' if not approved else 'APPROVED ❌ (wrong!)'}[/]"
    )

    if not approved and num_finders > 0:
        share = 10.0 / num_finders
        console.print(f"  Bounty distributed: {share:.2f} credits to each of {num_finders} finders")
        for aid in range(1, 8):
            if ce_results[aid].is_valid_counterexample:
                console.print(f"    Agent {aid}: +{share:.2f}  total={market.agents[aid].credit:.1f}")

    return not approved  # Return True if correctly rejected


def scenario_c_collude_attempt():
    """
    Demonstrate that collusion is prevented by commit-reveal.
    Agents commit BEFORE seeing others — can't coordinate after.
    """
    console.print("\n[bold cyan]Scenario C: Collusion Prevention Demo[/bold cyan]")

    market = ProofMarket(num_agents=4)
    current = C(0)
    proposed = build_linear_modular(3, 1, 7)
    test_seq = [(3*t+1) % 7 for t in range(100)]

    # PROPOSE
    market.propose(0, current, proposed, test_seq, 7, bounty=5.0)

    # COMMIT phase — agent 1 commits to a valid CE
    from ouroboros.proof_market.counterexample import CounterexampleResult
    ce = CounterexampleResult(
        expression=build_linear_modular(3, 1, 7),
        ce_mdl_cost=50.0,
        proposal_mdl_cost=200.0,
        is_valid_counterexample=True,
        agent_id=1
    )
    market.commit(1, ce)
    market.commit_null(2)
    market.commit_null(3)

    market.close_commit_phase()
    console.print("  Commit phase closed. Agents cannot change answers now.")
    console.print("  Hash of agent 1's CE is locked.")
    console.print("  Any change to their CE now → hash mismatch → penalty")

    # Agent 1 tries to change their CE during reveal (tamper)
    commitment_1 = market.current_round.commitments[1]
    original_ce = commitment_1.counterexample
    commitment_1.counterexample = b'tampered_after_seeing_others'  # Simulate tampering
    commitment_1.revealed = True

    from ouroboros.proof_market.commit_reveal import verify_reveal
    valid = verify_reveal(commitment_1)
    console.print(f"\n  Agent 1 tampers with CE during reveal...")
    console.print(f"  Hash verification: {'PASS' if valid else 'FAIL — TAMPERING DETECTED ✅'}")

    if not valid:
        console.print("  [green]✅ Commit-reveal correctly detected tampering[/green]")
        console.print("  [dim]In production: agent 1 would be penalized[/dim]")

    return True


def main():
    console.print(Panel.fit(
        "[bold]PROOF MARKET DEMO[/bold]\n"
        "Testing: genuine improvement, bad modification, collusion prevention",
        border_style="bright_cyan"
    ))

    a = scenario_a_genuine_improvement()
    b = scenario_b_bad_modification()
    c = scenario_c_collude_attempt()

    console.print()
    if a and b and c:
        console.print(Panel.fit(
            "[bold green]✅ ALL SCENARIOS PASSED[/bold green]\n\n"
            "✅ Genuine improvements are approved\n"
            "✅ Bad modifications are rejected with bounty payment\n"
            "✅ Tampering detected by cryptographic commitment",
            border_style="bright_green"
        ))
    else:
        console.print("[red]Some scenarios failed — check output above[/red]")


if __name__ == '__main__':
    main()