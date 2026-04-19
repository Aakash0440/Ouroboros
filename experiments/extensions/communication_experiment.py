"""
COMMUNICATION EXPERIMENT — Day 19 Core.
"""

import sys, json
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.environment.structured import ModularArithmeticEnv
from ouroboros.agents.self_modifying_agent import SelfModifyingAgent
from ouroboros.agents.communicating_agent import CommunicatingAgent
from ouroboros.agents.communication import MessageBus
from ouroboros.proof_market.market import ProofMarket
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.proof_market.ood_pressure import OODPressureModule
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
Path('experiments/extensions').mkdir(parents=True, exist_ok=True)

NUM_AGENTS = 6
NUM_ROUNDS = 20
STREAM_LEN = 300
MODULUS, SLOPE, INTERCEPT = 7, 3, 1


def run_group(group_name: str, agents: list, use_bus: bool = False,
              bus: MessageBus = None) -> dict:
    market = ProofMarket(num_agents=NUM_AGENTS, starting_credit=150.0)
    ood = OODPressureModule.default_suite()
    ce = CounterexampleSearcher(
        alphabet_size=MODULUS, beam_width=15, max_depth=3,
        mcmc_iterations=500, validity_threshold=0.90
    )

    discovery_round = None
    convergence_round = None

    for round_num in range(1, NUM_ROUNDS + 1):
        # FIX: advance seed each round so stream changes and search evolves
        env = ModularArithmeticEnv(MODULUS, SLOPE, INTERCEPT, seed=42 + round_num)
        env.reset(STREAM_LEN)
        stream = env.peek_all()

        if use_bus and bus:
            for agent in agents:
                agent.receive_hints(round_num)

        for agent in agents:
            agent.observation_history = list(stream)
            agent.search_and_update()
            agent.measure_compression_ratio()

        if discovery_round is None:
            for agent in agents:
                r = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
                if r < 0.05:
                    discovery_round = round_num
                    break

        if convergence_round is None:
            ratios = [a.compression_ratios[-1] for a in agents if a.compression_ratios]
            if ratios and max(ratios) < 0.10:
                convergence_round = round_num

        for agent in agents:
            if market.current_round is not None:
                continue
            proposal = agent.generate_proposal(stream)
            if proposal is None:
                continue
            other_ids = [a.agent_id for a in agents if a.agent_id != agent.agent_id]
            ce_results = {
                oid: ce.search(oid, proposal.proposed_expr, proposal.test_sequence)
                for oid in other_ids
            }
            try:
                approved = market.run_full_round(
                    proposer_id=agent.agent_id,
                    current_expr=proposal.current_expr,
                    proposed_expr=proposal.proposed_expr,
                    test_sequence=proposal.test_sequence,
                    alphabet_size=MODULUS,
                    adversarial_agents=other_ids,
                    ce_results=ce_results,
                    bounty=8.0
                )
                if approved:
                    ood_r = ood.test_modification(
                        f"r{round_num}", proposal.current_expr, proposal.proposed_expr
                    )
                    if not ood_r.revoked:
                        agent.apply_approved_modification(proposal, round_num)
            except Exception:
                pass

        if use_bus and bus:
            for agent in agents:
                agent.send_current_hint(round_num)
                agent.send_convergence_signal(round_num)
            bus.advance_round()

    final_ratios = [
        a.compression_ratios[-1] for a in agents if a.compression_ratios
    ]
    return {
        'group': group_name,
        'discovery_round': discovery_round,
        'convergence_round': convergence_round,
        'final_mean_ratio': sum(final_ratios)/len(final_ratios) if final_ratios else 1.0,
        'final_best_ratio': min(final_ratios) if final_ratios else 1.0,
    }


def main():
    console.print(Panel.fit(
        "[bold cyan]COMMUNICATION EXPERIMENT[/bold cyan]\n"
        "Group A: No communication vs Group B: With communication\n"
        f"ModArith({MODULUS},{SLOPE},{INTERCEPT}) · {NUM_ROUNDS} rounds each",
        border_style="bright_cyan"
    ))

    console.print("\n[bold]Group A: No Communication[/bold]")
    agents_a = [
        SelfModifyingAgent(
            i, MODULUS, beam_width=18, max_depth=3, const_range=14,
            mcmc_iterations=80, modification_threshold=0.5, seed=42+i*11
        )
        for i in range(NUM_AGENTS)
    ]
    stats_a = run_group("No Communication", agents_a, use_bus=False)
    console.print(f"  Discovery: round {stats_a['discovery_round'] or 'none'}")
    console.print(f"  Final mean ratio: {stats_a['final_mean_ratio']:.4f}")

    console.print("\n[bold]Group B: With Communication[/bold]")
    bus = MessageBus(num_agents=NUM_AGENTS)
    agents_b = [
        CommunicatingAgent(
            i, MODULUS, message_bus=bus,
            use_hints=True, send_hints=True,
            beam_width=18, max_depth=3, const_range=14,
            mcmc_iterations=80, modification_threshold=0.5, seed=42+i*11
        )
        for i in range(NUM_AGENTS)
    ]
    stats_b = run_group("With Communication", agents_b, use_bus=True, bus=bus)
    console.print(f"  Discovery: round {stats_b['discovery_round'] or 'none'}")
    console.print(f"  Final mean ratio: {stats_b['final_mean_ratio']:.4f}")

    table = Table(title="Communication Experiment Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Group A (No Comm)", style="yellow", justify="right")
    table.add_column("Group B (Comm)", style="green", justify="right")
    table.add_column("Difference", style="bold", justify="right")

    for metric, key, lower_better in [
        ("Discovery Round", "discovery_round", True),
        ("Convergence Round", "convergence_round", True),
        ("Final Mean Ratio", "final_mean_ratio", True),
        ("Final Best Ratio", "final_best_ratio", True),
    ]:
        a_val = stats_a.get(key)
        b_val = stats_b.get(key)
        a_str = str(a_val) if a_val is not None else "—"
        b_str = str(b_val) if b_val is not None else "—"
        if a_val and b_val and isinstance(a_val, (int, float)):
            diff = b_val - a_val
            diff_str = f"{diff:+.4f}" if isinstance(diff, float) else f"{diff:+d}"
            color = "green" if (diff < 0) == lower_better else "red"
            diff_str = f"[{color}]{diff_str}[/{color}]"
        else:
            diff_str = "—"
        table.add_row(metric, a_str, b_str, diff_str)

    console.print(table)

    console.print(f"\n[bold]MessageBus Stats:[/bold]")
    console.print(bus.summary())

    console.print("\n[bold]Agent Communication Stats:[/bold]")
    for agent in agents_b:
        s = agent.communication_stats()
        console.print(
            f"  Agent {s['agent_id']}: sent={s['hints_sent']}  "
            f"recv={s['hints_received']}  used={s['hints_used']}  "
            f"ignored={s['hints_ignored']}"
        )

    disc_a = stats_a['discovery_round'] or NUM_ROUNDS + 1
    disc_b = stats_b['discovery_round'] or NUM_ROUNDS + 1

    if disc_b < disc_a:
        verdict = f"✅ Communication HELPS: Group B discovered {disc_a - disc_b} rounds faster"
    elif disc_b > disc_a:
        verdict = f"⚠️  Communication HURTS: Group A discovered {disc_b - disc_a} rounds faster (herding)"
    else:
        verdict = "→ Communication is NEUTRAL: same discovery round"

    console.print(f"\n[bold]{verdict}[/bold]")

    results = {'group_a': stats_a, 'group_b': stats_b, 'bus_stats': bus.stats()}
    out = Path('experiments/extensions/communication_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    console.print(f"Results saved to: {out}")


if __name__ == '__main__':
    main()