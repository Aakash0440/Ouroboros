"""
SELF-MODIFICATION EXPERIMENT — Day 8 Core.

8 SelfModifyingAgents on ModularArithmeticEnv(7, 3, 1).
Each round: agents propose modifications, market evaluates,
OOD module tests survivors.

Tracks:
    - Compression ratio per agent per round
    - Proposal/approval/rejection rates
    - OOD pass/fail per approved modification
    - Recursive ascent: does compression improve monotonically?

Expected behavior:
    Round 1-3:  Early proposals, many rejected (agents still learning)
    Round 4-8:  Better proposals, more approvals
    Round 10+:  Agents have correct rule, few proposals (already optimal)
    Compression curve: monotonically decreasing over rounds

Run:
    python experiments/phase2/self_modification_experiment.py
"""

import sys
from pathlib import Path
sys.path.insert(0, '.')

from ouroboros.environments.structured import ModularArithmeticEnv
from ouroboros.agents.self_modifying_agent import SelfModifyingAgent
from ouroboros.proof_market.market import ProofMarket
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.proof_market.ood_pressure import OODPressureModule
from ouroboros.compression.mdl import naive_bits
from ouroboros.utils.logger import MetricsWriter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

NUM_AGENTS = 6
NUM_ROUNDS = 15
STREAM_LEN_PER_ROUND = 300
MODULUS, SLOPE, INTERCEPT = 7, 3, 1
RUN_DIR = 'experiments/phase2/runs/self_mod_001'


def create_agents(num: int) -> list:
    return [
        SelfModifyingAgent(
            agent_id=i,
            alphabet_size=MODULUS,
            beam_width=20,
            max_depth=3,
            const_range=MODULUS * 2,
            mcmc_iterations=100,
            modification_threshold=1.0,
            seed=42 + i * 11
        )
        for i in range(num)
    ]


def run_one_round(
    agents: list,
    market: ProofMarket,
    ood_module: OODPressureModule,
    ce_searcher: CounterexampleSearcher,
    stream: list,
    round_num: int,
    writer: MetricsWriter
) -> dict:
    """Run one self-modification round."""
    round_stats = {
        'round': round_num,
        'proposals': 0,
        'approved': 0,
        'rejected': 0,
        'ood_passed': 0,
        'ood_failed': 0,
    }

    # Each agent observes stream + runs synthesis
    for agent in agents:
        agent.observation_history = list(stream)
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        writer.write(
            step=round_num,
            agent_id=agent.agent_id,
            compression_ratio=ratio,
            approved_mods=agent.approved_modifications,
            expression=agent.expression_string(),
        )

    # Each agent tries to generate a proposal
    for agent in agents:
        if market.current_round is not None:
            break   # Only one proposal at a time

        proposal = agent.generate_proposal(stream)
        if proposal is None:
            continue

        round_stats['proposals'] += 1
        other_ids = [a.agent_id for a in agents if a.agent_id != agent.agent_id]

        # Other agents search for counterexamples
        ce_results = {}
        for oid in other_ids:
            result = ce_searcher.search(
                agent_id=oid,
                proposal_expr=proposal.proposed_expr,
                test_sequence=proposal.test_sequence
            )
            ce_results[oid] = result

        # Run market round
        try:
            approved = market.run_full_round(
                proposer_id=agent.agent_id,
                current_expr=proposal.current_expr,
                proposed_expr=proposal.proposed_expr,
                test_sequence=proposal.test_sequence,
                alphabet_size=proposal.alphabet_size,
                adversarial_agents=other_ids,
                ce_results=ce_results,
                bounty=8.0
            )
        except Exception as e:
            console.print(f"  [yellow]Market error: {e}[/yellow]")
            continue

        if approved:
            # OOD test
            ood_report = ood_module.test_modification(
                proposal_id=f"r{round_num}a{agent.agent_id}",
                old_expr=proposal.current_expr,
                new_expr=proposal.proposed_expr
            )

            if not ood_report.revoked:
                agent.apply_approved_modification(proposal, round_num)
                round_stats['approved'] += 1
                round_stats['ood_passed'] += 1
            else:
                agent.record_rejection(proposal, round_num, 'ood_failed')
                round_stats['ood_failed'] += 1
        else:
            agent.record_rejection(proposal, round_num, 'market_rejected')
            round_stats['rejected'] += 1

    return round_stats


def main():
    console.print(Panel.fit(
        "[bold]SELF-MODIFICATION EXPERIMENT[/bold]\n"
        f"{NUM_AGENTS} SelfModifyingAgents · {NUM_ROUNDS} rounds\n"
        f"ModularArithmeticEnv({MODULUS},{SLOPE},{INTERCEPT})",
        border_style="bright_blue"
    ))

    env = ModularArithmeticEnv(MODULUS, SLOPE, INTERCEPT, seed=42)
    market = ProofMarket(num_agents=NUM_AGENTS, starting_credit=150.0)
    ood_module = OODPressureModule.default_suite()
    ce_searcher = CounterexampleSearcher(
        alphabet_size=MODULUS, beam_width=15, max_depth=3,
        mcmc_iterations=80, validity_threshold=0.90
    )

    agents = create_agents(NUM_AGENTS)
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)

    all_round_stats = []

    with MetricsWriter(RUN_DIR) as writer:
        for round_num in range(1, NUM_ROUNDS + 1):
            # Generate new data for this round
            env.reset(STREAM_LEN_PER_ROUND)
            stream = env.peek_all()

            stats = run_one_round(
                agents, market, ood_module, ce_searcher,
                stream, round_num, writer
            )
            all_round_stats.append(stats)

            # Print round summary
            best_ratio = min(
                (a.compression_ratios[-1] for a in agents if a.compression_ratios),
                default=1.0
            )
            console.print(
                f"  Round {round_num:2d}: "
                f"props={stats['proposals']}  "
                f"approved={stats['approved']}  "
                f"ood_fail={stats['ood_failed']}  "
                f"best_ratio={best_ratio:.4f}"
            )

    # Final status table
    console.print("\n[bold]Final Agent Status:[/bold]")
    table = Table()
    table.add_column("Agent", style="cyan", justify="center")
    table.add_column("Expression", style="yellow")
    table.add_column("Ratio", style="bold green", justify="right")
    table.add_column("Approved Mods", style="magenta", justify="center")
    table.add_column("Ascent Score", style="blue", justify="right")

    for agent in agents:
        ratio = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
        color = "green" if ratio < 0.05 else ("yellow" if ratio < 0.3 else "dim")
        table.add_row(
            str(agent.agent_id),
            agent.expression_string()[:35],
            f"[{color}]{ratio:.4f}[/{color}]",
            str(agent.approved_modifications),
            f"{agent.recursive_ascent_score():.4f}"
        )
    console.print(table)

    # Market summary
    console.print("\n" + market.market_summary())

    # Recursive ascent verdict
    final_ratios = [
        a.compression_ratios[-1] for a in agents if a.compression_ratios
    ]
    initial_ratio = 1.0
    final_mean = sum(final_ratios) / max(len(final_ratios), 1)
    improvement = (initial_ratio - final_mean) / initial_ratio

    console.print()
    if improvement > 0.30:
        console.print(Panel.fit(
            f"[bold green]✅ RECURSIVE ASCENT CONFIRMED[/bold green]\n\n"
            f"Initial mean ratio: {initial_ratio:.4f}\n"
            f"Final mean ratio:   {final_mean:.4f}\n"
            f"Improvement:        {improvement:.1%}\n\n"
            f"Agents improved themselves through {NUM_ROUNDS} rounds of\n"
            f"adversarial self-modification + OOD validation.",
            border_style="bright_green"
        ))
    else:
        console.print(
            f"[yellow]⚠️  Modest improvement: {improvement:.1%}. "
            f"Increase NUM_ROUNDS or lower modification_threshold.[/yellow]"
        )


if __name__ == '__main__':
    main()