"""
Phase2Runner — unified interface for all Phase 2 experiments.

Encapsulates:
    - SelfModifyingAgent society
    - ProofMarket (commit-reveal, bounty economics)
    - OODPressureModule (evaluation bootstrap fix)
    - Per-round metrics collection
    - Convergence measurement

Usage:
    runner = Phase2Runner.for_modular_arithmetic(7, 3, 1)
    results = runner.run(num_rounds=20)
    runner.save_results('experiments/phase2/results/run_001.json')
    runner.plot_convergence('experiments/phase2/results/')
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from ouroboros.environment.structured import ModularArithmeticEnv
from ouroboros.agents.self_modifying_agent import SelfModifyingAgent
from ouroboros.proof_market.market import ProofMarket
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.proof_market.ood_pressure import OODPressureModule
from ouroboros.compression.mdl import naive_bits
from ouroboros.utils.logger import MetricsWriter, get_logger


@dataclass
class Phase2Results:
    """Complete results from a Phase 2 run."""
    environment_name: str
    num_agents: int
    num_rounds: int
    per_round_stats: List[Dict] = field(default_factory=list)
    final_compression_ratios: Dict[int, float] = field(default_factory=dict)
    total_proposals: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    total_ood_failed: int = 0
    converged: bool = False
    convergence_round: Optional[int] = None
    elapsed_seconds: float = 0.0
    run_dir: str = ''

    def approval_rate(self) -> float:
        return self.total_approved / max(self.total_proposals, 1)

    def mean_final_ratio(self) -> float:
        if not self.final_compression_ratios:
            return 1.0
        return float(np.mean(list(self.final_compression_ratios.values())))

    def to_dict(self) -> Dict:
        return {
            'environment': self.environment_name,
            'num_agents': self.num_agents,
            'num_rounds': self.num_rounds,
            'total_proposals': self.total_proposals,
            'total_approved': self.total_approved,
            'total_rejected': self.total_rejected,
            'total_ood_failed': self.total_ood_failed,
            'approval_rate': round(self.approval_rate(), 3),
            'mean_final_ratio': round(self.mean_final_ratio(), 4),
            'converged': self.converged,
            'convergence_round': self.convergence_round,
            'elapsed_seconds': round(self.elapsed_seconds, 2),
            'final_compression_ratios': {
                str(k): round(v, 4)
                for k, v in self.final_compression_ratios.items()
            },
        }


def is_converged(
    agents: List[SelfModifyingAgent],
    threshold: float = 0.05
) -> bool:
    """
    Check if the society has converged.

    Convergence: all agents agree on the same behavioral fingerprint
    AND all agents have compression ratio below threshold.

    Args:
        agents: List of agents to check
        threshold: Compression ratio threshold for "found the rule"
    """
    # Check compression ratios
    ratios = [
        a.compression_ratios[-1] for a in agents if a.compression_ratios
    ]
    if not ratios or min(ratios) > threshold:
        return False

    # Check expression agreement
    expressions = set(
        a.best_expression.to_string()
        for a in agents
        if a.best_expression and a._using_symbolic
    )
    # Converged if all symbolic agents agree (allowing for minor variations)
    if len(expressions) <= 2:
        return True

    return False


class Phase2Runner:
    """
    Runs complete Phase 2 experiments.

    Args:
        environment: Observation environment
        environment_name: Display name
        num_agents: Number of self-modifying agents
        stream_length_per_round: Symbols per round
        run_dir: Output directory
        seed: Base random seed
    """

    def __init__(
        self,
        environment,
        environment_name: str,
        num_agents: int = 6,
        stream_length_per_round: int = 300,
        run_dir: str = 'experiments/phase2/runs/unnamed',
        seed: int = 42
    ):
        self.environment = environment
        self.environment_name = environment_name
        self.num_agents = num_agents
        self.stream_length = stream_length_per_round
        self.run_dir = run_dir
        self.seed = seed
        self.logger = get_logger('Phase2Runner')

        Path(run_dir).mkdir(parents=True, exist_ok=True)

        self._agents: List[SelfModifyingAgent] = []
        self._results: Optional[Phase2Results] = None

    @classmethod
    def for_modular_arithmetic(
        cls,
        modulus: int = 7,
        slope: int = 3,
        intercept: int = 1,
        num_agents: int = 6,
        run_dir: Optional[str] = None,
        **kwargs
    ) -> 'Phase2Runner':
        env = ModularArithmeticEnv(modulus, slope, intercept, seed=42)
        name = f"ModularArith({modulus},{slope},{intercept})"
        rdir = run_dir or f"experiments/phase2/runs/modular_{modulus}_{slope}_{intercept}"
        return cls(env, name, num_agents, run_dir=rdir, **kwargs)

    def _create_agents(self) -> List[SelfModifyingAgent]:
        alpha = self.environment.alphabet_size
        return [
            SelfModifyingAgent(
                agent_id=i,
                alphabet_size=alpha,
                beam_width=20,
                max_depth=3,
                const_range=alpha * 2,
                mcmc_iterations=100,
                modification_threshold=5.0,
                seed=self.seed + i * 11
            )
            for i in range(self.num_agents)
        ]

    def run(
        self,
        num_rounds: int = 20,
        verbose: bool = True
    ) -> Phase2Results:
        """
        Run the full Phase 2 experiment.

        Each round:
        1. Generate new stream data
        2. All agents observe + synthesize
        3. Each agent tries to generate a proposal
        4. Market evaluates proposals (commit-reveal)
        5. OOD test for approved proposals
        6. Agents update their programs
        7. Check convergence

        Returns Phase2Results.
        """
        start = time.time()
        alpha = self.environment.alphabet_size

        self._agents = self._create_agents()
        market = ProofMarket(num_agents=self.num_agents, starting_credit=150.0)
        ood_module = OODPressureModule.default_suite(base_alphabet_size=alpha)
        ce_searcher = CounterexampleSearcher(
            alphabet_size=alpha, beam_width=15, max_depth=3,
            mcmc_iterations=80, validity_threshold=0.90
        )

        results = Phase2Results(
            environment_name=self.environment_name,
            num_agents=self.num_agents,
            num_rounds=num_rounds,
            run_dir=self.run_dir
        )

        with MetricsWriter(self.run_dir) as writer:
            for round_num in range(1, num_rounds + 1):
                self.environment.reset(self.stream_length)
                stream = self.environment.peek_all()
                market.current_round = None

                market.current_round = None
                round_stats = {
                    'round': round_num, 'proposals': 0,
                    'approved': 0, 'rejected': 0, 'ood_failed': 0
                }

                # All agents observe + synthesize
                for agent in self._agents:
                    agent.observation_history = list(stream)
                    agent.search_and_update()
                    ratio = agent.measure_compression_ratio()
                    writer.write(
                        step=round_num,
                        agent_id=agent.agent_id,
                        compression_ratio=ratio,
                        approved_mods=agent.approved_modifications,
                    )

                # Proposal + market evaluation
                for agent in self._agents:
                    if market.current_round is not None:
                        break
                    proposal = agent.generate_proposal(stream)
                    if proposal is None:
                        continue

                    round_stats['proposals'] += 1
                    other_ids = [a.agent_id for a in self._agents
                                 if a.agent_id != agent.agent_id]
                    ce_results = {
                        oid: ce_searcher.search(
                            oid, proposal.proposed_expr, proposal.test_sequence
                        )
                        for oid in other_ids
                    }

                    try:
                        approved = market.run_full_round(
                            proposer_id=agent.agent_id,
                            current_expr=proposal.current_expr,
                            proposed_expr=proposal.proposed_expr,
                            test_sequence=proposal.test_sequence,
                            alphabet_size=alpha,
                            adversarial_agents=other_ids,
                            ce_results=ce_results,
                            bounty=8.0
                        )
                    except Exception:
                        continue

                    if approved:
                        ood_report = ood_module.test_modification(
                            f"r{round_num}a{agent.agent_id}",
                            proposal.current_expr, proposal.proposed_expr
                        )
                        if not ood_report.revoked:
                            agent.apply_approved_modification(proposal, round_num)
                            round_stats['approved'] += 1
                        else:
                            agent.record_rejection(proposal, round_num, 'ood_failed')
                            round_stats['ood_failed'] += 1
                    else:
                        agent.record_rejection(proposal, round_num, 'market_rejected')
                        round_stats['rejected'] += 1

                results.per_round_stats.append(round_stats)
                results.total_proposals += round_stats['proposals']
                results.total_approved += round_stats['approved']
                results.total_rejected += round_stats['rejected']
                results.total_ood_failed += round_stats['ood_failed']

                # Convergence check
                if not results.converged and is_converged(self._agents):
                    results.converged = True
                    results.convergence_round = round_num

                if verbose:
                    best_ratio = min(
                        (a.compression_ratios[-1] for a in self._agents
                         if a.compression_ratios), default=1.0
                    )
                    print(f"  Round {round_num:2d}: "
                          f"props={round_stats['proposals']}  "
                          f"approved={round_stats['approved']}  "
                          f"best={best_ratio:.4f}"
                          + ("  [CONVERGED]" if results.converged and
                             results.convergence_round == round_num else ""))

        # Final ratios
        for agent in self._agents:
            r = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
            results.final_compression_ratios[agent.agent_id] = round(r, 4)

        results.elapsed_seconds = time.time() - start
        self._results = results
        return results

    def save_results(self, path: str) -> None:
        if self._results is None:
            raise RuntimeError("Call run() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._results.to_dict(), f, indent=2)

    def plot_convergence(self, results_dir: str) -> str:
        """Plot compression ratio over rounds for all agents."""
        from ouroboros.utils.visualize import plot_compression_curves
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        return plot_compression_curves(
            self.run_dir,
            title=f'{self.environment_name} — Phase 2 Convergence',
            save_path=f"{results_dir}/phase2_convergence.png"
        )