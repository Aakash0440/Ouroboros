"""
Phase3Runner — unified interface for Phase 3 experiments.

Extends Phase2Runner with:
    - TheoryAgents instead of SelfModifyingAgents
    - Theory-level proposals (multi-scale upgrades)
    - Cross-agent theory consistency measurement
    - CRT experiment support (joint theory from two environments)

Usage:
    # Single-environment theory building
    runner = Phase3Runner.for_modular_arithmetic(7, 3, 1)
    results = runner.run(num_rounds=30)

    # CRT experiment — joint theory from two environments
    runner = Phase3Runner.for_crt_experiment(mod1=7, mod2=11)
    crt_results = runner.run_crt(num_rounds=40)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

from ouroboros.environment.structured import ModularArithmeticEnv
from ouroboros.agents.theory_agent import TheoryAgent
from ouroboros.proof_market.market import ProofMarket
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.proof_market.ood_pressure import OODPressureModule
from ouroboros.utils.logger import MetricsWriter, get_logger


@dataclass
class Phase3Results:
    """Results from a Phase 3 experiment."""
    environment_name: str
    num_agents: int
    num_rounds: int
    per_round_richness: List[float] = field(default_factory=list)
    final_theories: List[Dict] = field(default_factory=list)
    mean_final_richness: float = 0.0
    best_final_richness: float = 0.0
    converged: bool = False
    convergence_round: Optional[int] = None
    elapsed_seconds: float = 0.0
    run_dir: str = ''

    def to_dict(self) -> Dict:
        return {
            'environment': self.environment_name,
            'num_agents': self.num_agents,
            'num_rounds': self.num_rounds,
            'mean_final_richness': round(self.mean_final_richness, 4),
            'best_final_richness': round(self.best_final_richness, 4),
            'converged': self.converged,
            'convergence_round': self.convergence_round,
            'elapsed_seconds': round(self.elapsed_seconds, 2),
            'final_theories': self.final_theories,
        }


class Phase3Runner:
    """
    Runs Phase 3 experiments with TheoryAgents.

    Args:
        environment: Primary observation environment
        environment_name: Display name
        num_agents: Number of theory agents
        scales: Theory scales
        stream_length_per_round: Symbols per round
        run_dir: Output directory
        seed: Random seed
    """

    def __init__(
        self,
        environment,
        environment_name: str,
        num_agents: int = 6,
        scales: List[int] = None,
        stream_length_per_round: int = 400,
        run_dir: str = 'experiments/phase3/runs/unnamed',
        seed: int = 42
    ):
        self.environment = environment
        self.environment_name = environment_name
        self.num_agents = num_agents
        self.scales = scales or [1, 4, 16, 32]
        self.stream_length = stream_length_per_round
        self.run_dir = run_dir
        self.seed = seed
        self.logger = get_logger('Phase3Runner')
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self._agents: List[TheoryAgent] = []
        self._results: Optional[Phase3Results] = None

    @classmethod
    def for_modular_arithmetic(
        cls,
        modulus: int = 7,
        slope: int = 3,
        intercept: int = 1,
        num_agents: int = 5,
        scales: List[int] = None,
        run_dir: Optional[str] = None,
        **kwargs
    ) -> 'Phase3Runner':
        env = ModularArithmeticEnv(modulus, slope, intercept, seed=42)
        name = f"ModularArith({modulus},{slope},{intercept})"
        rdir = run_dir or f"experiments/phase3/runs/modular_{modulus}"
        return cls(env, name, num_agents, scales=scales, run_dir=rdir, **kwargs)

    def _create_agents(self) -> List[TheoryAgent]:
        alpha = self.environment.alphabet_size
        return [
            TheoryAgent(
                agent_id=i,
                alphabet_size=alpha,
                scales=self.scales,
                beam_width=18,
                max_depth=3,
                const_range=alpha * 2,
                mcmc_iterations=80,
                seed=self.seed + i * 11
            )
            for i in range(self.num_agents)
        ]

    def run(
        self,
        num_rounds: int = 25,
        verbose: bool = True
    ) -> Phase3Results:
        """Run the full Phase 3 experiment."""
        start = time.time()
        alpha = self.environment.alphabet_size

        self._agents = self._create_agents()
        market = ProofMarket(num_agents=self.num_agents, starting_credit=150.0)
        ood_module = OODPressureModule.default_suite(base_alphabet_size=alpha)
        ce_searcher = CounterexampleSearcher(
            alphabet_size=alpha, beam_width=12, max_depth=3,
            mcmc_iterations=60, validity_threshold=0.90
        )

        results = Phase3Results(
            environment_name=self.environment_name,
            num_agents=self.num_agents,
            num_rounds=num_rounds,
            run_dir=self.run_dir
        )

        Path(self.run_dir).mkdir(parents=True, exist_ok=True)

        with MetricsWriter(self.run_dir) as writer:
            for round_num in range(1, num_rounds + 1):
                self.environment.reset(self.stream_length)
                stream = self.environment.peek_all()

                # All agents observe, search, update theories
                for agent in self._agents:
                    agent.observation_history = list(stream)
                    agent.search_and_update()
                    agent.update_theory_from_search(step=round_num)
                    agent.measure_compression_ratio()

                    writer.write(
                        step=round_num,
                        agent_id=agent.agent_id,
                        compression_ratio=(
                            agent.compression_ratios[-1]
                            if agent.compression_ratios else 1.0
                        ),
                        theory_richness=agent.theory.richness_score(),
                        theory_complete=agent.theory.is_complete(),
                    )

                # Theory proposal + market
                for agent in self._agents:
                    if market.current_round is not None:
                        break
                    theory_proposal = agent.generate_theory_proposal(
                        stream, min_richness_improvement=0.01
                    )
                    if theory_proposal is None:
                        continue

                    mod_proposal = theory_proposal.to_modification_proposal(agent.theory)
                    if mod_proposal is None or not mod_proposal.is_improvement():
                        continue

                    other_ids = [a.agent_id for a in self._agents
                                 if a.agent_id != agent.agent_id]
                    ce_results = {
                        oid: ce_searcher.search(
                            oid, mod_proposal.proposed_expr,
                            mod_proposal.test_sequence
                        )
                        for oid in other_ids
                    }

                    try:
                        approved = market.run_full_round(
                            proposer_id=agent.agent_id,
                            current_expr=mod_proposal.current_expr,
                            proposed_expr=mod_proposal.proposed_expr,
                            test_sequence=mod_proposal.test_sequence,
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
                            mod_proposal.current_expr, mod_proposal.proposed_expr
                        )
                        if not ood_report.revoked:
                            agent.apply_theory_upgrade(theory_proposal, round_num)
                            agent.theory.record_market_result(
                                theory_proposal.primary_scale, survived=True
                            )

                # Track mean richness
                richnesses = [a.theory.richness_score() for a in self._agents]
                mean_rich = float(np.mean(richnesses))
                results.per_round_richness.append(mean_rich)

                if verbose:
                    best_ratio = min(
                        (a.compression_ratios[-1] for a in self._agents
                         if a.compression_ratios), default=1.0
                    )
                    print(f"  Round {round_num:2d}: "
                          f"mean_richness={mean_rich:.4f}  "
                          f"best_ratio={best_ratio:.4f}")

        # Final theories
        results.final_theories = [a.theory.to_dict() for a in self._agents]
        richnesses = [a.theory.richness_score() for a in self._agents]
        results.mean_final_richness = round(float(np.mean(richnesses)), 4)
        results.best_final_richness = round(float(max(richnesses)), 4)
        results.elapsed_seconds = time.time() - start
        self._results = results
        return results

    def save_results(self, path: str) -> None:
        if self._results is None:
            raise RuntimeError("Call run() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._results.to_dict(), f, indent=2)
        print(f"Saved: {path}")