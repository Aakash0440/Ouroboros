"""
Phase2HyperRunner — Phase 2 with hyperparameter self-modification.

Every `hp_check_frequency` rounds, each agent checks whether its
hyperparameters should be updated. If a HyperparameterProposal is
generated, it goes through the HyperparameterMarket, then OOD test.

Two layers of self-modification running in parallel:
    Layer 0 (expression):     agents modify their symbolic programs
    Layer 1 (hyperparameter): agents modify their search parameters

The interaction is the key recursive loop:
    Round 1:   Agents start with HP(beam=25, mcmc=200)
    Round 5:   Agent 3 proposes HP(beam=30) — market approves
    Round 6:   Agent 3 now searches with beam=30 → finds better expression
    Round 10:  Agent 3 proposes HP(mcmc=300) — market approves
    Round 11+: Agent 3 searches with beam=30, mcmc=300 → even better

This is the recursive self-improvement signal.
The HP trajectory over rounds is the "meta-learning curve."

Usage:
    runner = Phase2HyperRunner.for_modular_arithmetic(7, 3, 1)
    results = runner.run_with_hp(num_rounds=30)
    runner.print_hp_evolution()
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from ouroboros.environment.structured import ModularArithmeticEnv
from ouroboros.agents.hyperparameter_agent import (
    HyperparameterAgent, HyperparameterSet, HyperparameterProposal
)
from ouroboros.proof_market.hp_market import HyperparameterMarket
from ouroboros.proof_market.ood_pressure import OODPressureModule
from ouroboros.proof_market.market import ProofMarket
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.utils.logger import MetricsWriter, get_logger


@dataclass
class Phase2HyperResults:
    """Results from a Phase 2 run with hyperparameter self-modification."""
    environment_name: str
    num_agents: int
    num_rounds: int
    per_round_ratios: List[float] = field(default_factory=list)
    per_round_hp_summaries: List[Dict] = field(default_factory=list)
    total_expr_proposals: int = 0
    total_expr_approved: int = 0
    total_hp_proposals: int = 0
    total_hp_approved: int = 0
    final_compression_ratios: Dict[int, float] = field(default_factory=dict)
    final_hyperparameters: Dict[int, Dict] = field(default_factory=dict)
    converged: bool = False
    convergence_round: Optional[int] = None
    elapsed_seconds: float = 0.0

    def hp_improvement_rate(self) -> float:
        return self.total_hp_approved / max(self.total_hp_proposals, 1)

    def to_dict(self) -> Dict:
        return {
            'environment': self.environment_name,
            'num_agents': self.num_agents,
            'num_rounds': self.num_rounds,
            'total_expr_proposals': self.total_expr_proposals,
            'total_expr_approved': self.total_expr_approved,
            'total_hp_proposals': self.total_hp_proposals,
            'total_hp_approved': self.total_hp_approved,
            'hp_improvement_rate': round(self.hp_improvement_rate(), 3),
            'converged': self.converged,
            'convergence_round': self.convergence_round,
            'elapsed_seconds': round(self.elapsed_seconds, 2),
            'final_compression_ratios': {
                str(k): round(v, 4)
                for k, v in self.final_compression_ratios.items()
            },
            'final_hyperparameters': self.final_hyperparameters,
        }


class Phase2HyperRunner:
    """
    Runs Phase 2 experiments with both expression and HP self-modification.

    Args:
        environment: Observation environment
        environment_name: Display name
        num_agents: Number of HyperparameterAgents
        initial_hp: Starting HP for all agents (varied per agent if None)
        hp_check_frequency: Rounds between HP checks
        stream_length_per_round: Symbols per round
        run_dir: Output directory
        seed: Random seed
    """

    def __init__(
        self,
        environment,
        environment_name: str,
        num_agents: int = 6,
        initial_hp: Optional[HyperparameterSet] = None,
        hp_check_frequency: int = 5,
        stream_length_per_round: int = 300,
        run_dir: str = 'experiments/phase2/runs/hyper_unnamed',
        seed: int = 42
    ):
        self.environment = environment
        self.environment_name = environment_name
        self.num_agents = num_agents
        self.initial_hp = initial_hp or HyperparameterSet()
        self.hp_check_frequency = hp_check_frequency
        self.stream_length = stream_length_per_round
        self.run_dir = run_dir
        self.seed = seed
        self.logger = get_logger('Phase2HyperRunner')
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self._agents: List[HyperparameterAgent] = []
        self._results: Optional[Phase2HyperResults] = None

    @classmethod
    def for_modular_arithmetic(
        cls,
        modulus: int = 7,
        slope: int = 3,
        intercept: int = 1,
        num_agents: int = 5,
        run_dir: Optional[str] = None,
        **kwargs
    ) -> 'Phase2HyperRunner':
        env = ModularArithmeticEnv(modulus, slope, intercept, seed=42)
        name = f"ModularArith({modulus},{slope},{intercept})"
        rdir = run_dir or f"experiments/phase2/runs/hyper_{modulus}"
        return cls(env, name, num_agents, run_dir=rdir, **kwargs)

    def _create_agents(self) -> List[HyperparameterAgent]:
        alpha = self.environment.alphabet_size
        agents = []
        for i in range(self.num_agents):
            # Slightly vary initial HP per agent for diversity
            hp = HyperparameterSet(
                beam_width=self.initial_hp.beam_width + (i % 3) * 3,
                mcmc_iterations=self.initial_hp.mcmc_iterations,
                const_range=self.initial_hp.const_range + i * 2,
                max_depth=self.initial_hp.max_depth,
                max_lag=self.initial_hp.max_lag,
            ).clamp()
            agents.append(HyperparameterAgent(
                agent_id=i,
                alphabet_size=alpha,
                initial_hp=hp,
                hp_mod_threshold=5.0,
                hp_eval_stream_length=150,
                hp_mod_frequency=self.hp_check_frequency,
                seed=self.seed + i * 11
            ))
        return agents

    def run_with_hp(
        self,
        num_rounds: int = 30,
        verbose: bool = True
    ) -> Phase2HyperResults:
        """
        Run Phase 2 with both expression and hyperparameter self-modification.
        """
        start = time.time()
        alpha = self.environment.alphabet_size

        self._agents = self._create_agents()
        expr_market = ProofMarket(num_agents=self.num_agents, starting_credit=150.0)
        hp_market = HyperparameterMarket(self._agents, starting_credit=100.0)
        ood_module = OODPressureModule.default_suite(base_alphabet_size=alpha)
        ce_searcher = CounterexampleSearcher(
            alphabet_size=alpha, beam_width=15, max_depth=3,
            mcmc_iterations=80, validity_threshold=0.90
        )

        results = Phase2HyperResults(
            environment_name=self.environment_name,
            num_agents=self.num_agents,
            num_rounds=num_rounds
        )

        Path(self.run_dir).mkdir(parents=True, exist_ok=True)

        with MetricsWriter(self.run_dir) as writer:
            for round_num in range(1, num_rounds + 1):
                self.environment.reset(self.stream_length)
                stream = self.environment.peek_all()

                # All agents observe + search with current HP
                for agent in self._agents:
                    agent.observation_history = list(stream)
                    agent.search_and_update()
                    ratio = agent.measure_compression_ratio()

                    writer.write(
                        step=round_num,
                        agent_id=agent.agent_id,
                        compression_ratio=ratio,
                        beam_width=agent.current_hp.beam_width,
                        mcmc_iterations=agent.current_hp.mcmc_iterations,
                        const_range=agent.current_hp.const_range,
                        hp_approved=agent.hp_approved,
                        expr_approved=agent.approved_modifications,
                    )

                # ── Expression self-modification ──────────────────────────
                for agent in self._agents:
                    if expr_market.current_round is not None:
                        break
                    proposal = agent.generate_proposal(stream)
                    if proposal is None or not proposal.is_improvement():
                        continue

                    results.total_expr_proposals += 1
                    other_ids = [a.agent_id for a in self._agents
                                 if a.agent_id != agent.agent_id]
                    ce_results = {
                        oid: ce_searcher.search(
                            oid, proposal.proposed_expr, proposal.test_sequence
                        )
                        for oid in other_ids
                    }

                    try:
                        approved = expr_market.run_full_round(
                            proposer_id=agent.agent_id,
                            current_expr=proposal.current_expr,
                            proposed_expr=proposal.proposed_expr,
                            test_sequence=proposal.test_sequence,
                            alphabet_size=alpha,
                            adversarial_agents=other_ids,
                            ce_results=ce_results,
                            bounty=8.0
                        )
                        if approved:
                            ood_r = ood_module.test_modification(
                                f"expr_r{round_num}", proposal.current_expr,
                                proposal.proposed_expr
                            )
                            if not ood_r.revoked:
                                agent.apply_approved_modification(proposal, round_num)
                                results.total_expr_approved += 1
                    except Exception as e:
                        self.logger.warning(f"Expr market error: {e}")

                # ── Hyperparameter self-modification ──────────────────────
                for agent in self._agents:
                    hp_proposal = agent.generate_hp_proposal(stream)
                    if hp_proposal is None or not hp_proposal.is_improvement():
                        continue

                    results.total_hp_proposals += 1
                    hp_approved, hp_stats = hp_market.run_hp_round(
                        hp_proposal, bounty=5.0
                    )

                    if hp_approved:
                        # OOD test for HP modification
                        # Test: does proposed HP find better expressions on
                        # never-seen environments?
                        ood_pass = self._ood_test_hp(
                            agent, hp_proposal, ood_module, alpha
                        )
                        if ood_pass:
                            agent.apply_hp_modification(hp_proposal, round_num)
                            results.total_hp_approved += 1
                            if verbose:
                                print(
                                    f"    ✅ HP UPDATED: Agent {agent.agent_id} "
                                    f"{hp_proposal.changed_param} "
                                    f"{hp_proposal.change_direction} "
                                    f"→ {agent.current_hp}"
                                )
                        else:
                            agent.record_hp_rejection(hp_proposal)
                    else:
                        agent.record_hp_rejection(hp_proposal)

                # Track round stats
                ratios = [
                    a.compression_ratios[-1] for a in self._agents
                    if a.compression_ratios
                ]
                mean_ratio = float(np.mean(ratios)) if ratios else 1.0
                results.per_round_ratios.append(mean_ratio)

                hp_summary = {
                    str(a.agent_id): a.current_hp.to_dict()
                    for a in self._agents
                }
                results.per_round_hp_summaries.append(hp_summary)

                # Convergence check
                if not results.converged and ratios:
                    if max(ratios) < 0.10 and len(set(
                        a.best_expression.to_string()
                        for a in self._agents
                        if a.best_expression and a._using_symbolic
                    )) <= 2:
                        results.converged = True
                        results.convergence_round = round_num

                if verbose:
                    best = min(ratios) if ratios else 1.0
                    hp_str = ", ".join(
                        f"A{a.agent_id}:{a.current_hp.beam_width}"
                        for a in self._agents
                    )
                    print(f"  Round {round_num:2d}: "
                          f"best={best:.4f}  "
                          f"hp_approved={results.total_hp_approved}  "
                          f"beams=[{hp_str}]")

        # Final state
        for agent in self._agents:
            r = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
            results.final_compression_ratios[agent.agent_id] = round(r, 4)
            results.final_hyperparameters[str(agent.agent_id)] = (
                agent.current_hp.to_dict()
            )

        results.elapsed_seconds = time.time() - start
        self._results = results
        return results

    def _ood_test_hp(
        self,
        agent: HyperparameterAgent,
        proposal: HyperparameterProposal,
        ood_module: OODPressureModule,
        alpha: int
    ) -> bool:
        """
        Test HP modification on OOD environments.

        Returns True if proposed HP finds better/equal expressions on
        at least 60% of OOD environments compared to current HP.
        """
        pass_count = 0
        total_count = 0

        for env_name, env, env_alpha in ood_module.ood_environments[:3]:
            if env_alpha != alpha:
                continue
            env.reset(200)
            ood_stream = env.peek_all()

            import copy as cp
            from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
            from ouroboros.compression.mcmc_refiner import MCMCRefiner

            # Current HP search
            synth_curr = BeamSearchSynthesizer(
                beam_width=proposal.current_hp.beam_width,
                max_depth=proposal.current_hp.max_depth,
                const_range=proposal.current_hp.const_range,
                alphabet_size=alpha
            )
            _, curr_cost = synth_curr.search(ood_stream[:100])

            # Proposed HP search
            synth_prop = BeamSearchSynthesizer(
                beam_width=proposal.proposed_hp.beam_width,
                max_depth=proposal.proposed_hp.max_depth,
                const_range=proposal.proposed_hp.const_range,
                alphabet_size=alpha
            )
            _, prop_cost = synth_prop.search(ood_stream[:100])

            total_count += 1
            if prop_cost <= curr_cost * 1.05:  # Allow 5% degradation
                pass_count += 1

        if total_count == 0:
            return True  # No applicable OOD envs: approve

        return (pass_count / total_count) >= 0.60

    def save_results(self, path: str) -> None:
        if self._results is None:
            raise RuntimeError("Call run_with_hp() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._results.to_dict(), f, indent=2)

    def print_hp_evolution(self) -> None:
        """Print how each agent's hyperparameters changed over time."""
        from rich.console import Console
        from rich.table import Table
        console = Console()
        table = Table(title="Hyperparameter Evolution")
        table.add_column("Agent", style="cyan")
        table.add_column("Initial HP", style="dim")
        table.add_column("Final HP", style="bold green")
        table.add_column("Changes", justify="center")
        table.add_column("HP Score", justify="right")

        for agent in self._agents:
            initial_hp = HyperparameterSet()
            final_hp = agent.current_hp
            changes = agent.hp_approved
            score = agent.hp_improvement_score()
            changed = initial_hp != final_hp
            table.add_row(
                str(agent.agent_id),
                str(initial_hp),
                f"[{'green' if changed else 'dim'}]{final_hp}[/]",
                str(changes),
                f"{score:.1f}bits"
            )
        console.print(table)
        