"""
Phase2FormalRunner — Phase 2 with formal verification.

Extends Phase2Runner with:
    - FormalProofMarket instead of ProofMarket
    - FormalAxiomRegistry instead of KnowledgeBase
    - Verification report logging
    - Lean4-proved modification count

Usage:
    runner = Phase2FormalRunner.for_modular_arithmetic(7, 3, 1)
    results = runner.run_formal(num_rounds=20)
    runner.export_discovered_axioms('lean4_verification/discovered.lean')
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass, field

from ouroboros.core.phase2_runner import Phase2Runner, Phase2Results
from ouroboros.proof_market.lean4_bridge import FormalProofMarket, VerificationResult
from ouroboros.proof_market.counterexample import CounterexampleSearcher
from ouroboros.proof_market.ood_pressure import OODPressureModule
from ouroboros.emergence.formal_axiom_registry import FormalAxiomRegistry
from ouroboros.utils.logger import MetricsWriter, get_logger


@dataclass
class Phase2FormalResults(Phase2Results):
    """Extended results including formal verification stats."""
    lean4_verifications: int = 0
    lean4_proved: int = 0
    lean4_refuted: int = 0
    lean4_timeouts: int = 0
    empirical_verifications: int = 0
    formally_verified_axioms: int = 0

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({
            'lean4_verifications': self.lean4_verifications,
            'lean4_proved': self.lean4_proved,
            'lean4_refuted': self.lean4_refuted,
            'lean4_timeouts': self.lean4_timeouts,
            'empirical_verifications': self.empirical_verifications,
            'formally_verified_axioms': self.formally_verified_axioms,
        })
        return d


class Phase2FormalRunner(Phase2Runner):
    """
    Phase 2 runner with Lean4 formal verification.

    All modifications approved by the proof market are additionally
    verified by Lean4 (or empirical fallback if Lean4 unavailable).

    Only Lean4-verified (or empirically-verified) modifications
    are applied to agent programs.
    """

    def __init__(self, *args, lean_timeout: int = 30,
                 kb_path: str = 'ouroboros_formal.db', **kwargs):
        super().__init__(*args, **kwargs)
        self.lean_timeout = lean_timeout
        self.kb_path = kb_path
        self.logger = get_logger('Phase2FormalRunner')

    @classmethod
    def for_modular_arithmetic(cls, modulus=7, slope=3, intercept=1,
                                num_agents=6, run_dir=None, **kwargs):
        from ouroboros.environments.structured import ModularArithmeticEnv
        env = ModularArithmeticEnv(modulus, slope, intercept, seed=42)
        name = f"ModularArith({modulus},{slope},{intercept})"
        rdir = run_dir or f"experiments/phase2/runs/formal_{modulus}"
        return cls(env, name, num_agents, run_dir=rdir, **kwargs)

    def run_formal(
        self,
        num_rounds: int = 20,
        verbose: bool = True
    ) -> Phase2FormalResults:
        """
        Run Phase 2 with formal verification.
        """
        start = time.time()
        alpha = self.environment.alphabet_size

        self._agents = self._create_agents()
        formal_market = FormalProofMarket(
            num_agents=self.num_agents,
            lean_timeout=self.lean_timeout,
            starting_credit=150.0
        )
        ood_module = OODPressureModule.default_suite(base_alphabet_size=alpha)
        ce_searcher = CounterexampleSearcher(
            alphabet_size=alpha, beam_width=15, max_depth=3,
            mcmc_iterations=80, validity_threshold=0.90
        )
        registry = FormalAxiomRegistry(self.kb_path)

        results = Phase2FormalResults(
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

                for agent in self._agents:
                    if formal_market.market.current_round is not None:
                        break

                    proposal = agent.generate_proposal(stream)
                    if proposal is None:
                        continue

                    results.total_proposals += 1
                    other_ids = [a.agent_id for a in self._agents
                                 if a.agent_id != agent.agent_id]
                    ce_results = {
                        oid: ce_searcher.search(
                            oid, proposal.proposed_expr, proposal.test_sequence
                        )
                        for oid in other_ids
                    }

                    try:
                        approved, formal_report = formal_market.run_formal_round(
                            proposer_id=agent.agent_id,
                            current_expr=proposal.current_expr,
                            proposed_expr=proposal.proposed_expr,
                            test_sequence=proposal.test_sequence,
                            alphabet_size=alpha,
                            adversarial_agents=other_ids,
                            ce_results=ce_results,
                            bounty=8.0
                        )
                    except Exception as e:
                        self.logger.warning(f"Round error: {e}")
                        continue

                    # Track verification stats
                    if formal_report.method == 'lean4':
                        results.lean4_verifications += 1
                        if formal_report.result == VerificationResult.PROVED:
                            results.lean4_proved += 1
                        elif formal_report.result == VerificationResult.REFUTED:
                            results.lean4_refuted += 1
                        elif formal_report.result == VerificationResult.TIMEOUT:
                            results.lean4_timeouts += 1
                    else:
                        results.empirical_verifications += 1

                    if approved:
                        ood_report = ood_module.test_modification(
                            f"r{round_num}a{agent.agent_id}",
                            proposal.current_expr, proposal.proposed_expr
                        )
                        if not ood_report.revoked:
                            agent.apply_approved_modification(proposal, round_num)
                            results.total_approved += 1

                            # Save to formal registry
                            from ouroboros.emergence.proto_axiom_pool import ProtoAxiom
                            fp = tuple(
                                proposal.proposed_expr.evaluate(t) % alpha
                                for t in range(min(100, len(stream)))
                            )
                            from ouroboros.compression.mdl import MDLCost, naive_bits
                            mdl = MDLCost()
                            preds = proposal.proposed_expr.predict_sequence(len(stream), alpha)
                            cost = mdl.total_cost(
                                proposal.proposed_expr.to_bytes(),
                                preds, stream, alpha
                            )
                            nb = naive_bits(stream, alpha)
                            ax_obj = ProtoAxiom(
                                expression=proposal.proposed_expr,
                                supporting_agents=[agent.agent_id],
                                confidence=1.0 - (cost/nb if nb > 0 else 1.0),
                                environment_name=self.environment_name,
                                compression_ratio=cost/nb if nb > 0 else 1.0,
                                discovery_step=round_num,
                                fingerprint=fp
                            )
                            registry.verify_and_register(
                                ax_obj, stream[:50], alpha, self.environment_name
                            )
                        else:
                            results.total_ood_failed += 1
                    else:
                        results.total_rejected += 1

                if verbose:
                    best_ratio = min(
                        (a.compression_ratios[-1] for a in self._agents
                         if a.compression_ratios), default=1.0
                    )
                    print(f"  Round {round_num:2d}: best={best_ratio:.4f}  "
                          f"approved={results.total_approved}  "
                          f"lean4={results.lean4_proved}p/"
                          f"{results.lean4_refuted}r")

        for agent in self._agents:
            r = agent.compression_ratios[-1] if agent.compression_ratios else 1.0
            results.final_compression_ratios[agent.agent_id] = round(r, 4)

        results.elapsed_seconds = time.time() - start
        formally_verified = registry.get_fully_verified_axioms()
        results.formally_verified_axioms = len(formally_verified)

        if verbose:
            print(f"\n{registry.registry_summary()}")
            fstats = formal_market.formal_stats()
            print(f"Lean4 stats: {fstats}")

        self._results = results
        self._registry = registry
        return results

    def export_discovered_axioms(self, output_path: str) -> None:
        """Export formally verified axioms as Lean4 library."""
        if hasattr(self, '_registry'):
            self._registry.export_lean4_library(output_path)
            print(f"Exported axioms to: {output_path}")