"""
ContinuousSynthesisAgent — continuous counterpart to SynthesisAgent.

Wraps ContinuousBeamSearch to provide the same interface as the discrete
SynthesisAgent, allowing AgentSociety to run mixed discrete+continuous
agent societies.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Optional

from ouroboros.continuous.environments import ContinuousEnvironment
from ouroboros.continuous.beam_search import (
    ContinuousBeamSearch, ContinuousBeamConfig, ContinuousCandidate
)
from ouroboros.continuous.expr_nodes import ContinuousExprNode
from ouroboros.continuous.mdl import GaussianMDLResult


@dataclass
class ContinuousDiscovery:
    """Records a single discovery event by a continuous agent."""
    agent_id: str
    environment_name: str
    expression: ContinuousExprNode
    expression_str: str
    mdl_result: GaussianMDLResult
    step: int
    is_axiom_candidate: bool   # R² > threshold


@dataclass
class ContinuousAxiomProposal:
    """A continuous axiom proposed for the ProtoAxiomPool equivalent."""
    expression_str: str
    environment_name: str
    r_squared: float
    compression_ratio: float
    supporting_agents: List[str]
    step: int

    @property
    def confidence(self) -> float:
        """Combined confidence: R² × (1 - compression_ratio)."""
        return self.r_squared * (1.0 - max(0.0, self.compression_ratio))


class ContinuousSynthesisAgent:
    """
    A single continuous synthesis agent.
    
    Observes a ContinuousEnvironment, runs beam search + L-BFGS,
    maintains the best expression found so far, and reports
    when it discovers a high-quality fit (R² > threshold).
    """

    def __init__(
        self,
        agent_id: str,
        config: ContinuousBeamConfig = None,
        axiom_r2_threshold: float = 0.95,
    ):
        self.agent_id = agent_id
        self.cfg = config or ContinuousBeamConfig()
        self.axiom_r2_threshold = axiom_r2_threshold

        self._searcher = ContinuousBeamSearch(self.cfg)
        self._best_expr: Optional[ContinuousExprNode] = None
        self._best_mdl: Optional[GaussianMDLResult] = None
        self._discoveries: List[ContinuousDiscovery] = []
        self._step = 0

    def observe_and_search(
        self,
        env: ContinuousEnvironment,
        stream_length: int = 300,
        verbose: bool = False,
    ) -> Optional[ContinuousDiscovery]:
        """
        Generate a stream from env, run beam search, update best expression.
        Returns a ContinuousDiscovery if a good fit is found (R² > threshold).
        """
        observations = env.generate(stream_length)
        self._step += 1

        # Run beam search
        beam = self._searcher.search(observations, verbose=verbose)
        if not beam:
            return None

        best = beam[0]

        # Update internal best
        if (self._best_mdl is None or
                best.mdl.total_mdl_cost < self._best_mdl.total_mdl_cost):
            self._best_expr = copy.deepcopy(best.expr)
            self._best_mdl = best.mdl

        # Report if this is a good fit
        is_candidate = best.mdl.r_squared >= self.axiom_r2_threshold
        discovery = ContinuousDiscovery(
            agent_id=self.agent_id,
            environment_name=env.name,
            expression=best.expr,
            expression_str=best.expr.to_string(),
            mdl_result=best.mdl,
            step=self._step,
            is_axiom_candidate=is_candidate,
        )
        self._discoveries.append(discovery)

        if verbose and is_candidate:
            print(f"  [{self.agent_id}] AXIOM CANDIDATE at step {self._step}")
            print(f"    Expression: {best.expr.to_string()}")
            print(f"    R²={best.mdl.r_squared:.4f}  MDL={best.mdl.total_mdl_cost:.2f}")

        return discovery

    @property
    def best_expression(self) -> Optional[ContinuousExprNode]:
        return self._best_expr

    @property
    def best_r_squared(self) -> float:
        return self._best_mdl.r_squared if self._best_mdl else 0.0

    @property
    def has_good_fit(self) -> bool:
        return self.best_r_squared >= self.axiom_r2_threshold


class ContinuousAgentSociety:
    """
    A society of ContinuousSynthesisAgents running on the same environment.
    
    Consensus detection: if >= consensus_fraction of agents independently
    find expressions with the same behavioral fingerprint (similar predictions)
    and R² > threshold, that pattern is promoted to a ContinuousAxiomProposal.
    
    Behavioral fingerprint: the first 20 predictions rounded to 2 decimal places.
    Two expressions with the same fingerprint are behaviorally equivalent.
    """

    def __init__(
        self,
        n_agents: int = 6,
        consensus_fraction: float = 0.5,
        axiom_r2_threshold: float = 0.95,
        beam_config: ContinuousBeamConfig = None,
    ):
        self.n_agents = n_agents
        self.consensus_fraction = consensus_fraction
        self.axiom_r2_threshold = axiom_r2_threshold

        cfg = beam_config or ContinuousBeamConfig()
        self.agents = [
            ContinuousSynthesisAgent(
                agent_id=f"CONT_AGENT_{i:02d}",
                config=ContinuousBeamConfig(
                    beam_width=cfg.beam_width,
                    max_depth=cfg.max_depth,
                    lbfgs_iterations=cfg.lbfgs_iterations,
                    lbfgs_top_k=cfg.lbfgs_top_k,
                    enable_lbfgs=cfg.enable_lbfgs,
                    allow_sin=cfg.allow_sin,
                    allow_cos=cfg.allow_cos,
                    allow_exp=cfg.allow_exp,
                    allow_log=cfg.allow_log,
                    random_seed=42 + i * 7,  # different seed per agent
                ),
                axiom_r2_threshold=axiom_r2_threshold,
            )
            for i in range(n_agents)
        ]
        self._promoted_axioms: List[ContinuousAxiomProposal] = []

    def run_round(
        self,
        env: ContinuousEnvironment,
        stream_length: int = 300,
        verbose: bool = False,
    ) -> Optional[ContinuousAxiomProposal]:
        """
        Run one round: all agents observe the environment and search.
        Check for consensus. Return promoted axiom if consensus reached.
        """
        discoveries = []
        for agent in self.agents:
            d = agent.observe_and_search(env, stream_length, verbose=False)
            if d is not None:
                discoveries.append(d)

        # Find candidates with good R²
        good = [d for d in discoveries if d.is_axiom_candidate]
        if not good:
            return None

        # Behavioral fingerprinting: round predictions to 2 decimal places
        def fingerprint(d: ContinuousDiscovery) -> str:
            obs = env.generate(20)
            preds = [d.expression.evaluate(t, obs[:t]) for t in range(20)]
            return str([round(p, 2) for p in preds])

        # Group by fingerprint
        from collections import Counter
        fp_counts: Counter = Counter()
        fp_to_agents: dict = {}
        for d in good:
            fp = fingerprint(d)
            fp_counts[fp] += 1
            fp_to_agents.setdefault(fp, []).append(d.agent_id)

        # Check if any fingerprint reached consensus threshold
        threshold = int(self.n_agents * self.consensus_fraction)
        for fp, count in fp_counts.most_common(1):
            if count >= threshold:
                # Promote!
                supporting_agents = fp_to_agents[fp]
                # Use the best (highest R²) discovery with this fingerprint
                best_d = max(
                    [d for d in good if fingerprint(d) == fp],
                    key=lambda d: d.mdl_result.r_squared,
                    default=good[0],
                )
                proposal = ContinuousAxiomProposal(
                    expression_str=best_d.expression_str,
                    environment_name=env.name,
                    r_squared=best_d.mdl_result.r_squared,
                    compression_ratio=best_d.mdl_result.compression_ratio,
                    supporting_agents=supporting_agents,
                    step=best_d.step,
                )
                self._promoted_axioms.append(proposal)
                if verbose:
                    print(f"\n🎯 CONTINUOUS AXIOM PROMOTED: {proposal.expression_str}")
                    print(f"   R²={proposal.r_squared:.4f}, agents={supporting_agents}")
                return proposal

        return None

    @property
    def promoted_axioms(self) -> List[ContinuousAxiomProposal]:
        return list(self._promoted_axioms)