"""
CommunicationExperiment — The definitive A/B test of agent communication.

Research question: Does sharing mathematical discoveries between agents
accelerate convergence or cause herding (everyone explores the same region)?

Design:
  Condition A (SOLO): N agents, no communication. Each agent runs beam search
    independently with different random seeds.
  
  Condition B (COMM): Same N agents, but every K rounds they share:
    - AxiomHint messages (compressed discoveries, Day 19 style)
    - ExpressionFragments (partial programs, Day 25 style)
  
  Condition C (FRAGMENT): Agents share only fragments (Day 25 mechanism),
    not full hints. This isolates the effect of structural sharing.

Metrics:
  1. rounds_to_consensus: how many rounds until ProtoAxiomPool promotes an axiom?
  2. final_compression_ratio: what is the best MDL cost achieved?
  3. herding_index: how similar are agents' search regions? (0=diverse, 1=all_same)
  4. unique_expressions_found: how many distinct structural patterns were found?

Statistical tests:
  - Mann-Whitney U (nonparametric): SOLO vs COMM for rounds_to_consensus
  - Cohen's d: effect size
  - 95% confidence intervals: bootstrap

Expected result:
  On simple environments (ModArith): SOLO ≈ COMM (communication is neutral)
  On complex environments (DampedOsc): COMM < SOLO (communication helps)
  Herding index: COMM > SOLO (communication reduces diversity, trade-off)
"""

from __future__ import annotations
import copy
import random
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class AgentState:
    """State of a single agent in the communication experiment."""
    agent_id: str
    best_expr_str: Optional[str]
    best_mdl_cost: float
    rounds_to_good_fit: Optional[int]   # first round with compression_ratio < 0.1
    search_centroid: Optional[List[int]]  # first 10 predictions (fingerprint)


@dataclass
class ExperimentRun:
    """Result of one run of the communication experiment."""
    condition: str      # "SOLO", "COMM", "FRAGMENT"
    seed: int
    n_rounds: int
    agent_states: List[AgentState]

    # Aggregate metrics
    rounds_to_consensus: Optional[int]
    final_best_cost: float
    herding_index: float        # Jaccard similarity of agent search regions
    unique_expressions: int     # distinct expression structures found

    @property
    def consensus_achieved(self) -> bool:
        return self.rounds_to_consensus is not None

    def description(self) -> str:
        return (
            f"Run({self.condition}, seed={self.seed}): "
            f"consensus={'round ' + str(self.rounds_to_consensus) if self.consensus_achieved else 'not reached'}, "
            f"best_cost={self.final_best_cost:.2f}, "
            f"herding={self.herding_index:.3f}, "
            f"unique_exprs={self.unique_expressions}"
        )


def _compute_herding_index(agent_states: List[AgentState]) -> float:
    """
    Measure how similar agents' searches are.
    
    Method: compare the first 20 predictions of each agent's best expression.
    If agents have the same fingerprint (same first 20 predictions) they
    are "herded" — exploring the same region.
    
    Returns: fraction of agent pairs with identical fingerprints.
    0.0 = fully diverse, 1.0 = all agents have identical search fingerprints.
    """
    fingerprints = [
        tuple(s.search_centroid) if s.search_centroid else None
        for s in agent_states
    ]
    valid = [f for f in fingerprints if f is not None]
    if len(valid) < 2:
        return 0.0

    n_pairs = len(valid) * (len(valid) - 1) // 2
    if n_pairs == 0:
        return 0.0

    matching_pairs = 0
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            if valid[i] == valid[j]:
                matching_pairs += 1

    return matching_pairs / n_pairs


def run_solo_condition(
    env_factory,
    n_agents: int,
    n_rounds: int,
    stream_length: int,
    beam_width: int,
    seed: int,
) -> ExperimentRun:
    """Run the SOLO condition: agents search independently."""
    from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
    from ouroboros.compression.mdl_engine import MDLEngine
    from ouroboros.agents.proto_axiom_pool import ProtoAxiomPool

    env = env_factory(seed=seed)
    pool = ProtoAxiomPool(consensus_threshold=0.5, n_agents=n_agents)
    mdl = MDLEngine()
    agent_best: Dict[str, Dict] = {f"A{i}": {"cost": float('inf'), "expr": None} for i in range(n_agents)}
    consensus_round = None

    for round_num in range(1, n_rounds + 1):
        obs = env.generate(stream_length, start=(round_num-1)*stream_length)

        for i in range(n_agents):
            cfg = BeamConfig(
                beam_width=beam_width, const_range=20, max_depth=4,
                mcmc_iterations=80, random_seed=seed * 100 + i * 7 + round_num,
            )
            expr = BeamSearchSynthesizer(cfg).search(obs)
            if expr is not None:
                preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                r = mdl.compute(preds, obs, expr.node_count(), expr.constant_count())
                pool.submit(f"A{i}", expr, r.total_mdl_cost, round_num)
                if r.total_mdl_cost < agent_best[f"A{i}"]["cost"]:
                    agent_best[f"A{i}"] = {"cost": r.total_mdl_cost, "expr": expr}

        if pool.has_promoted_axiom() and consensus_round is None:
            consensus_round = round_num

    # Build agent states
    states = []
    obs_sample = env.generate(20)
    for i in range(n_agents):
        info = agent_best[f"A{i}"]
        centroid = None
        if info["expr"]:
            centroid = [info["expr"].evaluate(t, obs_sample[:t]) for t in range(min(20, len(obs_sample)))]
        states.append(AgentState(
            agent_id=f"A{i}",
            best_expr_str=info["expr"].to_string() if info["expr"] else None,
            best_mdl_cost=info["cost"],
            rounds_to_good_fit=None,
            search_centroid=centroid,
        ))

    unique_exprs = len(set(s.best_expr_str for s in states if s.best_expr_str))

    return ExperimentRun(
        condition="SOLO",
        seed=seed,
        n_rounds=n_rounds,
        agent_states=states,
        rounds_to_consensus=consensus_round,
        final_best_cost=min(s.best_mdl_cost for s in states),
        herding_index=_compute_herding_index(states),
        unique_expressions=unique_exprs,
    )


def run_comm_condition(
    env_factory,
    n_agents: int,
    n_rounds: int,
    stream_length: int,
    beam_width: int,
    seed: int,
    hint_interval: int = 2,
) -> ExperimentRun:
    """
    Run the COMM condition: agents share AxiomHint messages.
    
    Every hint_interval rounds, the agent with the best expression
    broadcasts its search hint to all other agents.
    Receiving agents bias their next search toward the hinted region.
    """
    from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
    from ouroboros.compression.mdl_engine import MDLEngine
    from ouroboros.agents.proto_axiom_pool import ProtoAxiomPool
    from ouroboros.synthesis.expr_node import ExprNode, NodeType, build_linear_modular

    env = env_factory(seed=seed)
    pool = ProtoAxiomPool(consensus_threshold=0.5, n_agents=n_agents)
    mdl = MDLEngine()
    agent_best: Dict[str, Dict] = {f"A{i}": {"cost": float('inf'), "expr": None} for i in range(n_agents)}
    consensus_round = None

    # Shared hint: the best expression found so far (for warm starting)
    shared_hint: Optional[ExprNode] = None

    for round_num in range(1, n_rounds + 1):
        obs = env.generate(stream_length, start=(round_num-1)*stream_length)

        for i in range(n_agents):
            # Build seed list: include shared hint if available
            seeds = [shared_hint] if shared_hint is not None else []

            cfg = BeamConfig(
                beam_width=beam_width, const_range=20, max_depth=4,
                mcmc_iterations=80, random_seed=seed * 100 + i * 7 + round_num,
                seed_expressions=seeds,
            )
            expr = BeamSearchSynthesizer(cfg).search(obs)
            if expr is not None:
                preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                r = mdl.compute(preds, obs, expr.node_count(), expr.constant_count())
                pool.submit(f"A{i}", expr, r.total_mdl_cost, round_num)
                if r.total_mdl_cost < agent_best[f"A{i}"]["cost"]:
                    agent_best[f"A{i}"] = {"cost": r.total_mdl_cost, "expr": expr}

        # Update shared hint every K rounds
        if round_num % hint_interval == 0:
            best_agent = min(agent_best.items(), key=lambda x: x[1]["cost"])
            if best_agent[1]["expr"] is not None:
                shared_hint = copy.deepcopy(best_agent[1]["expr"])

        if pool.has_promoted_axiom() and consensus_round is None:
            consensus_round = round_num

    states = []
    obs_sample = env.generate(20)
    for i in range(n_agents):
        info = agent_best[f"A{i}"]
        centroid = None
        if info["expr"]:
            centroid = [info["expr"].evaluate(t, obs_sample[:t]) for t in range(min(20, len(obs_sample)))]
        states.append(AgentState(
            agent_id=f"A{i}",
            best_expr_str=info["expr"].to_string() if info["expr"] else None,
            best_mdl_cost=info["cost"],
            rounds_to_good_fit=None,
            search_centroid=centroid,
        ))

    unique_exprs = len(set(s.best_expr_str for s in states if s.best_expr_str))
    return ExperimentRun(
        condition="COMM",
        seed=seed,
        n_rounds=n_rounds,
        agent_states=states,
        rounds_to_consensus=consensus_round,
        final_best_cost=min(s.best_mdl_cost for s in states),
        herding_index=_compute_herding_index(states),
        unique_expressions=unique_exprs,
    )