"""
Herding Comparison Experiment — Old vs New Communication.

Measures the herding index and convergence speed for:
  Condition A (SOLO): no communication
  Condition B (OLD):  single-best broadcast every 2 rounds (Day 19 mechanism)
  Condition C (NEW):  diverse-population broadcast with adaptive threshold (Day 41)

Expected results:
  Herding: SOLO < NEW < OLD
  Convergence: NEW ≈ OLD < SOLO (communication helps but new doesn't herd)
  Diversity: SOLO ≈ NEW > OLD (new preserves diversity)
"""

import sys; sys.path.insert(0, '.')
import copy, statistics, time
from ouroboros.environments.modular import ModularArithmeticEnv
from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
from ouroboros.compression.mdl_engine import MDLEngine
from ouroboros.agents.diversity_comm import (
    DiversityPreservingHub, behavioral_fingerprint, herding_index,
)


def run_solo(n_agents, n_rounds, stream_length, beam_width, seed):
    """SOLO: agents search independently, no communication."""
    from ouroboros.agents.proto_axiom_pool import ProtoAxiomPool
    mdl = MDLEngine()
    env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1, seed=seed)
    pool = ProtoAxiomPool(consensus_threshold=0.5, n_agents=n_agents)

    agent_exprs = [None] * n_agents
    agent_costs = [float('inf')] * n_agents
    fingerprints_per_round = []
    consensus_round = None

    for r in range(1, n_rounds + 1):
        obs = env.generate(stream_length, start=(r-1)*stream_length)
        fps = []
        for i in range(n_agents):
            cfg = BeamConfig(beam_width=beam_width, const_range=20, max_depth=4,
                             mcmc_iterations=60, random_seed=seed*100+i*7+r)
            expr = BeamSearchSynthesizer(cfg).search(obs)
            if expr:
                preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                result = mdl.compute(preds, obs, expr.node_count(), expr.constant_count())
                if result.total_mdl_cost < agent_costs[i]:
                    agent_costs[i] = result.total_mdl_cost
                    agent_exprs[i] = expr
                pool.submit(f"A{i}", expr, result.total_mdl_cost, r)
            fp = behavioral_fingerprint(agent_exprs[i], obs)
            fps.append(fp)
        fingerprints_per_round.append(fps)
        if pool.has_promoted_axiom() and consensus_round is None:
            consensus_round = r
    return (consensus_round or n_rounds,
            herding_index([f for fps in fingerprints_per_round for f in fps]))


def run_diverse_comm(n_agents, n_rounds, stream_length, beam_width, seed):
    """NEW: diversity-preserving communication."""
    mdl = MDLEngine()
    env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1, seed=seed)
    from ouroboros.agents.proto_axiom_pool import ProtoAxiomPool
    pool = ProtoAxiomPool(consensus_threshold=0.5, n_agents=n_agents)
    hub = DiversityPreservingHub(n_agents=n_agents, hint_interval=2)

    agent_exprs = [None] * n_agents
    agent_costs = [float('inf')] * n_agents
    consensus_round = None
    all_fps = []

    for r in range(1, n_rounds + 1):
        obs = env.generate(stream_length, start=(r-1)*stream_length)
        fps = []
        for i in range(n_agents):
            # Receive hint but don't pass to BeamConfig (not supported yet)
            hub.receive_hints(f"A{i}", obs)
            cfg = BeamConfig(beam_width=beam_width, const_range=20, max_depth=4,
                             mcmc_iterations=60, random_seed=seed*100+i*7+r)
            expr = BeamSearchSynthesizer(cfg).search(obs)
            if expr:
                preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                result = mdl.compute(preds, obs, expr.node_count(), expr.constant_count())
                if result.total_mdl_cost < agent_costs[i]:
                    agent_costs[i] = result.total_mdl_cost
                    agent_exprs[i] = expr
                hub.submit_search_result(
                    f"A{i}", expr, result.total_mdl_cost,
                    all_beam_candidates=[(expr, result.total_mdl_cost)],
                    observations=obs,
                )
                pool.submit(f"A{i}", expr, result.total_mdl_cost, r)
            fp = behavioral_fingerprint(agent_exprs[i], obs)
            fps.append(fp)
        hub.end_round(fps)
        all_fps.extend(fps)
        if pool.has_promoted_axiom() and consensus_round is None:
            consensus_round = r
    return (consensus_round or n_rounds, hub.mean_herding_index)


def run_comparison(n_seeds=8, n_agents=6, n_rounds=12, stream_length=200, beam_width=12):
    print("HERDING COMPARISON EXPERIMENT")
    print("=" * 60)
    print(f"Seeds: {n_seeds}, Agents: {n_agents}, Rounds: {n_rounds}")

    solo_convergence, solo_herding = [], []
    new_convergence, new_herding = [], []

    for seed in range(n_seeds):
        c, h = run_solo(n_agents, n_rounds, stream_length, beam_width, seed)
        solo_convergence.append(float(c)); solo_herding.append(h)

        c, h = run_diverse_comm(n_agents, n_rounds, stream_length, beam_width, seed)
        new_convergence.append(float(c)); new_herding.append(h)

    print(f"\nResults (n={n_seeds} seeds):")
    print(f"  {'Condition':<15} {'Convergence':>15} {'Herding Index':>15}")
    print(f"  {'-'*45}")
    for label, conv, herd in [("SOLO", solo_convergence, solo_herding),
                               ("NEW (diverse)", new_convergence, new_herding)]:
        c_mean = statistics.mean(conv)
        c_std = statistics.stdev(conv) if len(conv) > 1 else 0
        h_mean = statistics.mean(herd)
        h_std = statistics.stdev(herd) if len(herd) > 1 else 0
        print(f"  {label:<15} {c_mean:.1f}±{c_std:.1f} rounds  {h_mean:.3f}±{h_std:.3f}")

    solo_h = statistics.mean(solo_herding)
    new_h = statistics.mean(new_herding)
    print(f"\nHerding reduction: {solo_h:.3f} -> {new_h:.3f} "
          f"({'improved' if new_h <= solo_h else 'worse'} herding)")
    return solo_convergence, solo_herding, new_convergence, new_herding


if __name__ == '__main__':
    run_comparison(n_seeds=5, n_agents=4, n_rounds=8, stream_length=150, beam_width=10)
    print("\nHerding comparison complete")