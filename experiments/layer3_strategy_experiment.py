"""
Layer 3 strategy experiment.

Demonstrates that agents starting with BeamSearch can discover that
RandomRestart or Hybrid performs better on certain environments,
and permanently switch strategies through the proof market.

Expected result: on ModularArithmetic(7), BeamSearch is generally
optimal (it was designed for this). On MultiScaleEnv, MultiScale
strategy may outperform pure BeamSearch.
"""

import sys; sys.path.insert(0, '.')
import json
from ouroboros.meta.meta_runner import MetaSearchRunner, MetaRunnerConfig
from ouroboros.meta.strategy_library import STRATEGY_LIBRARY
from ouroboros.environments.modular import ModularArithmeticEnv
from ouroboros.environments.multi_scale import MultiScaleEnv
from ouroboros.environments.noise import NoiseEnv


def run_modular_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT: Layer 3 on ModularArithmetic(7)")
    print("Expected: BeamSearch stays dominant (it's already optimal)")
    print("="*60)

    runner = MetaSearchRunner.for_modular_arithmetic(
        modulus=7, n_rounds=4, verbose=False
    )
    result = runner.run()

    print(f"\nResults:")
    print(f"  Strategy proposals: {result.total_strategy_proposals}")
    print(f"  Strategy approvals: {result.total_strategy_approvals}")
    print(f"  Strategy evolution:")
    for agent_id, history in result.strategy_evolution.items():
        print(f"    {agent_id}: {' → '.join(history)}")

    return result


def run_multiscale_experiment():
    print("\n" + "="*60)
    print("EXPERIMENT: Layer 3 on MultiScaleEnv")
    print("Expected: MultiScaleStrategy may be approved over BeamSearch")
    print("="*60)

    from ouroboros.meta.layer3_agent import Layer3Agent, Layer3AgentConfig
    from ouroboros.meta.strategy_market import StrategyProofMarket, StrategyMarketConfig
    from ouroboros.meta.search_strategy import SearchConfig

    env = MultiScaleEnv()
    val_envs = [ModularArithmeticEnv(modulus=7), NoiseEnv(alphabet_size=8)]

    agents = [
        Layer3Agent(
            config=Layer3AgentConfig(
                agent_id=f"L3_{i:02d}",
                strategy_proposal_interval=4,
                search_time_budget=2.0,
                search_beam_width=15,
                random_seed=42 + i * 7,
            )
        )
        for i in range(4)
    ]

    market = StrategyProofMarket(
        config=StrategyMarketConfig(
            min_cost_improvement_bits=2.0,
            search_time_budget=2.0,
        ),
        validation_environments=val_envs,
    )

    proposals_made = 0
    proposals_approved = 0

    for round_num in range(1, 17):
        for agent in agents:
            log = agent.run_round(
                env=env,
                strategy_market=market,
                validation_envs=val_envs,
                round_num=round_num,
                stream_length=300,
                verbose=False,
            )
            proposals_made += int(log["strategy_proposed"])
            proposals_approved += int(log["strategy_approved"])

    print(f"\nResults:")
    print(f"  Proposals made: {proposals_made}")
    print(f"  Proposals approved: {proposals_approved}")
    print(f"  Final strategies:")
    for agent in agents:
        print(f"    {agent.cfg.agent_id}: {agent.current_strategy.name()}")
        print(f"    History: {' → '.join(agent.stats.strategy_history)}")

    return proposals_made, proposals_approved


def run_strategy_comparison():
    """Compare all strategies head-to-head on ModularArithmetic."""
    print("\n" + "="*60)
    print("STRATEGY COMPARISON: Head-to-head on ModArith(7)")
    print("="*60)

    from ouroboros.meta.search_strategy import SearchConfig

    env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
    obs = env.generate(400)
    config = SearchConfig(
        beam_width=20,
        time_budget_seconds=3.0,
        n_restarts=5,
        node_budget=2000,
        mcmc_iterations=100,
        random_seed=42,
    )

    results = []
    for strategy in STRATEGY_LIBRARY.all_strategies():
        result = strategy.search(obs, config)
        results.append((strategy.name(), result))
        print(
            f"  {strategy.name():30s}: "
            f"cost={result.best_mdl_cost:.2f} bits, "
            f"time={result.wall_time_seconds:.2f}s, "
            f"evals={result.n_evaluations}"
        )

    # Sort by cost
    results.sort(key=lambda x: x[1].best_mdl_cost)
    print(f"\nRanking:")
    for i, (name, r) in enumerate(results):
        print(f"  #{i+1}: {name} — {r.best_mdl_cost:.2f} bits")

    return results


if __name__ == '__main__':
    run_strategy_comparison()
    run_modular_experiment()
    run_multiscale_experiment()
    print("\n✅ Layer 3 strategy experiment complete")