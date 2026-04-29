"""
Run the Mathematical Civilization Simulation.

Fast mode:  16 agents × 10 envs × 50 rounds ≈ 10 minutes
Full mode:  64 agents × 20 envs × 200 rounds ≈ 90 minutes
"""

import sys; sys.path.insert(0, '.')
import argparse, json
from pathlib import Path
from ouroboros.civilization.simulator import CivilizationSimulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()

    if args.full:
        sim = CivilizationSimulator(
            n_agents=64, n_rounds=200, stream_length=300,
            beam_width=15, n_iterations=8,
            verbose=True, report_every=20,
        )
    else:
        sim = CivilizationSimulator(
            n_agents=16, n_rounds=50, stream_length=200,
            beam_width=10, n_iterations=5,
            verbose=True, report_every=10,
        )

    result = sim.run()

    # Save results
    Path("results").mkdir(exist_ok=True)
    data = {
        "n_agents": result.n_agents, "n_rounds": result.n_rounds,
        "total_discoveries": result.total_discoveries,
        "order_correlation": result.order_correlation,
        "specialization_emerged": result.specialization_emerged,
        "ouroboros_order": result.ouroboros_discovery_order,
        "human_order": result.human_discovery_order,
        "discoveries": [
            {"concept": d.concept.name, "round": d.round_discovered,
             "agent": d.agent_id, "env": d.environment_name}
            for d in result.concept_discoveries
        ],
        "runtime_seconds": result.total_runtime_seconds,
    }
    Path("results/civilization_result.json").write_text(json.dumps(data, indent=2))
    print("\nSaved to results/civilization_result.json")


if __name__ == '__main__':
    main()