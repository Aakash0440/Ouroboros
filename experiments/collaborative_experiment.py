"""
Collaborative synthesis speedup experiment.

Measures: does cross-agent program sharing speed up discovery?
Hypothesis: on complex environments, collaboration reduces the
number of solo search steps needed before an axiom is promoted.

Experiment design:
  Condition A (solo): 6 independent agents, each runs full beam search
  Condition B (collab): same 6 agents, but share fragments each round

Expected result:
  On ModularArithmetic(7): similar (beam already finds it quickly)
  On DampedOscillatorEnv: collaboration significantly faster
    (structure is complex, but once one agent finds the shape,
     others can quickly fill the constants)
"""

import sys; sys.path.insert(0, '.')
import time
from ouroboros.agents.collaborative_proof import (
    CollaborativeAgent, CollaborativeProofSession,
)
from ouroboros.environments.modular import ModularArithmeticEnv


def run_solo_baseline(n_agents: int, env, stream_length: int):
    """Run agents without collaboration."""
    from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
    from ouroboros.compression.mdl_engine import MDLEngine

    mdl = MDLEngine()
    best_cost = float('inf')
    total_time = 0

    for i in range(n_agents):
        start = time.time()
        obs = env.generate(stream_length)
        cfg = BeamConfig(beam_width=15, const_range=20, max_depth=4,
                         mcmc_iterations=100, random_seed=42 + i*7)
        synth = BeamSearchSynthesizer(cfg)
        expr = synth.search(obs)
        elapsed = time.time() - start
        total_time += elapsed

        if expr is not None:
            preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
            result = mdl.compute(preds, obs, expr.node_count(), expr.constant_count())
            if result.total_mdl_cost < best_cost:
                best_cost = result.total_mdl_cost

    return best_cost, total_time


def run_collaborative(n_agents: int, env, stream_length: int, n_rounds: int):
    """Run agents with collaboration."""
    agents = [
        CollaborativeAgent(
            agent_id=f"COLLAB_{i:02d}",
            const_range=20, beam_width=12,
            max_depth=4, random_seed=42 + i*7,
        )
        for i in range(n_agents)
    ]
    session = CollaborativeProofSession(
        agents=agents,
        n_rounds=n_rounds,
        stream_length=stream_length,
        n_holes_to_broadcast=1,
    )

    start = time.time()
    result = session.run(env, verbose=False)
    elapsed = time.time() - start

    return result.best_mdl_cost, elapsed, result


def main():
    print("\n" + "="*60)
    print("COLLABORATIVE SYNTHESIS SPEEDUP EXPERIMENT")
    print("="*60)

    N_AGENTS = 6
    STREAM = 300
    N_ROUNDS = 4

    for modulus in [5, 7, 11]:
        env = ModularArithmeticEnv(modulus=modulus, slope=3, intercept=1)
        print(f"\n── ModularArithmetic(mod={modulus}) ──")

        solo_cost, solo_time = run_solo_baseline(N_AGENTS, env, STREAM)
        print(f"  Solo:  cost={solo_cost:.2f} bits, time={solo_time:.2f}s")

        collab_cost, collab_time, session_result = run_collaborative(
            N_AGENTS, env, STREAM, N_ROUNDS
        )
        print(f"  Collab: cost={collab_cost:.2f} bits, time={collab_time:.2f}s")
        print(f"  Fragments broadcast: {session_result.fragments_broadcast}")
        print(f"  Completions: {session_result.completions_successful}/{session_result.completions_attempted}")

        benefit = solo_cost - collab_cost
        speedup = solo_time / collab_time if collab_time > 0 else 1.0
        print(f"  Benefit: {benefit:+.2f} bits, {speedup:.2f}× {'faster' if speedup>1 else 'slower'}")


if __name__ == '__main__':
    main()
    print("\n✅ Collaborative experiment complete")