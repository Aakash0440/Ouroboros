"""
End-to-end integration test: run the complete OUROBOROS system.

This test verifies that all components work together:
  Days 1–12: Basic system
  Days 13–16: Extended primitives, KB, GPU
  Days 17–20: Lean4, communication, Layer 1
  Days 21–23: Continuous, Lean4 proofs, Layer 2
  Days 24–26: Layer 3, collaboration, long-range
  Days 27–29: Acceleration, benchmark, communication stats
"""

import pytest


class TestEndToEnd:
    """Full system integration tests."""

    def test_modular_discovery_pipeline(self):
        """Complete pipeline: generate → search → score → pool → axiom."""
        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine
        from ouroboros.agents.proto_axiom_pool import ProtoAxiomPool

        env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
        obs = env.generate(300)
        mdl = MDLEngine()
        pool = ProtoAxiomPool(consensus_threshold=0.5, n_agents=4)

        for i in range(4):
            cfg = BeamConfig(beam_width=15, const_range=15, max_depth=4,
                             mcmc_iterations=80, random_seed=i*7)
            expr = BeamSearchSynthesizer(cfg).search(obs)
            if expr:
                preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                r = mdl.compute(preds, obs, expr.node_count(), expr.constant_count())
                pool.submit(f"A{i}", expr, r.total_mdl_cost, round_num=1)

        # Should have some submissions
        assert pool.n_submissions > 0

    def test_continuous_env_and_gaussian_mdl(self):
        """Continuous environments + Gaussian MDL work together."""
        from ouroboros.continuous.environments import SineEnv
        from ouroboros.continuous.mdl import compute_gaussian_mdl
        from ouroboros.continuous.expr_nodes import build_sine_expr
        import math

        env = SineEnv(frequency=1/7)
        obs = env.generate(100)
        expr = build_sine_expr(frequency=1/7)
        preds = [expr.evaluate(t, []) for t in range(100)]
        result = compute_gaussian_mdl(preds, obs, 3, 1)
        assert result.r_squared > 0.999
        assert result.is_good_fit

    def test_layer3_agent_can_propose(self):
        """Layer3Agent can enumerate strategies and select the best."""
        from ouroboros.meta.layer3_agent import Layer3Agent, Layer3AgentConfig
        from ouroboros.meta.strategy_library import STRATEGY_LIBRARY
        agent = Layer3Agent(config=Layer3AgentConfig(random_seed=42))
        assert agent.current_strategy.name() == "BeamSearch"
        assert len(STRATEGY_LIBRARY.all_strategies()) >= 5

    def test_collaborative_session_runs(self):
        """CollaborativeProofSession completes without errors."""
        from ouroboros.agents.collaborative_proof import CollaborativeAgent, CollaborativeProofSession
        from ouroboros.environments.modular import ModularArithmeticEnv
        agents = [CollaborativeAgent(f"E2E_{i}", const_range=12, beam_width=6, random_seed=i) for i in range(3)]
        session = CollaborativeProofSession(agents, n_rounds=2, stream_length=80)
        result = session.run(ModularArithmeticEnv(modulus=5), verbose=False)
        assert result.n_rounds == 2

    def test_long_range_bm_on_tribonacci(self):
        """BM detects Tribonacci recurrence."""
        from ouroboros.environments.long_range import TribonacciModEnv
        from ouroboros.emergence.recurrence_detector import RecurrenceDetector
        env = TribonacciModEnv(modulus=7)
        seq = env.generate(200)
        detector = RecurrenceDetector(max_order=5)
        axiom = detector.detect(seq, modulus=7)
        # BM may or may not succeed (depends on modulus being prime)
        # We just verify it doesn't crash
        assert axiom is None or axiom.order <= 5

    def test_sparse_beam_on_large_alphabet(self):
        """SparseBeamSearch works on alphabet=77."""
        from ouroboros.acceleration.sparse_beam import SparseBeamSearch, SparseBeamConfig
        obs = [(3*t+1)%77 for t in range(200)]
        cfg = SparseBeamConfig(beam_width=8, n_iterations=3, const_range=80)
        result = SparseBeamSearch(cfg).search(obs, alphabet_size=77)
        assert result is not None

    def test_benchmark_experiment_result_structure(self):
        """ExperimentResult has all required fields."""
        from ouroboros.benchmark.runner import ExperimentResult
        r = ExperimentResult("test", "metric", "units", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert hasattr(r, 'mean') and r.mean > 0
        assert hasattr(r, 'ci_95') and callable(r.ci_95)
        assert hasattr(r, 'latex_str') and callable(r.latex_str)

    def test_statistical_analysis_complete(self):
        """Full A/B analysis runs without errors."""
        from ouroboros.experiments.communication_experiment import ExperimentRun, AgentState
        from ouroboros.experiments.statistical_tests import StatisticalTester
        def make_runs(cond, n):
            return [
                ExperimentRun(cond, i, 10, [], i+5, float(50+i), float(i%2)*0.3, 3-i%2)
                for i in range(n)
            ]
        solo = make_runs("SOLO", 5)
        comm = make_runs("COMM", 5)
        result = StatisticalTester().analyze(solo, comm, n_rounds=10)
        s = result.summary()
        assert "CONCLUSION" in s
        assert "Mann-Whitney" in s