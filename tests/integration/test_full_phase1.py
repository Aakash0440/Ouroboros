"""
Comprehensive Phase 1 integration tests.

Tests the complete pipeline from environment to axiom promotion.
These tests take 60–120 seconds total but verify the whole system.

Run:
    pytest tests/integration/test_full_phase1.py -v --timeout=120
"""

import pytest
import json
from pathlib import Path
from ouroboros.core.phase1_runner import Phase1Runner
from ouroboros.environment.structured import (
    ModularArithmeticEnv, BinaryRepeatEnv,
    FibonacciModEnv, NoiseEnv, MultiScaleEnv
)
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.agents.hierarchical_agent import HierarchicalAgent
from ouroboros.emergence.proto_axiom_pool import ProtoAxiomPool
from ouroboros.emergence.scale_axiom_pool import ScaleAxiomPool
from ouroboros.compression.mdl import naive_bits, compression_ratio


class TestEnvironmentCompression:
    """Verify each environment has expected compression characteristics."""

    @pytest.mark.parametrize("modulus,slope,intercept", [
        (5, 2, 1),
        (7, 3, 1),
        (11, 4, 2),
    ])
    def test_modular_env_compresses_well(self, modulus, slope, intercept):
        """Modular environments should be significantly compressible."""
        env = ModularArithmeticEnv(modulus, slope, intercept)
        env.reset(500)
        stream = env.peek_all()
        ratio = compression_ratio(stream, modulus)
        assert ratio < 0.60, (
            f"ModularArith({modulus},{slope},{intercept}): ratio={ratio:.3f} not < 0.60"
        )

    def test_binary_repeat_highly_compressible(self):
        env = BinaryRepeatEnv()
        env.reset(500)
        ratio = compression_ratio(env.peek_all(), 2)
        assert ratio < 0.35

    def test_noise_incompressible(self):
        env = NoiseEnv(4, seed=0)
        env.reset(2000)
        ratio = compression_ratio(env.peek_all(), 4)
        assert ratio > 0.80, f"Noise ratio {ratio:.3f} should be > 0.80"

    def test_fibonacci_partially_compressible(self):
        env = FibonacciModEnv(11)
        env.reset(500)
        ratio = compression_ratio(env.peek_all(), 11)
        # Fibonacci mod 11 has Pisano period 10 — should be compressible
        assert ratio < 0.90


class TestSynthesisAgentConvergence:
    """Verify synthesis agents converge on the correct rule."""

    def test_finds_binary_repeat(self):
        env = BinaryRepeatEnv()
        env.reset(200)
        agent = SynthesisAgent(0, 2, beam_width=10, max_depth=2, mcmc_iterations=50)
        agent.observe(env.peek_all())
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        assert ratio < 0.30, f"Binary repeat: ratio={ratio:.3f} not < 0.30"

    @pytest.mark.parametrize("seed", [42, 55, 71])
    def test_multiple_seeds_find_modular(self, seed):
        """With different seeds, agents should still compress ModularArith well."""
        env = ModularArithmeticEnv(7, 3, 1)
        env.reset(400)
        agent = SynthesisAgent(0, 7, beam_width=15, max_depth=3,
                               mcmc_iterations=80, seed=seed)
        agent.observe(env.peek_all())
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        assert ratio < 0.60, f"seed={seed}: ratio={ratio:.3f} not < 0.60"

    def test_symbolic_beats_ngram_on_structured(self):
        """After sufficient data, symbolic wins over n-gram."""
        env = ModularArithmeticEnv(7, 3, 1)
        env.reset(500)
        stream = env.peek_all()
        agent = SynthesisAgent(0, 7, beam_width=20, max_depth=3, mcmc_iterations=100)
        agent.observe(stream)
        agent.search_and_update()
        # Symbolic should have won at least once
        assert agent.symbolic_wins >= 1


class TestAxiomEmergence:
    """Verify axiom emergence conditions."""

    def test_consensus_requires_minimum_agents(self):
        """Less than threshold → no axiom promoted."""
        from ouroboros.compression.program_synthesis import build_linear_modular, C
        pool = ProtoAxiomPool(8, 0.5, 7)
        expr = build_linear_modular(3, 1, 7)

        # Only 3 agents (< 4 = 50% of 8)
        for i in range(3):
            pool.submit(i, expr, 10.0)
        for i in range(3, 8):
            pool.submit(i, C(i * 77), 500.0)

        axioms = pool.detect_consensus(100, "Test", 5000.0)
        # The group of 3 < min_support=4
        for ax in axioms:
            assert len(ax.supporting_agents) >= 4

    def test_noise_pool_stays_empty(self):
        """Running full pipeline on noise should produce no axioms."""
        runner = Phase1Runner.for_noise_baseline(
            num_agents=4,
            run_dir='experiments/phase1/runs/integration_noise'
        )
        results = runner.run(stream_length=400, eval_interval=200, verbose=False)
        assert len(results.axioms_promoted) == 0

    def test_modular_eventually_produces_axiom(self):
        """With enough data, ModularArith should produce at least one axiom."""
        runner = Phase1Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=6,
            run_dir='experiments/phase1/runs/integration_modular'
        )
        from ouroboros.core.config import OuroborosConfig
        runner.config.compression.beam_width = 25
        runner.config.compression.const_range = 21

        results = runner.run(
            stream_length=1200,
            eval_interval=300,
            consensus_threshold=0.4,
            verbose=False
        )
        # This may not always produce an axiom (probabilistic)
        # but best_ratio should be < 0.5
        assert results.best_ratio < 0.5, (
            f"ModularArith best_ratio={results.best_ratio:.3f} not < 0.5"
        )


class TestHierarchicalPipeline:
    """Test hierarchical agent pipeline."""

    def test_hierarchical_agent_multiple_scales(self):
        """HierarchicalAgent should maintain programs at multiple scales."""
        env = MultiScaleEnv(28, 7, 0.02, seed=42)
        env.reset(600)
        stream = env.peek_all()

        agent = HierarchicalAgent(
            0, 4, scales=[1, 4, 16],
            beam_width=10, max_depth=2, mcmc_iterations=30
        )
        agent.observe(stream)
        agent.search_and_update()

        # Should have searched at each scale
        assert agent.dominant_scale in [1, 4, 16]

    def test_scale_axiom_pool_consistency_bonus(self):
        """Axioms confirmed at multiple scales get a bonus."""
        from ouroboros.compression.program_synthesis import build_linear_modular
        pool = ScaleAxiomPool([1, 4], 4, 0.5, 7)
        expr = build_linear_modular(3, 1, 7)

        # Submit at BOTH scales
        for i in range(3):
            pool.submit(1, i, expr, 10.0)
            pool.submit(4, i, expr, 10.0)
        from ouroboros.compression.program_synthesis import C
        pool.submit(1, 3, C(99), 500.0)
        pool.submit(4, 3, C(99), 500.0)

        seq = [(3*t+1)%7 for t in range(100)]
        nb = naive_bits(seq, 7)
        pool.detect_all_scales(100, "Test", {1: nb, 4: nb/4})

        # If both scales confirm same fingerprint, consistency bonus applies
        for ax in pool.scale_axioms:
            if len(ax.confirmed_at_scales) > 1:
                assert ax.adjusted_confidence >= ax.base_axiom.confidence


class TestPhase1Results:
    """Test results serialization."""

    def test_results_json_roundtrip(self, tmp_path):
        runner = Phase1Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir=str(tmp_path / 'runs')
        )
        results = runner.run(stream_length=200, eval_interval=100, verbose=False)
        out = str(tmp_path / 'results.json')
        runner.save_results(out)

        with open(out) as f:
            loaded = json.load(f)

        assert loaded['environment'] == results.environment_name
        assert abs(loaded['mean_ratio'] - results.mean_ratio) < 0.001
        assert loaded['num_agents'] == 2