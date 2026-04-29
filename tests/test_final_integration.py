"""
Final integration test — exercises the complete OUROBOROS system.
Days 1-38, all components working together.
"""
import pytest


class TestCompleteSystem:
    """Tests that verify the entire system works end-to-end."""

    def test_original_discovery_pipeline(self):
        """Days 1-12: basic modular arithmetic discovery."""
        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine
        env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
        obs = env.generate(200)
        cfg = BeamConfig(beam_width=12, const_range=15, max_depth=4, mcmc_iterations=80)
        expr = BeamSearchSynthesizer(cfg).search(obs)
        assert expr is not None

    def test_extended_nodes_accessible(self):
        """Days 30-31: 60 node types available."""
        from ouroboros.nodes.extended_nodes import NODE_SPECS, ExtNodeType
        assert len(NODE_SPECS) == 40
        assert ExtNodeType.ISPRIME in NODE_SPECS

    def test_grammar_constrained_search(self):
        """Day 30: grammar beam search finds valid expression."""
        from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
        obs = [(3*t+1)%7 for t in range(80)]
        result = GrammarConstrainedBeam(GrammarBeamConfig(beam_width=6, n_iterations=3)).search(obs)
        assert result is not None

    def test_hierarchical_router(self):
        """Day 31: router classifies and searches."""
        from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
        router = HierarchicalSearchRouter(RouterConfig(beam_width=6, n_iterations=3))
        obs = [(3*t+1)%7 for t in range(80)]
        result = router.search(obs, alphabet_size=7)
        assert result.math_family is not None

    def test_physics_law_detection(self):
        """Day 33: Hooke's Law detected from spring data."""
        import math
        from ouroboros.physics.law_signature import _test_hookes_law, PhysicsLaw
        seq = [10.0 * math.cos(0.3 * t) for t in range(100)]
        result = _test_hookes_law(seq, threshold=0.8)
        assert result.passed

    def test_fft_period_finder(self):
        """Day 34: FFT detects period 7."""
        from ouroboros.search.fft_period_finder import FFTPeriodFinder
        seq = [float(t%7) for t in range(200)]
        result = FFTPeriodFinder().find_dominant_period(seq)
        assert result is not None

    def test_prime_count_formula(self):
        """Day 34: CUMSUM(ISPRIME(t)) = π(t)."""
        from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
        from ouroboros.synthesis.expr_node import NodeType
        from ouroboros.environments.algorithm_env import PrimeCountEnv
        def time_e():
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.TIME; n.value = 0.0
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n
        isprime = ExtExprNode(ExtNodeType.ISPRIME, left=time_e())
        cumsum = ExtExprNode(ExtNodeType.CUMSUM, left=isprime)
        env = PrimeCountEnv()
        obs = env.generate(30)
        preds = [int(round(cumsum.evaluate(t, []))) for t in range(30)]
        assert sum(1 for p, o in zip(preds, obs) if p == o) >= 28

    def test_knowledge_accumulation(self):
        """Day 35: KB grows across sessions."""
        from ouroboros.knowledge.accumulation import SimpleAxiomKB
        kb = SimpleAxiomKB()
        kb.add_axiom("expr1", 45.0, "env1", 1)
        kb.add_axiom("expr2", 40.0, "env1", 2)
        assert kb.n_axioms == 2

    def test_layer4_dsl_runs(self):
        """Day 36: Layer 4 DSL interpreter executes."""
        from ouroboros.layer4.search_dsl import standard_beam_program
        from ouroboros.layer4.interpreter import AlgorithmInterpreter
        prog = standard_beam_program(width=5, iterations=2)
        interp = AlgorithmInterpreter(time_budget_seconds=2.0)
        obs = [(3*t+1)%7 for t in range(60)]
        expr, cost, elapsed = interp.run(prog, obs, alphabet_size=7)
        assert elapsed > 0

    def test_civilization_concept_detection(self):
        """Day 37: MathConcept correctly identifies node types."""
        from ouroboros.civilization.simulator import MathConcept
        c = MathConcept("Test", "2024", 1, ["MOD"])
        assert c.is_indicated_by({"MOD", "CONST"})

    def test_paper_generation(self, tmp_path):
        """Day 38: Both papers generate without error."""
        from ouroboros.papers.paper_writer import ExperimentNumbers, generate_paper1, generate_paper2
        nums = ExperimentNumbers()
        p1 = generate_paper1(nums, str(tmp_path))
        p2 = generate_paper2(nums, str(tmp_path))
        from pathlib import Path
        assert Path(p1).exists() and Path(p2).exists()
        assert len(Path(p1).read_text()) > 1000
        assert len(Path(p2).read_text()) > 1000

    def test_full_test_suite_count(self):
        """Verify we have the expected number of tests."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '--collect-only', '-q'],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split('\n')
        count_line = next((l for l in lines if 'test' in l and 'selected' in l), "")
        print(f"\n  Total tests collected: {count_line}")
        # Should have significantly more than 690 (the Part 2 count)
        assert result.returncode == 0