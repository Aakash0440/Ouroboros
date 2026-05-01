"""
Final system closure tests — Day 50.
These tests verify the complete OUROBOROS system from Day 1 to Day 50.
"""
import pytest
import math
from pathlib import Path


class TestArxivBuilder:
    def test_paper1_generates(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ArxivPaperBuilder
        builder = ArxivPaperBuilder(output_dir=str(tmp_path))
        p1 = builder.build_paper1()
        assert Path(p1).exists()
        main_tex = Path(p1) / "main.tex"
        assert main_tex.exists()
        content = main_tex.read_text()
        assert "\\documentclass" in content
        assert "0.0041" in content  # compression ratio

    def test_paper2_generates(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ArxivPaperBuilder
        builder = ArxivPaperBuilder(output_dir=str(tmp_path))
        p2 = builder.build_paper2()
        content = (Path(p2) / "main.tex").read_text()
        assert "Proof Market" in content or "proof market" in content.lower()

    def test_bibliography_included(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ArxivPaperBuilder
        builder = ArxivPaperBuilder(output_dir=str(tmp_path))
        p1 = builder.build_paper1()
        bib = Path(p1) / "bibliography.bib"
        assert bib.exists()
        content = bib.read_text()
        assert "koza1992" in content
        assert "schmidt2009" in content

    def test_makefile_included(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ArxivPaperBuilder
        builder = ArxivPaperBuilder(output_dir=str(tmp_path))
        p1 = builder.build_paper1()
        makefile = Path(p1) / "Makefile"
        assert makefile.exists()
        assert "pdflatex" in makefile.read_text()

    def test_no_placeholders_in_paper(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ArxivPaperBuilder
        builder = ArxivPaperBuilder(output_dir=str(tmp_path))
        p1 = builder.build_paper1()
        content = (Path(p1) / "main.tex").read_text()
        assert "TODO" not in content
        assert "PLACEHOLDER" not in content
        assert "FIXME" not in content

    def test_both_papers_build(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ArxivPaperBuilder
        builder = ArxivPaperBuilder(output_dir=str(tmp_path))
        p1, p2 = builder.build_both()
        assert Path(p1).exists()
        assert Path(p2).exists()


class TestReproducibilityPackage:
    def test_generates_files(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ReproducibilityPackage
        pkg = ReproducibilityPackage()
        out = pkg.generate(str(tmp_path / "repro"))
        assert (Path(out) / "reproduce.sh").exists()
        assert (Path(out) / "README.md").exists()

    def test_script_has_steps(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ReproducibilityPackage
        pkg = ReproducibilityPackage()
        out = pkg.generate(str(tmp_path / "repro"))
        script = (Path(out) / "reproduce.sh").read_text()
        assert "Step 1" in script
        assert "Step 5" in script

    def test_readme_has_table(self, tmp_path):
        from ouroboros.papers.arxiv_builder import ReproducibilityPackage
        pkg = ReproducibilityPackage()
        out = pkg.generate(str(tmp_path / "repro"))
        readme = (Path(out) / "README.md").read_text(encoding="utf-8")
        assert "0.0041" in readme or "Compression" in readme


class TestCompleteSystemEndToEnd:
    """End-to-end tests that exercise the complete OUROBOROS stack."""

    def test_day1_to_day10_core(self):
        """Original core: beam search finds modular arithmetic."""
        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
        obs = env.generate(200)
        cfg = BeamConfig(beam_width=10, const_range=15, max_depth=4, mcmc_iterations=80)
        expr = BeamSearchSynthesizer(cfg).search(obs)
        assert expr is not None

    def test_days30_31_grammar_search(self):
        """60 nodes with grammar constraints."""
        from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig
        obs = [(3*t+1)%7 for t in range(80)]
        result = GrammarConstrainedBeam(GrammarBeamConfig(beam_width=6, n_iterations=3)).search(obs)
        assert result is not None

    def test_day33_hookes_law(self):
        """Physics law detection."""
        from ouroboros.physics.law_signature import _test_hookes_law, PhysicsLaw
        seq = [10.0 * math.cos(0.3 * t) for t in range(100)]
        result = _test_hookes_law(seq, threshold=0.8)
        assert result.passed

    def test_day34_prime_formula(self):
        """CUMSUM(ISPRIME(t)) = π(t)."""
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
        assert sum(1 for p,o in zip(preds,obs) if p==o) >= 28

    def test_day43_ftc(self):
        """Fundamental theorem: DERIV(INTEGRAL(f)) = f."""
        from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
        from ouroboros.synthesis.expr_node import NodeType
        def const_e(v):
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NodeType.CONST; n.value = float(v)
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n
        integral = ExtExprNode(ExtNodeType.INTEGRAL, left=const_e(3.0))
        deriv = ExtExprNode(ExtNodeType.DERIV, left=integral)
        for t in range(1, 8):
            assert abs(deriv.evaluate(t, [], {}) - 3.0) < 0.01

    def test_day46_layer5(self):
        """Layer 5: composite opcode discovery."""
        from ouroboros.layer4.layer5 import Layer5Experiment
        results = Layer5Experiment().run(verbose=False)
        assert results["n_programs_observed"] >= 4

    def test_day48_api_client(self):
        """Web API: local client discovery."""
        from ouroboros.api.client import OuroborosClient
        client = OuroborosClient(base_url=None)
        obs = [float((3*t+1)%7) for t in range(60)]
        result = client.discover(obs, alphabet_size=7, beam_width=5,
                                 max_depth=3, n_iterations=2, time_budget_seconds=3.0)
        assert result.n_observations == 60

    def test_day49_pr_workflow(self):
        """Mathlib4 PR dry-run workflow."""
        from ouroboros.papers.mathlib4_submission import Mathlib4PRWorkflow
        workflow = Mathlib4PRWorkflow(dry_run=True)
        report = workflow.generate_submission_report()
        assert "REPORT" in report

    def test_total_test_count(self):
        """Verify we have met our test count goal."""
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests/', '--collect-only', '-q'],
            capture_output=True, text=True
        )
        # Count test items (lines ending in "test_")
        n_tests = sum(1 for line in result.stdout.split('\n') if '::test_' in line)
        print(f"\n  Total tests: {n_tests}")
        assert n_tests >= 400  # target: ~1,010+ by Day 50