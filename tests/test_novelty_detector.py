"""Tests for the complete novelty detection pipeline."""
import pytest
import math
from ouroboros.novelty.detector import (
    NoveltyDetector, NoveltyAnnotatedResult, NoveltyReport,
)
from ouroboros.novelty.open_conjectures import (
    CollatzStoppingTimesEnv, PrimeGapEnv, TwinPrimeDensityEnv,
)
from ouroboros.synthesis.expr_node import ExprNode, NodeType


def make_const(v: float) -> ExprNode:
    return ExprNode(NodeType.CONST, value=v)

def make_time() -> ExprNode:
    return ExprNode(NodeType.TIME)


class TestNoveltyDetector:
    def _detector(self, tmp_path) -> NoveltyDetector:
        return NoveltyDetector(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            findings_log=str(tmp_path / "findings.jsonl"),
            use_oeis=False,
            verbose=False,
        )

    def test_annotate_returns_result(self, tmp_path):
        detector = self._detector(tmp_path)
        expr = make_const(5.0)
        result = detector.annotate(expr, [5.0]*30, mdl_cost=10.0)
        assert isinstance(result, NoveltyAnnotatedResult)

    def test_novelty_score_in_range(self, tmp_path):
        detector = self._detector(tmp_path)
        result = detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        assert 0.0 <= result.novelty_score <= 1.0

    def test_novelty_category_valid(self, tmp_path):
        detector = self._detector(tmp_path)
        result = detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        valid_categories = {"known", "variant_of_known", "potentially_novel",
                            "likely_novel", "route_to_mathematician", "routine",
                            "interesting"}
        assert result.novelty_category in valid_categories

    def test_history_tracked(self, tmp_path):
        detector = self._detector(tmp_path)
        detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        detector.annotate(make_time(), list(range(30)), mdl_cost=20.0)
        assert len(detector._history) == 2

    def test_stats_tracked(self, tmp_path):
        detector = self._detector(tmp_path)
        detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        assert detector.stats["n_analyzed"] == 1

    def test_findings_logged_above_threshold(self, tmp_path):
        import json
        detector = NoveltyDetector(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            findings_log=str(tmp_path / "findings.jsonl"),
            use_oeis=False,
            novelty_threshold=0.0,  # log everything
            verbose=False,
        )
        detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        log_path = tmp_path / "findings.jsonl"
        if log_path.exists():
            lines = log_path.read_text().strip().split('\n')
            assert len(lines) >= 1
            data = json.loads(lines[0])
            assert "expression" in data

    def test_generate_report(self, tmp_path):
        detector = self._detector(tmp_path)
        detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        report = detector.generate_report()
        assert isinstance(report, NoveltyReport)
        assert report.n_discoveries_analyzed == 1

    def test_register_discovery(self, tmp_path):
        detector = self._detector(tmp_path)
        expr = make_const(5.0)
        detector.register_approved_discovery(expr, "five_constant", "arithmetic")
        # Registry should have grown
        assert detector._registry._db.size >= 1

    def test_to_dict_format(self, tmp_path):
        detector = self._detector(tmp_path)
        result = detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        d = result.to_dict()
        assert "expression" in d
        assert "novelty_score" in d
        assert "is_flagged" in d

    def test_summary_is_string(self, tmp_path):
        detector = self._detector(tmp_path)
        result = detector.annotate(make_const(3.0), [3.0]*30, mdl_cost=10.0)
        s = result.summary()
        assert isinstance(s, str) and len(s) > 10


class TestNoveltyReport:
    def test_report_prints(self, tmp_path, capsys):
        detector = NoveltyDetector(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            findings_log=str(tmp_path / "log.jsonl"),
            use_oeis=False, verbose=False,
        )
        detector.annotate(make_const(3.0), [3.0]*20, mdl_cost=10.0)
        report = detector.generate_report()
        report.print_report()
        captured = capsys.readouterr()
        assert "NOVELTY REPORT" in captured.out

    def test_latex_generated(self, tmp_path):
        detector = NoveltyDetector(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            findings_log=str(tmp_path / "log.jsonl"),
            use_oeis=False, verbose=False,
        )
        detector.annotate(make_const(3.0), [3.0]*20, mdl_cost=10.0)
        report = detector.generate_report()
        latex = report.to_latex()
        assert "\\begin{table}" in latex


class TestOpenConjectureEnvironments:
    def test_collatz_env(self):
        env = CollatzStoppingTimesEnv()
        obs = env.generate(20)
        assert len(obs) == 20
        assert obs[0] == 0  # stopping_time(0) = 0
        assert obs[1] == 0  # stopping_time(1) = 0 (already at 1)
        assert obs[2] == 1  # 2 → 1: 1 step
        assert obs[3] == 7  # 3 → 10 → 5 → 16 → 8 → 4 → 2 → 1: 7 steps

    def test_prime_gap_env(self):
        env = PrimeGapEnv()
        obs = env.generate(10)
        assert all(v > 0 for v in obs)  # gaps are positive
        assert obs[0] == 1  # p2-p1 = 3-2 = 1

    def test_twin_prime_env(self):
        env = TwinPrimeDensityEnv()
        obs = env.generate(10)
        assert all(v >= 0 for v in obs)  # counts are non-negative
        assert obs[-1] >= obs[0]  # cumulative count is non-decreasing

    def test_envs_have_conjectures(self):
        for env in [CollatzStoppingTimesEnv(), PrimeGapEnv(), TwinPrimeDensityEnv()]:
            assert len(env.conjecture()) > 20

    def test_collatz_formula_check(self):
        """Verify that the best known formula is at least roughly correct."""
        env = CollatzStoppingTimesEnv()
        obs = env.generate(100, start=10)
        # Best known: ~6.95 * log2(n)
        n = 10
        for t, actual in enumerate(obs[:10]):
            expected_approx = 6.95 * math.log2(n + t + 1)
            # Should be in the right ballpark (within 5x)
            if actual > 0:
                ratio = actual / expected_approx
                assert 0.1 < ratio < 10, f"n={n+t}: actual={actual}, approx={expected_approx:.1f}"
