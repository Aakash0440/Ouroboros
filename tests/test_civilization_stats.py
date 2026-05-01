"""Tests for civilization simulation statistical analysis."""
import pytest
import statistics
from ouroboros.civilization.statistics import (
    spearman_rho, bootstrap_spearman_ci, SpearmanBootstrapResult,
    CivilizationStatisticalReport, CivilizationStatisticalAnalyzer,
)


class TestSpearmanRho:
    def test_identical_orders(self):
        order = ["A", "B", "C", "D", "E"]
        assert abs(spearman_rho(order, order) - 1.0) < 0.01

    def test_reversed_orders(self):
        order = ["A", "B", "C", "D", "E"]
        rho = spearman_rho(order, order[::-1])
        assert rho < -0.8

    def test_partial_overlap(self):
        a = ["A", "B", "C", "D"]
        b = ["C", "A", "B", "D"]
        rho = spearman_rho(a, b)
        assert -1.0 <= rho <= 1.0

    def test_too_short_returns_zero(self):
        assert spearman_rho(["A"], ["A"]) == 0.0
        assert spearman_rho([], []) == 0.0

    def test_no_common_elements(self):
        assert spearman_rho(["A", "B"], ["C", "D"]) == 0.0

    def test_known_correlation(self):
        # Rank permutation: [0,1,2,3,4] → [1,0,2,3,4] — one swap
        a = ["A", "B", "C", "D", "E"]
        b = ["B", "A", "C", "D", "E"]
        rho = spearman_rho(a, b)
        assert rho > 0.7  # mostly aligned


class TestBootstrapSpearmanCI:
    def _make_orders(self, n: int, consistent: bool) -> list:
        if consistent:
            return [["A", "B", "C", "D", "E"]] * n
        return [["A", "B", "C", "D", "E"],
                ["B", "A", "C", "D", "E"],
                ["A", "C", "B", "D", "E"]] * (n // 3 + 1)[:n]

    def test_consistent_orders_positive_ci(self):
        orders = self._make_orders(5, consistent=True)
        human = ["A", "B", "C", "D", "E"]
        result = bootstrap_spearman_ci(orders, human, n_bootstrap=200, seed=42)
        assert result.observed_rho > 0.9
        assert result.ci_lower > 0.5  # significantly positive

    def test_ci_is_ordered(self):
        orders = self._make_orders(4, consistent=True)
        human = ["A", "B", "C", "D", "E"]
        result = bootstrap_spearman_ci(orders, human, n_bootstrap=100, seed=42)
        assert result.ci_lower <= result.observed_rho
        assert result.observed_rho <= result.ci_upper + 0.1  # allow small numerical error

    def test_single_run(self):
        orders = [["A", "B", "C"]]
        human = ["A", "B", "C"]
        result = bootstrap_spearman_ci(orders, human, n_bootstrap=100, seed=42)
        assert isinstance(result.observed_rho, float)
        assert result.n_runs == 1

    def test_result_fields(self):
        orders = [["A", "B", "C"]] * 3
        human = ["A", "B", "C"]
        result = bootstrap_spearman_ci(orders, human, n_bootstrap=50, seed=42)
        assert result.n_bootstrap == 50
        assert result.n_runs == 3
        assert isinstance(result.is_significant, bool)

    def test_ci_width_positive(self):
        orders = [["A", "B", "C"]] * 5
        human = ["A", "B", "C"]
        result = bootstrap_spearman_ci(orders, human, n_bootstrap=100, seed=42)
        assert result.ci_width >= 0

    def test_latex_str_format(self):
        orders = [["A", "B", "C"]] * 3
        human = ["A", "B", "C"]
        result = bootstrap_spearman_ci(orders, human, n_bootstrap=50, seed=42)
        s = result.latex_str()
        assert "\\rho" in s or "rho" in s.lower()
        assert "CI" in s


class TestCivilizationStatisticalReport:
    def _make_report(self) -> CivilizationStatisticalReport:
        from ouroboros.civilization.statistics import SpearmanBootstrapResult
        sr = SpearmanBootstrapResult(
            observed_rho=0.71,
            bootstrap_rhos=[0.65, 0.71, 0.77] * 100,
            ci_lower=0.52,
            ci_upper=0.84,
            n_bootstrap=300,
            n_runs=5,
        )
        return CivilizationStatisticalReport(
            n_runs=5, n_agents=16, n_rounds=30,
            n_concepts_discovered={0: 8, 1: 7, 2: 9, 3: 8, 4: 7},
            spearman_result=sr,
            discovery_consistency=0.75,
            mean_convergence_round=18.0,
        )

    def test_description_is_string(self):
        report = self._make_report()
        s = report.description()
        assert isinstance(s, str) and len(s) > 50

    def test_latex_section_has_rho(self):
        report = self._make_report()
        latex = report.latex_section()
        assert "rho" in latex.lower() or "\\rho" in latex
        assert "0.71" in latex

    def test_latex_section_has_ci(self):
        report = self._make_report()
        latex = report.latex_section()
        assert "CI" in latex or "ci" in latex.lower()


class TestCivilizationStatisticalAnalyzer:
    def test_tiny_analysis_runs(self):
        analyzer = CivilizationStatisticalAnalyzer(
            n_runs=2, n_agents=4, n_rounds=5,
            stream_length=80, beam_width=5, n_iterations=2,
            n_bootstrap=100,
        )
        report = analyzer.run_full_analysis(verbose=False)
        assert isinstance(report, CivilizationStatisticalReport)
        assert report.n_runs == 2

    def test_bootstrap_ci_computed(self):
        analyzer = CivilizationStatisticalAnalyzer(
            n_runs=2, n_agents=4, n_rounds=5,
            stream_length=80, beam_width=5, n_iterations=2,
            n_bootstrap=50,
        )
        report = analyzer.run_full_analysis(verbose=False)
        assert report.spearman_result.n_bootstrap == 50
        assert -1.0 <= report.spearman_result.observed_rho <= 1.0

    def test_consistency_in_range(self):
        analyzer = CivilizationStatisticalAnalyzer(
            n_runs=2, n_agents=4, n_rounds=5,
            stream_length=80, beam_width=5, n_iterations=2,
            n_bootstrap=50,
        )
        report = analyzer.run_full_analysis(verbose=False)
        assert 0.0 <= report.discovery_consistency <= 1.0