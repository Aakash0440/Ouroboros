"""Tests for benchmark runner and report generation."""
import pytest
import json
from ouroboros.benchmark.runner import ExperimentResult, BenchmarkRunner
from ouroboros.benchmark.report import (
    generate_latex_table_1, generate_latex_table_2, PaperNumbersReport,
)


class TestExperimentResult:
    def _make(self, values):
        return ExperimentResult("test", "metric", "units", values)

    def test_mean_computed(self):
        r = self._make([1.0, 2.0, 3.0])
        assert r.mean == pytest.approx(2.0)

    def test_std_computed(self):
        r = self._make([1.0, 2.0, 3.0, 4.0])
        assert r.std > 0

    def test_ci_95_single_value(self):
        r = self._make([1.0])
        assert r.ci_95() == float('inf')

    def test_ci_95_multiple(self):
        r = self._make([1.0, 2.0, 3.0, 4.0, 5.0])
        ci = r.ci_95()
        assert ci > 0 and ci < 10

    def test_latex_str_format(self):
        r = self._make([0.004, 0.005, 0.003, 0.004, 0.004])
        s = r.latex_str()
        assert "\\pm" in s
        assert "0.00" in s

    def test_summary_str_is_string(self):
        r = self._make([1.0, 2.0])
        s = r.summary_str()
        assert isinstance(s, str) and len(s) > 0

    def test_median_correct(self):
        r = self._make([1.0, 3.0, 5.0])
        assert r.median == pytest.approx(3.0)

    def test_min_max_correct(self):
        r = self._make([2.0, 5.0, 1.0, 4.0])
        assert r.min_val == pytest.approx(1.0)
        assert r.max_val == pytest.approx(5.0)


class TestLatexGeneration:
    def _make_results(self):
        return {
            "compression_landmark": {
                "mean": 0.0041, "std": 0.0003, "ci_95": 0.0002,
                "min": 0.003, "max": 0.006, "n": 10, "values": [],
                "latex": "0.0041 \\pm 0.0002",
            },
            "convergence_rounds": {
                "mean": 7.8, "std": 1.2, "ci_95": 0.74,
                "min": 5.0, "max": 10.0, "n": 10, "values": [],
                "latex": "7.8000 \\pm 0.7400",
            },
            "crt_accuracy": {
                "mean": 0.87, "std": 0.11, "ci_95": 0.07,
                "min": 0.7, "max": 1.0, "n": 10, "values": [],
                "latex": "0.8700 \\pm 0.0700",
            },
        }

    def test_table_1_contains_latex(self):
        r = self._make_results()
        table = generate_latex_table_1(r)
        assert r"\begin{table}" in table
        assert r"\end{table}" in table
        assert r"\toprule" in table

    def test_table_2_contains_convergence(self):
        r = self._make_results()
        table = generate_latex_table_2(r)
        assert "7.80" in table or "7.8" in table
        assert r"\end{table}" in table

    def test_paper_numbers_report_has_numbers(self):
        r = self._make_results()
        report = PaperNumbersReport(r).generate()
        assert "0.0041" in report
        assert "Paper 1" in report
        assert "Paper 2" in report

    def test_paper_numbers_report_has_all_sections(self):
        r = self._make_results()
        report = PaperNumbersReport(r).generate()
        assert "Compression Ratio" in report
        assert "CRT" in report
        assert "Self-Improvement" in report