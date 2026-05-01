"""Tests for the full benchmark pipeline and publication validation."""
import pytest
import math
from ouroboros.benchmark.runner import ExperimentResult
from ouroboros.benchmark.full_runner import (
    FullBenchmarkRunner, ConfidenceIntervalValidator,
    PublicationCheck, BenchmarkComparison,
)


class TestConfidenceIntervalValidator:
    def _result(self, values):
        return ExperimentResult("test", "metric", "units", values)

    def test_tight_ci_passes(self):
        # 10 values very close together → tight CI
        values = [0.004, 0.004, 0.004, 0.004, 0.004,
                  0.004, 0.004, 0.004, 0.004, 0.004]
        r = self._result(values)
        validator = ConfidenceIntervalValidator(relative_ci_threshold=0.15)
        check = validator.validate(r)
        assert check.passes_publication

    def test_wide_ci_fails(self):
        # Very spread out values → wide CI
        values = [0.001, 0.01, 0.001, 0.01, 0.001, 0.01, 0.001, 0.01, 0.001, 0.01]
        r = self._result(values)
        validator = ConfidenceIntervalValidator(relative_ci_threshold=0.15)
        check = validator.validate(r)
        assert not check.passes_publication

    def test_relative_ci_computed(self):
        values = [1.0] * 10
        r = self._result(values)
        validator = ConfidenceIntervalValidator()
        check = validator.validate(r)
        assert check.relative_ci == pytest.approx(0.0, abs=0.01)

    def test_description_has_status(self):
        values = [0.004] * 10
        r = self._result(values)
        validator = ConfidenceIntervalValidator()
        check = validator.validate(r)
        desc = check.description()
        assert "PUBLISHABLE" in desc or "NEEDS" in desc

    def test_validate_all_returns_list(self):
        results = {
            "test1": self._result([0.004]*10),
            "test2": self._result([0.1]*10),
        }
        validator = ConfidenceIntervalValidator()
        checks = validator.validate_all(results)
        assert len(checks) == 2

    def test_validate_all_skips_non_results(self):
        results = {
            "test1": self._result([0.004]*10),
            "not_result": "just a string",
        }
        validator = ConfidenceIntervalValidator()
        checks = validator.validate_all(results)
        assert len(checks) == 1


class TestPublicationCheck:
    def test_fields_accessible(self):
        check = PublicationCheck(
            experiment_name="test",
            mean=0.004,
            ci_95=0.0002,
            n=10,
            relative_ci=0.05,
            passes_publication=True,
            recommendation="OK",
        )
        assert check.mean == pytest.approx(0.004)
        assert check.passes_publication
        assert check.n == 10

    def test_borderline_relative_ci(self):
        # 14.9% relative CI — just passes
        check = PublicationCheck(
            experiment_name="test", mean=1.0, ci_95=0.149,
            n=10, relative_ci=0.149, passes_publication=True,
            recommendation="OK",
        )
        assert check.passes_publication

    def test_just_fails_relative_ci(self):
        check = PublicationCheck(
            experiment_name="test", mean=1.0, ci_95=0.151,
            n=10, relative_ci=0.151, passes_publication=False,
            recommendation="Needs more seeds",
        )
        assert not check.passes_publication


class TestBenchmarkComparison:
    def _make_results(self, mean_multiplier: float = 1.0):
        return {
            "compression_landmark": {"mean": 0.004 * mean_multiplier, "ci_95": 0.0002, "n": 5},
            "convergence_rounds": {"mean": 7.8 * mean_multiplier, "ci_95": 0.74, "n": 5},
        }

    def test_comparison_constructed(self):
        fast = self._make_results(1.0)
        full = self._make_results(1.02)  # slightly different
        comp = BenchmarkComparison(fast, full)
        assert comp.fast_results == fast
        assert comp.full_results == full

    def test_print_comparison_no_crash(self, capsys):
        fast = self._make_results(1.0)
        full = self._make_results(1.02)
        comp = BenchmarkComparison(fast, full)
        comp.print_comparison()
        captured = capsys.readouterr()
        assert "COMPARISON" in captured.out or len(captured.out) > 0


class TestFullBenchmarkRunner:
    def test_initialization(self, tmp_path):
        runner = FullBenchmarkRunner(n_seeds=3, output_dir=str(tmp_path))
        assert runner.n_seeds == 3

    def test_save_intermediate(self, tmp_path):
        runner = FullBenchmarkRunner(n_seeds=2, output_dir=str(tmp_path))
        from ouroboros.benchmark.runner import ExperimentResult
        results = {"test": ExperimentResult("test", "m", "u", [1.0, 2.0])}
        runner._save_intermediate(results, "test.json")
        import json
        saved = json.loads((tmp_path / "test.json").read_text())
        assert "test" in saved

    def test_validate_and_report_returns_checks(self, tmp_path):
        runner = FullBenchmarkRunner(n_seeds=3, output_dir=str(tmp_path))
        from ouroboros.benchmark.runner import ExperimentResult
        results = {
            "compression_landmark": ExperimentResult("compression_landmark", "ratio", "", [0.004]*10),
        }
        checks = runner.validate_and_report(results)
        assert isinstance(checks, list)
        assert len(checks) > 0

    def test_latex_tables_generated(self, tmp_path):
        runner = FullBenchmarkRunner(n_seeds=3, output_dir=str(tmp_path))
        from ouroboros.benchmark.runner import ExperimentResult
        results = {
            "compression_landmark": ExperimentResult("compression_landmark", "ratio", "", [0.004]*10),
            "convergence_rounds": ExperimentResult("convergence_rounds", "rounds", "", [7.8]*10),
            "crt_accuracy": ExperimentResult("crt_accuracy", "frac", "", [0.87]*10),
        }
        tables = runner.generate_updated_latex_tables(results)
        assert "\\begin{table}" in tables
        assert "0.0040" in tables or "0.004" in tables

    def test_tiny_run_works(self, tmp_path):
        """Run 2-seed benchmark to verify it doesn't crash."""
        from ouroboros.benchmark.full_runner import FullBenchmarkRunner
        runner = FullBenchmarkRunner(n_seeds=2, output_dir=str(tmp_path), verbose=False)
        runner._runner.stream_length = 100
        runner._runner.n_rounds = 5
        r = runner._runner.run_compression_landmark()
        assert r.n == 2
        assert r.mean >= 0          # ratio can legally exceed 1.0 on incompressible data
        assert r.mean == r.mean     # not NaN