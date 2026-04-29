"""Tests for the automated paper writing pipeline."""
import pytest
from pathlib import Path
from ouroboros.papers.paper_writer import (
    ExperimentNumbers, generate_paper1, generate_paper2, _fmt,
)


class TestExperimentNumbers:
    def test_defaults_reasonable(self):
        nums = ExperimentNumbers()
        assert 0 < nums.compression_ratio_mean < 0.1
        assert nums.convergence_rounds_mean > 1
        assert 0 < nums.crt_success_rate <= 1.0

    def test_from_empty_json(self, tmp_path):
        path = str(tmp_path / "results.json")
        Path(path).write_text("{}")
        nums = ExperimentNumbers.from_benchmark_json(path)
        assert isinstance(nums, ExperimentNumbers)

    def test_from_real_json(self, tmp_path):
        import json
        data = {
            "compression_landmark": {"mean": 0.003, "ci_95": 0.001, "n": 5},
            "convergence_rounds": {"mean": 6.5, "ci_95": 0.8, "n": 5},
        }
        path = str(tmp_path / "results.json")
        Path(path).write_text(json.dumps(data))
        nums = ExperimentNumbers.from_benchmark_json(path)
        assert abs(nums.compression_ratio_mean - 0.003) < 0.001
        assert abs(nums.convergence_rounds_mean - 6.5) < 0.01

    def test_from_nonexistent_path(self):
        nums = ExperimentNumbers.from_benchmark_json("/nonexistent/path.json")
        assert isinstance(nums, ExperimentNumbers)


class TestFormatHelper:
    def test_fmt_basic(self):
        s = _fmt(0.0041, 0.0002)
        assert "0.0041" in s
        assert "0.0002" in s
        assert "pm" in s or "\\pm" in s

    def test_fmt_decimals(self):
        s = _fmt(7.8, 0.74, decimals=2)
        assert "7.80" in s
        assert "0.74" in s

    def test_fmt_returns_string(self):
        assert isinstance(_fmt(1.0, 0.1), str)


class TestPaperGeneration:
    def test_paper1_generates_file(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper1(nums, str(tmp_path))
        assert Path(path).exists()

    def test_paper1_is_latex(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper1(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "\\documentclass" in content
        assert "\\begin{document}" in content
        assert "\\end{document}" in content

    def test_paper1_has_numbers(self, tmp_path):
        nums = ExperimentNumbers()
        nums.compression_ratio_mean = 0.0041
        nums.convergence_rounds_mean = 7.8
        path = generate_paper1(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "0.0041" in content

    def test_paper1_no_placeholders(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper1(nums, str(tmp_path))
        content = Path(path).read_text()
        # Should not have TODO or PLACEHOLDER
        assert "TODO" not in content
        assert "PLACEHOLDER" not in content

    def test_paper2_generates_file(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper2(nums, str(tmp_path))
        assert Path(path).exists()

    def test_paper2_has_four_layers(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper2(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "Layer 0" in content
        assert "Layer 4" in content

    def test_paper2_has_convergence_numbers(self, tmp_path):
        nums = ExperimentNumbers()
        nums.convergence_rounds_mean = 7.8
        path = generate_paper2(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "7.8" in content or "7.80" in content

    def test_both_papers_different_content(self, tmp_path):
        nums = ExperimentNumbers()
        path1 = generate_paper1(nums, str(tmp_path))
        path2 = generate_paper2(nums, str(tmp_path))
        c1 = Path(path1).read_text()
        c2 = Path(path2).read_text()
        assert c1 != c2

    def test_paper1_mentions_60_nodes(self, tmp_path):
        nums = ExperimentNumbers(n_node_types=60)
        path = generate_paper1(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "60" in content

    def test_paper1_mentions_lean4(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper1(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "Lean4" in content or "lean4" in content.lower()

    def test_paper2_mentions_sha256(self, tmp_path):
        nums = ExperimentNumbers()
        path = generate_paper2(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "SHA-256" in content or "hash" in content.lower()

    def test_tables_have_real_numbers(self, tmp_path):
        nums = ExperimentNumbers()
        nums.physics_hookes_law_corr = -0.94
        nums.physics_decay_corr = -0.97
        path = generate_paper1(nums, str(tmp_path))
        content = Path(path).read_text()
        assert "-0.94" in content
        assert "-0.97" in content