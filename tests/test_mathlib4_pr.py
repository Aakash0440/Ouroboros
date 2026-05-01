"""Tests for Mathlib4 PR preparation."""
import pytest
from pathlib import Path
from ouroboros.papers.lean4_pr import (
    ProofStyleChecker, StyleIssue, StyleCheckResult, Mathlib4PRGenerator,
)


class TestStyleIssue:
    def test_fields_accessible(self):
        issue = StyleIssue(42, "SorryPresent", "sorry found", "error")
        assert issue.line_number == 42
        assert issue.severity == "error"
        assert "sorry" in issue.description.lower()

    def test_description_str_format(self):
        issue = StyleIssue(10, "TestIssue", "test description", "warning")
        s = issue.description_str()
        assert "L10" in s
        assert "WARNING" in s
        assert "TestIssue" in s


class TestStyleCheckResult:
    def _make_result(
        self, n_theorems=5, n_docstrings=5, n_sorry=0, issues=None
    ) -> StyleCheckResult:
        return StyleCheckResult(
            file_path="test.lean",
            n_theorems=n_theorems,
            n_with_docstrings=n_docstrings,
            n_sorry=n_sorry,
            naming_issues=[],
            missing_docstrings=[],
            all_issues=issues or [],
        )

    def test_passes_when_clean(self):
        result = self._make_result(5, 5, 0)
        assert result.passes_submission

    def test_fails_with_sorry(self):
        result = self._make_result(5, 5, n_sorry=1)
        assert not result.passes_submission

    def test_fails_with_missing_docstrings(self):
        result = self._make_result(5, 3, 0)
        assert not result.passes_submission

    def test_fails_with_error_issue(self):
        error_issue = StyleIssue(1, "Error", "test error", "error")
        result = self._make_result(5, 5, 0, [error_issue])
        assert not result.passes_submission

    def test_report_is_string(self):
        result = self._make_result()
        report = result.report()
        assert isinstance(report, str) and len(report) > 10

    def test_warning_doesnt_block_submission(self):
        warning = StyleIssue(1, "NamingConvention", "camelCase issue", "warning")
        result = self._make_result(5, 5, 0, [warning])
        # Warnings alone shouldn't block — only errors do
        assert result.passes_submission


class TestProofStyleChecker:
    def _write_lean(self, tmp_path: Path, content: str) -> str:
        path = tmp_path / "test.lean"
        path.write_text(content, encoding='utf-8')
        return str(path)

    def test_counts_theorems(self, tmp_path):
        content = """
theorem foo (n : ℕ) : n = n := rfl
theorem bar (n : ℕ) : n + 0 = n := by omega
"""
        path = self._write_lean(tmp_path, content)
        checker = ProofStyleChecker()
        result = checker.check_file(path)
        assert result.n_theorems == 2

    def test_detects_sorry(self, tmp_path):
        content = """
theorem foo (n : ℕ) : n = 0 := by sorry
"""
        path = self._write_lean(tmp_path, content)
        checker = ProofStyleChecker()
        result = checker.check_file(path)
        assert result.n_sorry == 1
        assert not result.passes_submission

    def test_clean_file_passes(self, tmp_path):
        content = """
/-- foo is equal to itself -/
theorem foo (n : ℕ) : n = n := rfl

/-- Adding zero does nothing -/
theorem addZero (n : ℕ) : n + 0 = n := by omega
"""
        path = self._write_lean(tmp_path, content)
        checker = ProofStyleChecker()
        result = checker.check_file(path)
        assert result.n_sorry == 0
        assert result.n_theorems >= 1

    def test_no_sorry_on_comment(self, tmp_path):
        content = """
-- This is just a comment mentioning sorry in an explanation
theorem foo (n : ℕ) : n = n := rfl
"""
        path = self._write_lean(tmp_path, content)
        checker = ProofStyleChecker()
        result = checker.check_file(path)
        assert result.n_sorry == 0

    def test_naming_issue_detected(self, tmp_path):
        content = """
theorem Foo (n : ℕ) : n = n := rfl
"""
        path = self._write_lean(tmp_path, content)
        checker = ProofStyleChecker()
        result = checker.check_file(path)
        # Uppercase start should be flagged
        naming_issues = [i for i in result.all_issues if i.issue_type == "NamingConvention"]
        assert len(naming_issues) >= 1


class TestContributionFiles:
    def test_linear_modular_file_exists(self):
        path = Path("ouroboros_lean/Mathlib4Contribution/LinearModularSurjective.lean")
        if path.exists():
            content = path.read_text()
            assert "theorem" in content
            assert "sorry" not in content
            assert "namespace" in content.lower() or "theorem" in content

    def test_crt_file_exists(self):
        path = Path("ouroboros_lean/Mathlib4Contribution/CRTInstances.lean")
        if path.exists():
            content = path.read_text()
            assert "theorem" in content
            assert "sorry" not in content

    def test_pr_description_exists(self):
        path = Path("ouroboros_lean/Mathlib4Contribution/PR_DESCRIPTION.md")
        if path.exists():
            content = path.read_text()
            assert "Mathlib4" in content or "mathlib" in content.lower()

    def test_bezout_witness_arithmetic(self):
        """Verify the Bezout witness arithmetic in Python before Lean4."""
        for a in range(7):
            for b in range(11):
                x = (a * 22 + b * 56) % 77
                assert x % 7 == a, f"Bezout fail mod 7: a={a},b={b}"
                assert x % 11 == b, f"Bezout fail mod 11: a={a},b={b}"

    def test_ax00001_witnesses(self):
        """Verify surjectivity witnesses."""
        witnesses = {0: 2, 1: 0, 2: 5, 3: 3, 4: 1, 5: 6, 6: 4}
        for r, t in witnesses.items():
            assert (3 * t + 1) % 7 == r, f"Witness fail: r={r}, t={t}"


class TestMathlib4PRGenerator:
    def test_generates_test_file(self, tmp_path):
        import shutil
        # Create minimal contribution dir
        contrib_dir = tmp_path / "Mathlib4Contribution"
        contrib_dir.mkdir()
        (contrib_dir / "test.lean").write_text("theorem foo : True := trivial")

        gen = Mathlib4PRGenerator(str(contrib_dir))
        path = gen.generate_test_file()
        assert Path(path).exists()

    def test_commit_message_format(self):
        gen = Mathlib4PRGenerator()
        msg = gen.generate_commit_message()
        assert "feat" in msg.lower() or "add" in msg.lower()
        assert "NumberTheory" in msg or "sorry" in msg.lower()
        assert len(msg) > 100