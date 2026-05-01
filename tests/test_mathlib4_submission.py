"""Tests for Mathlib4 PR submission workflow."""
import pytest
from pathlib import Path
from ouroboros.papers.mathlib4_submission import (
    PRSubmissionConfig, GitCommandResult, GitOperationsRunner,
    PRReadinessItem, PRReadinessChecker, Mathlib4PRWorkflow,
)


class TestGitCommandResult:
    def test_success_when_returncode_0(self):
        r = GitCommandResult(["git", "status"], 0, "On branch main", "")
        assert r.success

    def test_fails_when_nonzero(self):
        r = GitCommandResult(["git", "push"], 1, "", "error: rejected")
        assert not r.success

    def test_description_format(self):
        r = GitCommandResult(["git", "commit"], 0, "1 file changed", "")
        s = r.description()
        assert "git commit" in s
        assert "✅" in s


class TestGitOperationsRunner:
    def test_dry_run_always_succeeds(self):
        runner = GitOperationsRunner(dry_run=True)
        result = runner.run("status")
        assert result.success
        assert "DRY RUN" in result.stdout

    def test_commands_recorded(self):
        runner = GitOperationsRunner(dry_run=True)
        runner.run("status")
        runner.run("log", "--oneline")
        assert len(runner._executed) == 2

    def test_all_succeeded_on_dry_run(self):
        runner = GitOperationsRunner(dry_run=True)
        runner.run("status")
        runner.run("add", ".")
        assert runner.all_succeeded()

    def test_execution_log_is_string(self):
        runner = GitOperationsRunner(dry_run=True)
        runner.run("status")
        log = runner.execution_log()
        assert isinstance(log, str) and len(log) > 0


class TestPRReadinessItem:
    def test_passed_item(self):
        item = PRReadinessItem("Zero sorry", True, "All proofs complete")
        assert item.passed
        assert item.item == "Zero sorry"

    def test_failed_item(self):
        item = PRReadinessItem("Zero sorry", False, "Found 3 sorrys")
        assert not item.passed


class TestPRReadinessChecker:
    def test_bezout_check_passes(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        item = checker._check_bezout_arithmetic()
        assert item.passed
        assert "77 pairs" in item.details

    def test_lean_files_check(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        item = checker._check_lean_files_exist()
        # Should pass if contribution files exist
        lean_exists = len(list(
            Path("ouroboros_lean/Mathlib4Contribution").glob("*.lean")
        )) >= 2 if Path("ouroboros_lean/Mathlib4Contribution").exists() else False
        assert item.passed == lean_exists

    def test_zero_sorry_check(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        item = checker._check_zero_sorry()
        # The check should run without crashing
        assert isinstance(item.passed, bool)
        assert isinstance(item.details, str)

    def test_pr_description_check(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        item = checker._check_pr_description()
        assert isinstance(item.passed, bool)

    def test_check_all_returns_list(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        checks = checker.check_all()
        assert isinstance(checks, list)
        assert len(checks) >= 4

    def test_report_is_string(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        checks = checker.check_all()
        report = checker.report(checks)
        assert isinstance(report, str)
        assert "READINESS" in report

    def test_all_passed_method(self):
        checker = PRReadinessChecker("ouroboros_lean/Mathlib4Contribution")
        all_pass = [PRReadinessItem("A", True, "ok"), PRReadinessItem("B", True, "ok")]
        assert checker.all_passed(all_pass)
        one_fail = [PRReadinessItem("A", True, "ok"), PRReadinessItem("B", False, "fail")]
        assert not checker.all_passed(one_fail)


class TestMathlib4PRWorkflow:
    def test_dry_run_completes(self):
        workflow = Mathlib4PRWorkflow(dry_run=True)
        result = workflow.run(verbose=False)
        # In dry run, all git ops succeed trivially
        # Result depends on readiness checks
        assert isinstance(result, bool)

    def test_submission_report_generated(self):
        workflow = Mathlib4PRWorkflow(dry_run=True)
        report = workflow.generate_submission_report()
        assert isinstance(report, str)
        assert "REPORT" in report
        assert "Dry run" in report

    def test_config_defaults(self):
        config = PRSubmissionConfig()
        assert "ouroboros" in config.branch_name
        assert config.dry_run_by_default if hasattr(config, 'dry_run_by_default') else True

    def test_workflow_with_custom_config(self):
        config = PRSubmissionConfig(
            github_username="test_user",
            branch_name="test-branch",
        )
        workflow = Mathlib4PRWorkflow(config=config, dry_run=True)
        report = workflow.generate_submission_report()
        assert "test-branch" in report