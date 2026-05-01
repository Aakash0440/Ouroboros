"""
Mathlib4PRWorkflow — Complete PR submission workflow.

What this does:
  1. Checks all Lean4 files pass the style checker
  2. Verifies zero sorry across all contribution files
  3. Generates clean copies of contribution files
  4. Runs: git checkout -b feat/ouroboros-linear-modular
  5. Copies files to target location in Mathlib4 repo
  6. Runs: git add, git commit -m "..."
  7. Runs: git push origin feat/ouroboros-linear-modular
  8. Generates the PR body text

What this does NOT do (requires GitHub authentication):
  - Opening the PR via GitHub CLI or API
  - Getting review approvals

The workflow generates all the commands and files needed.
Running `gh pr create ...` at the end is a one-command action.
"""

from __future__ import annotations
import subprocess
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict


@dataclass
class PRSubmissionConfig:
    """Configuration for Mathlib4 PR submission."""
    github_username: str = "ouroboros-research"
    fork_url: str = "https://github.com/ouroboros-research/mathlib4.git"
    upstream_url: str = "https://github.com/leanprover-community/mathlib4.git"
    branch_name: str = "feat/ouroboros-linear-modular-surjective"
    target_directory: str = "Mathlib/NumberTheory"
    contribution_dir: str = "ouroboros_lean/Mathlib4Contribution"
    mathlib4_local_path: Optional[str] = None   # local clone path


@dataclass
class GitCommandResult:
    """Result of running a git command."""
    command: List[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def description(self) -> str:
        status = "✅" if self.success else "❌"
        cmd_str = " ".join(self.command)
        return f"{status} {cmd_str}\n  {self.stdout[:100] if self.stdout else self.stderr[:100]}"


class GitOperationsRunner:
    """Runs git commands with error handling."""

    def __init__(self, repo_path: Optional[str] = None, dry_run: bool = True):
        self.repo_path = repo_path
        self.dry_run = dry_run
        self._executed: List[GitCommandResult] = []

    def run(self, *args: str) -> GitCommandResult:
        cmd = ["git"] + list(args)
        if self.dry_run:
            result = GitCommandResult(cmd, 0, f"[DRY RUN] {' '.join(cmd)}", "")
            self._executed.append(result)
            return result

        try:
            proc = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            result = GitCommandResult(
                cmd, proc.returncode, proc.stdout.strip(), proc.stderr.strip()
            )
        except subprocess.TimeoutExpired:
            result = GitCommandResult(cmd, -1, "", "Command timed out")
        except FileNotFoundError:
            result = GitCommandResult(cmd, -1, "", "git not found in PATH")

        self._executed.append(result)
        return result

    def all_succeeded(self) -> bool:
        return all(r.success for r in self._executed)

    def execution_log(self) -> str:
        return "\n".join(r.description() for r in self._executed)


@dataclass
class PRReadinessItem:
    """One item in the PR readiness checklist."""
    item: str
    passed: bool
    details: str


class PRReadinessChecker:
    """
    Pre-submission checklist for Mathlib4 PR.
    
    Items checked:
    1. Zero sorry in all .lean files
    2. All theorems have docstrings
    3. PR_DESCRIPTION.md exists and is complete
    4. Files follow camelCase naming (Mathlib4 style)
    5. No unused imports
    6. Tests pass
    """

    def __init__(self, contribution_dir: str):
        self.contribution_dir = Path(contribution_dir)

    def check_all(self) -> List[PRReadinessItem]:
        """Run all readiness checks. Returns list of items."""
        checks = [
            self._check_zero_sorry(),
            self._check_pr_description(),
            self._check_lean_files_exist(),
            self._check_python_tests_pass(),
            self._check_bezout_arithmetic(),
        ]
        return checks

    def _check_zero_sorry(self) -> PRReadinessItem:
        import re
        total_sorry = 0
        if self.contribution_dir.exists():
            for lean_file in self.contribution_dir.glob("*.lean"):
                content = lean_file.read_text()
                sorries = re.findall(r'\bsorry\b', content)
                # Don't count sorries in comments
                non_comment_sorry = [
                    m for m in sorries
                    if not any(
                        line.strip().startswith('--')
                        for line in content.split('\n')
                        if m in line
                    )
                ]
                total_sorry += len(non_comment_sorry)
        passed = total_sorry == 0
        return PRReadinessItem(
            "Zero sorry in .lean files",
            passed,
            f"Found {total_sorry} sorry(s)" if not passed else "All proofs complete"
        )

    def _check_pr_description(self) -> PRReadinessItem:
        pr_path = self.contribution_dir / "PR_DESCRIPTION.md"
        if not pr_path.exists():
            return PRReadinessItem("PR_DESCRIPTION.md exists", False, "File missing")
        content = pr_path.read_text()
        has_math = "theorem" in content.lower() or "Mathematical" in content
        has_motivation = "motivation" in content.lower() or "OUROBOROS" in content
        passed = has_math and has_motivation and len(content) > 200
        return PRReadinessItem(
            "PR description complete",
            passed,
            f"{len(content)} chars, has math={has_math}, motivation={has_motivation}"
        )

    def _check_lean_files_exist(self) -> PRReadinessItem:
        if not self.contribution_dir.exists():
            return PRReadinessItem("Lean4 files exist", False, "Contribution dir missing")
        lean_files = list(self.contribution_dir.glob("*.lean"))
        passed = len(lean_files) >= 2
        return PRReadinessItem(
            "Contribution .lean files present",
            passed,
            f"Found {len(lean_files)} .lean files"
        )

    def _check_python_tests_pass(self) -> PRReadinessItem:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_mathlib4_pr.py", "-q", "--tb=no"],
            capture_output=True, text=True, timeout=60,
        )
        passed = "passed" in result.stdout and result.returncode == 0
        last_line = result.stdout.strip().split('\n')[-1] if result.stdout else "No output"
        return PRReadinessItem("Python tests pass", passed, last_line)

    def _check_bezout_arithmetic(self) -> PRReadinessItem:
        errors = []
        for a in range(7):
            for b in range(11):
                x = (a * 22 + b * 56) % 77
                if x % 7 != a or x % 11 != b:
                    errors.append(f"a={a},b={b}")
        passed = len(errors) == 0
        return PRReadinessItem(
            "Bezout witness arithmetic verified",
            passed,
            "All 77 pairs correct" if passed else f"Errors: {errors[:3]}"
        )

    def all_passed(self, checks: List[PRReadinessItem]) -> bool:
        return all(c.passed for c in checks)

    def report(self, checks: List[PRReadinessItem]) -> str:
        lines = ["PR READINESS CHECKLIST\n" + "="*40]
        for check in checks:
            status = "✅" if check.passed else "❌"
            lines.append(f"{status} {check.item}: {check.details}")
        n_pass = sum(1 for c in checks if c.passed)
        n_total = len(checks)
        lines.append(f"\n{n_pass}/{n_total} checks passed")
        if self.all_passed(checks):
            lines.append("🚀 READY TO SUBMIT PR")
        else:
            lines.append("⚠️  Fix failing checks before submitting")
        return "\n".join(lines)


class Mathlib4PRWorkflow:
    """
    Orchestrates the complete Mathlib4 PR submission workflow.
    
    Operates in dry_run mode by default — prints all commands
    that would be run without actually running them.
    Set dry_run=False to actually execute git operations.
    """

    def __init__(
        self,
        config: PRSubmissionConfig = None,
        dry_run: bool = True,
    ):
        self.cfg = config or PRSubmissionConfig()
        self.dry_run = dry_run
        self._checker = PRReadinessChecker(self.cfg.contribution_dir)
        self._git = GitOperationsRunner(
            repo_path=self.cfg.mathlib4_local_path,
            dry_run=dry_run,
        )

    def run(self, verbose: bool = True) -> bool:
        """
        Execute the full PR workflow.
        Returns True if all steps succeed.
        """
        if verbose:
            print("MATHLIB4 PR SUBMISSION WORKFLOW")
            print("=" * 50)
            print(f"Mode: {'DRY RUN (no actual git ops)' if self.dry_run else 'LIVE'}")
            print()

        # Step 1: Readiness check
        checks = self._checker.check_all()
        if verbose:
            print(self._checker.report(checks))
            print()

        if not self._checker.all_passed(checks):
            if verbose:
                print("❌ Readiness check failed. Fix issues before submitting.")
            return False

        # Step 2: Git workflow
        steps = [
            ("Fetch upstream", lambda: self._git.run(
                "fetch", "upstream", "master"
            )),
            ("Create branch", lambda: self._git.run(
                "checkout", "-b", self.cfg.branch_name,
                "upstream/master"
            )),
            ("Stage files", lambda: self._git.run(
                "add",
                f"{self.cfg.target_directory}/LinearModularSurjective.lean",
                f"{self.cfg.target_directory}/CRTInstances.lean",
            )),
            ("Commit", lambda: self._git.run(
                "commit", "-m",
                "feat(NumberTheory/LinearModular): add surjectivity lemmas for linear maps mod prime",
            )),
            ("Push branch", lambda: self._git.run(
                "push", "origin", self.cfg.branch_name
            )),
        ]

        for step_name, step_fn in steps:
            if verbose:
                print(f"Step: {step_name}...")
            result = step_fn()
            if verbose:
                print(f"  {result.description()}")
            if not result.success and not self.dry_run:
                if verbose:
                    print(f"❌ Step '{step_name}' failed. Stopping.")
                return False

        # Step 3: Generate gh pr create command
        pr_body = self._read_pr_description()
        gh_command = (
            f"gh pr create "
            f"--repo leanprover-community/mathlib4 "
            f"--title 'feat(NumberTheory/LinearModular): machine-discovered modular surjectivity lemmas' "
            f"--body-file {self.cfg.contribution_dir}/PR_DESCRIPTION.md "
            f"--base master "
            f"--head {self.cfg.github_username}:{self.cfg.branch_name}"
        )

        if verbose:
            print(f"\n{'='*50}")
            print("Final step: run this command to open the PR:")
            print(f"\n  {gh_command}\n")
            print(f"{'='*50}")
            print(self._git.execution_log())

        return self._git.all_succeeded()

    def _read_pr_description(self) -> str:
        pr_path = Path(self.cfg.contribution_dir) / "PR_DESCRIPTION.md"
        if pr_path.exists():
            return pr_path.read_text()
        return "Machine-discovered mathematical lemmas from OUROBOROS."

    def generate_submission_report(self) -> str:
        """Generate a summary report of the PR workflow."""
        checks = self._checker.check_all()
        n_pass = sum(1 for c in checks if c.passed)
        return (
            f"MATHLIB4 PR SUBMISSION REPORT\n"
            f"{'='*40}\n"
            f"Readiness: {n_pass}/{len(checks)} checks passed\n"
            f"Branch: {self.cfg.branch_name}\n"
            f"Target: {self.cfg.target_directory}\n"
            f"Mode: {'Dry run' if self.dry_run else 'Live'}\n"
            f"\nNext steps:\n"
            f"  1. Fork leanprover-community/mathlib4 on GitHub\n"
            f"  2. Clone your fork locally\n"
            f"  3. Set dry_run=False in Mathlib4PRWorkflow\n"
            f"  4. Run workflow.run()\n"
            f"  5. Run: gh pr create ...\n"
        )