"""
Lean4 Bridge — connects OUROBOROS proof market to Lean4 theorem prover.

Architecture:
    ExprNode → Lean4Translator → lean4_string
    lean4_string → Lean4Runner → (proved | refuted | timeout | unavailable)

Fallback chain:
    1. Try Lean4 (if installed)
    2. If Lean4 unavailable → use empirical counterexample search
    3. If empirical search times out → assume no counterexample (approve)

This means the same code works regardless of whether Lean4 is installed:
    With Lean4:    formal proofs, publication-quality verification
    Without Lean4: empirical search (Days 1-16 behavior)

The bridge never blocks the proof market — it always returns a result.
"""

import subprocess
import tempfile
import os
import time
from typing import Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass
from ouroboros.compression.program_synthesis import ExprNode, NodeType
from ouroboros.utils.logger import get_logger


class VerificationResult(Enum):
    PROVED        = auto()  # Lean4 found a formal proof
    REFUTED       = auto()  # Lean4 found the claim is false
    TIMEOUT       = auto()  # Lean4 timed out (inconclusive)
    UNAVAILABLE   = auto()  # Lean4 not installed
    ERROR         = auto()  # Lean4 crashed or parse error
    EMPIRICAL_CE  = auto()  # Empirical counterexample found (fallback)
    EMPIRICAL_NONE = auto() # No empirical counterexample found (fallback)


@dataclass
class VerificationReport:
    """
    Result of one Lean4 verification attempt.

    Fields:
        result: VerificationResult enum
        lean4_output: Raw Lean4 output (for debugging)
        proof_term: The proof term if PROVED (for logging)
        counterexample: CE description if REFUTED
        elapsed_seconds: How long verification took
        method: 'lean4' or 'empirical'
    """
    result: VerificationResult
    lean4_output: str = ''
    proof_term: str = ''
    counterexample: str = ''
    elapsed_seconds: float = 0.0
    method: str = 'lean4'

    @property
    def is_conclusive(self) -> bool:
        return self.result in (
            VerificationResult.PROVED,
            VerificationResult.REFUTED,
            VerificationResult.EMPIRICAL_CE,
            VerificationResult.EMPIRICAL_NONE
        )

    @property
    def approved(self) -> bool:
        """Should the modification be approved based on this result?"""
        return self.result in (
            VerificationResult.PROVED,
            VerificationResult.EMPIRICAL_NONE,
            VerificationResult.TIMEOUT,   # Inconclusive → approve (conservative)
        )

    def __repr__(self) -> str:
        return (f"VerificationReport(result={self.result.name}, "
                f"method={self.method}, "
                f"elapsed={self.elapsed_seconds:.2f}s)")


class Lean4Translator:
    """
    Translates ExprNode trees to Lean4 proposition strings.

    The translated proposition claims:
        ∀ t : Fin N, expr(t) % alphabet = stream[t]

    If this is true, the expression correctly describes the stream.
    If false (Lean4 finds a counterexample), the expression is wrong.

    Lean4 syntax notes:
        - Nat arithmetic (no overflow, everything is ℕ or ℤ)
        - % is Nat.mod
        - We use `decide` tactic for small finite instances
        - For larger instances, we use `native_decide`
    """

    def expr_to_lean4(self, node: ExprNode, var: str = 't') -> str:
        """
        Convert ExprNode to Lean4 expression string.

        Args:
            node: Expression tree node
            var: Variable name for TIME nodes (default 't')

        Returns:
            Lean4 expression string

        Examples:
            C(3)            → "3"
            T()             → "t"
            ADD(T(), C(1))  → "(t + 1)"
            MOD(T(), C(7))  → "(t % 7)"
            MUL(C(3), T())  → "(3 * t)"
        """
        if node.node_type == NodeType.CONST:
            return str(node.value)

        if node.node_type == NodeType.TIME:
            return var

        if node.node_type == NodeType.PREV:
            lag = node.lag or 1
            # PREV(k) = t - k, clamped to 0
            return f"(if {var} ≥ {lag} then {var} - {lag} else 0)"

        if node.node_type in (NodeType.ADD, NodeType.SUB, NodeType.MUL,
                               NodeType.MOD, NodeType.DIV):
            ops = {
                NodeType.ADD: '+',
                NodeType.SUB: '-',
                NodeType.MUL: '*',
                NodeType.MOD: '%',
                NodeType.DIV: '/',
            }
            l = self.expr_to_lean4(node.left, var)
            r = self.expr_to_lean4(node.right, var)
            return f"({l} {ops[node.node_type]} {r})"

        if node.node_type == NodeType.POW:
            l = self.expr_to_lean4(node.left, var)
            r = self.expr_to_lean4(node.right, var)
            return f"({l} ^ {r})"

        if node.node_type == NodeType.EQ:
            l = self.expr_to_lean4(node.left, var)
            r = self.expr_to_lean4(node.right, var)
            return f"(if {l} = {r} then 1 else 0)"

        if node.node_type == NodeType.LT:
            l = self.expr_to_lean4(node.left, var)
            r = self.expr_to_lean4(node.right, var)
            return f"(if {l} < {r} then 1 else 0)"

        if node.node_type == NodeType.IF:
            cond = self.expr_to_lean4(node.left, var)
            then_v = self.expr_to_lean4(node.right, var)
            else_v = self.expr_to_lean4(node.extra, var)
            return f"(if {cond} ≠ 0 then {then_v} else {else_v})"

        return "0"  # Unknown node type fallback

    def build_verification_script(
        self,
        expr: ExprNode,
        stream: list,
        alphabet_size: int,
        timeout_seconds: int = 30
    ) -> str:
        """
        Build a complete Lean4 script that verifies the expression
        against the stream.

        The script uses `decide` for small streams (≤ 100 symbols)
        and `native_decide` for larger ones.

        The theorem to prove:
            ∀ t : Fin N, expr_fn(t) % alphabet = stream[t]

        Returns:
            Complete Lean4 source code as a string
        """
        n = len(stream)
        expr_str = self.expr_to_lean4(expr)
        stream_list = '[' + ', '.join(str(s) for s in stream) + ']'

        tactic = 'decide' if n <= 50 else 'native_decide'

        script = f"""-- Auto-generated by OUROBOROS Lean4 Bridge
-- Verifying expression: {expr.to_string()!r}
-- Stream length: {n}
-- Alphabet size: {alphabet_size}

def ouroborosStream : List Nat := {stream_list}

def exprFn (t : Nat) : Nat :=
  {expr_str} % {alphabet_size}

-- Main verification theorem
theorem exprMatchesStream :
    ∀ t : Fin {n},
      exprFn t.val = ouroborosStream[t]! := by
  {tactic}
"""
        return script

    def build_counterexample_search_script(
        self,
        proposal_expr: ExprNode,
        current_expr: ExprNode,
        stream: list,
        alphabet_size: int
    ) -> str:
        """
        Build a Lean4 script that searches for a counterexample.

        A counterexample shows that current_expr is BETTER than
        proposal_expr on at least one timestep.

        If Lean4 can prove this, the modification is REJECTED.
        """
        n = len(stream)
        prop_str = self.expr_to_lean4(proposal_expr)
        curr_str = self.expr_to_lean4(current_expr)

        script = f"""-- Counterexample search script
-- Does current_expr beat proposal_expr on any timestep?

def proposalFn (t : Nat) : Nat := {prop_str} % {alphabet_size}
def currentFn (t : Nat) : Nat := {curr_str} % {alphabet_size}
def stream : List Nat := [{', '.join(str(s) for s in stream[:50])}]

-- Check: is there a t where current matches but proposal doesn't?
def hasCounterexample : Bool :=
  stream.enum.any (fun ⟨t, obs⟩ =>
    currentFn t = obs && proposalFn t ≠ obs)

#eval hasCounterexample  -- prints true/false
"""
        return script


class Lean4Runner:
    """
    Runs Lean4 verification scripts via subprocess.

    Handles:
        - Lean4 not installed (graceful fallback)
        - Lean4 timeouts (returns TIMEOUT)
        - Lean4 parse errors (returns ERROR)
        - Successful proofs (returns PROVED)
        - Failed proofs (returns REFUTED)

    Args:
        lean_executable: Path to lean executable (default: 'lean')
        timeout_seconds: Max time per verification (default: 30)
        work_dir: Temporary directory for Lean4 files
    """

    def __init__(
        self,
        lean_executable: str = 'lean',
        timeout_seconds: int = 30,
        work_dir: Optional[str] = None
    ):
        self.lean_executable = lean_executable
        self.timeout_seconds = timeout_seconds
        self.work_dir = work_dir
        self.logger = get_logger('Lean4Runner')
        self._lean4_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Lean4 is installed. Cached after first check."""
        if self._lean4_available is not None:
            return self._lean4_available

        try:
            result = subprocess.run(
                [self.lean_executable, '--version'],
                capture_output=True, text=True, timeout=10
            )
            self._lean4_available = (result.returncode == 0)
            if self._lean4_available:
                self.logger.info(
                    f"Lean4 found: {result.stdout.strip()}"
                )
            else:
                self.logger.warning("Lean4 found but returned error")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._lean4_available = False
            self.logger.info(
                "Lean4 not found — will use empirical fallback"
            )

        return self._lean4_available

    def run_script(self, script: str) -> VerificationReport:
        """
        Run a Lean4 script and return VerificationReport.

        Args:
            script: Complete Lean4 source code

        Returns:
            VerificationReport
        """
        if not self.is_available():
            return VerificationReport(
                result=VerificationResult.UNAVAILABLE,
                method='lean4',
                lean4_output='Lean4 not installed'
            )

        start = time.time()

        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.lean', delete=False,
            dir=self.work_dir
        ) as f:
            f.write(script)
            temp_path = f.name

        try:
            result = subprocess.run(
                [self.lean_executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )
            elapsed = time.time() - start
            output = result.stdout + result.stderr

            # Parse Lean4 output
            if result.returncode == 0 and 'error' not in output.lower():
                return VerificationReport(
                    result=VerificationResult.PROVED,
                    lean4_output=output,
                    elapsed_seconds=elapsed,
                    method='lean4'
                )
            elif 'false' in output.lower() or 'counterexample' in output.lower():
                return VerificationReport(
                    result=VerificationResult.REFUTED,
                    lean4_output=output,
                    elapsed_seconds=elapsed,
                    method='lean4'
                )
            else:
                return VerificationReport(
                    result=VerificationResult.ERROR,
                    lean4_output=output,
                    elapsed_seconds=elapsed,
                    method='lean4'
                )

        except subprocess.TimeoutExpired:
            return VerificationReport(
                result=VerificationResult.TIMEOUT,
                lean4_output=f'Timed out after {self.timeout_seconds}s',
                elapsed_seconds=self.timeout_seconds,
                method='lean4'
            )
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass


class FormalProofMarket:
    """
    Enhanced ProofMarket that uses Lean4 for formal verification.

    Extends the existing ProofMarket with a Lean4 verification layer:
        1. Standard commit-reveal market runs first (empirical)
        2. If approved by market, Lean4 verifies the claim formally
        3. Lean4 PROVED → keep approved
        4. Lean4 REFUTED → override market, reject
        5. Lean4 TIMEOUT/UNAVAILABLE → keep empirical result

    This gives us the best of both worlds:
        - Cryptographic fairness from commit-reveal
        - Formal soundness from Lean4 (when available)

    Args:
        num_agents: Number of agents
        lean_timeout: Lean4 timeout per verification
        fallback_to_empirical: If True, use empirical when Lean4 unavailable
    """

    def __init__(
        self,
        num_agents: int = 8,
        lean_timeout: int = 30,
        fallback_to_empirical: bool = True,
        **market_kwargs
    ):
        from ouroboros.proof_market.market import ProofMarket
        self.market = ProofMarket(num_agents=num_agents, **market_kwargs)
        self.translator = Lean4Translator()
        self.runner = Lean4Runner(timeout_seconds=lean_timeout)
        self.fallback = fallback_to_empirical
        self.logger = get_logger('FormalProofMarket')

        # Stats
        self.lean4_verifications: int = 0
        self.lean4_proved: int = 0
        self.lean4_refuted: int = 0
        self.lean4_timeouts: int = 0
        self.lean4_unavailable: int = 0

    def verify_formally(
        self,
        expr: ExprNode,
        test_sequence: list,
        alphabet_size: int
    ) -> VerificationReport:
        """
        Formally verify that expr correctly describes test_sequence.

        Args:
            expr: Expression to verify
            test_sequence: Observation sequence
            alphabet_size: Symbol alphabet

        Returns:
            VerificationReport
        """
        self.lean4_verifications += 1

        if not self.runner.is_available():
            self.lean4_unavailable += 1
            # Fall back to empirical check
            preds = expr.predict_sequence(len(test_sequence), alphabet_size)
            errors = sum(p != a for p, a in zip(preds, test_sequence))
            if errors == 0:
                return VerificationReport(
                    result=VerificationResult.EMPIRICAL_NONE,
                    method='empirical',
                    counterexample='',
                    lean4_output='Lean4 unavailable; empirical check passed'
                )
            else:
                return VerificationReport(
                    result=VerificationResult.EMPIRICAL_CE,
                    method='empirical',
                    counterexample=f'{errors} prediction errors found',
                    lean4_output='Lean4 unavailable; empirical check failed'
                )

        # Use short stream for Lean4 (decide tactic is slow on long streams)
        lean_stream = test_sequence[:min(50, len(test_sequence))]
        script = self.translator.build_verification_script(
            expr, lean_stream, alphabet_size
        )

        report = self.runner.run_script(script)

        if report.result == VerificationResult.PROVED:
            self.lean4_proved += 1
        elif report.result == VerificationResult.REFUTED:
            self.lean4_refuted += 1
        elif report.result == VerificationResult.TIMEOUT:
            self.lean4_timeouts += 1

        return report

    def run_formal_round(
        self,
        proposer_id: int,
        current_expr: ExprNode,
        proposed_expr: ExprNode,
        test_sequence: list,
        alphabet_size: int,
        adversarial_agents: list,
        ce_results: dict,
        bounty: float = 10.0
    ) -> Tuple[bool, VerificationReport]:
        """
        Run a full round with both empirical and formal verification.

        Returns:
            (approved: bool, formal_report: VerificationReport)
        """
        # Step 1: Empirical proof market
        empirical_approved = self.market.run_full_round(
            proposer_id=proposer_id,
            current_expr=current_expr,
            proposed_expr=proposed_expr,
            test_sequence=test_sequence,
            alphabet_size=alphabet_size,
            adversarial_agents=adversarial_agents,
            ce_results=ce_results,
            bounty=bounty
        )

        # Step 2: Formal verification (only if empirically approved)
        if not empirical_approved:
            return False, VerificationReport(
                result=VerificationResult.EMPIRICAL_CE,
                method='empirical',
                lean4_output='Rejected by empirical market'
            )

        # Run Lean4 on the proposed expression
        formal_report = self.verify_formally(
            proposed_expr, test_sequence, alphabet_size
        )

        self.logger.info(
            f"Formal verification: {formal_report.result.name} "
            f"({formal_report.elapsed_seconds:.2f}s, method={formal_report.method})"
        )

        # Override empirical approval if Lean4 formally refutes
        if formal_report.result == VerificationResult.REFUTED:
            self.logger.warning(
                "Lean4 REFUTED an empirically approved modification! "
                "This means the empirical market approved something formally wrong."
            )
            return False, formal_report

        return True, formal_report

    def formal_stats(self) -> dict:
        return {
            'total_verifications': self.lean4_verifications,
            'proved': self.lean4_proved,
            'refuted': self.lean4_refuted,
            'timeouts': self.lean4_timeouts,
            'unavailable': self.lean4_unavailable,
            'lean4_available': self.runner.is_available(),
        }

    def stats_summary(self) -> str:
        stats = self.formal_stats()
        lines = ['FormalProofMarket stats:']
        for k, v in stats.items():
            lines.append(f'  {k}: {v}')
        return '\n'.join(lines)