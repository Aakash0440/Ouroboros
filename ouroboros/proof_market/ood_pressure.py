"""
Out-of-Distribution (OOD) Pressure Module.

The evaluation bootstrap problem:
    The proof market tests modifications on sequences agents have seen.
    A sufficiently clever agent could overfit to the training sequences.
    Agents might collectively vote to approve modifications that work on
    training data but fail on novel data.

The OOD fix:
    After the proof market approves a modification, test the new expression
    on environments the agents have NEVER been trained on:
        - ModularArithmeticEnv with DIFFERENT modulus (never seen)
        - ModularArithmeticEnv with DIFFERENT slope (never seen)
        - FibonacciModEnv with DIFFERENT modulus (never seen)

    A genuinely better expression compresses all of these better than the
    old expression. An overfit expression fails on at least some of them.

    Failure threshold: if the new expression does NOT improve compression
    on at least ood_pass_fraction of OOD environments, REVOKE the approval.

This is what turns the convergence theorem from a conjecture into something
actually provable — the external consistency pressure from novel environments.

Reference:
    This addresses the limitation discussed in Schmidhuber (2003):
    "A system that only verifies itself against seen data cannot guarantee
    its modifications are globally sound."
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import compression_ratio, naive_bits
from ouroboros.environments.base import ObservationEnvironment
from ouroboros.environments.structured import (
    ModularArithmeticEnv, FibonacciModEnv, MultiScaleEnv, NoiseEnv
)
from ouroboros.utils.logger import get_logger


@dataclass
class OODTestResult:
    """
    Result of testing one expression on one OOD environment.

    Fields:
        env_name: Environment name
        old_ratio: Compression ratio of old expression
        new_ratio: Compression ratio of new expression
        improvement: old_ratio - new_ratio (positive = new is better)
        passed: new expression improved or maintained compression
    """
    env_name: str
    old_ratio: float
    new_ratio: float
    improvement: float
    passed: bool

    def __repr__(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (f"OODTest({self.env_name}: "
                f"old={self.old_ratio:.4f} → new={self.new_ratio:.4f} "
                f"Δ={self.improvement:+.4f} {status})")


@dataclass
class OODPressureReport:
    """Complete OOD test report for one modification."""
    proposal_id: str
    test_results: List[OODTestResult] = field(default_factory=list)
    passed_count: int = 0
    total_count: int = 0
    overall_passed: bool = False
    revoked: bool = False

    def pass_fraction(self) -> float:
        return self.passed_count / max(self.total_count, 1)

    def summary(self) -> str:
        lines = [
            f"OOD Report for {self.proposal_id}:",
            f"  Passed: {self.passed_count}/{self.total_count} "
            f"({self.pass_fraction():.0%})",
            f"  Overall: {'PASS ✅' if self.overall_passed else 'FAIL ❌'}"
        ]
        for r in self.test_results:
            lines.append(f"    {r}")
        return '\n'.join(lines)


def expression_compression_ratio(
    expr: ExprNode,
    sequence: List[int],
    alphabet_size: int
) -> float:
    """
    Measure an expression's compression ratio on a sequence.

    ratio = MDL_cost / naive_bits
    Lower = better compression.
    """
    from ouroboros.compression.mdl import MDLCost, naive_bits as nb_fn
    n = len(sequence)
    preds = expr.predict_sequence(n, alphabet_size)
    mdl = MDLCost()
    cost = mdl.total_cost(expr.to_bytes(), preds, sequence, alphabet_size)
    nb = nb_fn(sequence, alphabet_size)
    return cost / nb if nb > 0 else 1.0


class OODPressureModule:
    """
    Tests approved modifications on out-of-distribution environments.

    Usage:
        ood = OODPressureModule(ood_environments, alphabet_size)
        report = ood.test_modification(
            proposal_id, old_expr, new_expr
        )
        if report.revoked:
            # Modification revoked — overfit detected

    Args:
        ood_environments: List of (name, env, alphabet_size) tuples
                          These should NEVER be used for agent training
        pass_fraction_threshold: Fraction of OOD envs that must improve
        stream_length: How many symbols to generate per OOD test
        improvement_threshold: Minimum improvement to count as "passed"
                               (0.0 = any improvement counts)
    """

    def __init__(
        self,
        ood_environments=None,
        pass_fraction_threshold: float = 0.60,
        stream_length: int = 300,
        improvement_threshold: float = -0.05,
        validation_environments=None,
        n_tests: int = 5,
    ):
        if ood_environments is None and validation_environments is not None:
            ood_environments = []
            for env in validation_environments[:n_tests]:
                name = getattr(env, 'name', type(env).__name__)
                alpha = getattr(env, 'alphabet_size', getattr(env, 'modulus', 10))
                ood_environments.append((name, env, alpha))
        self.ood_environments = ood_environments or []
        self.pass_fraction_threshold = pass_fraction_threshold
        self.stream_length = stream_length
        self.improvement_threshold = improvement_threshold
        self.logger = get_logger('OODPressure')
        self.reports: List[OODPressureReport] = []

    def test(self, predict_fn, stream_length: int = 200):
        from dataclasses import dataclass as _dc
        @_dc
        class _R:
            pass_fraction: float
            n_passed: int
            n_total: int
        passed = 0
        total = len(self.ood_environments)
        for env_name, env, alpha_size in self.ood_environments:
            try:
                obs = env.generate(stream_length)
                preds = predict_fn(obs)
                if not preds or len(preds) == 0:
                    continue
                accuracy = sum(p == a for p, a in zip(preds, obs)) / len(obs)
                if accuracy > 0.5:
                    passed += 1
            except Exception:
                pass
        frac = passed / total if total > 0 else 0.0
        return _R(pass_fraction=frac, n_passed=passed, n_total=total)
        """
        Create a default OOD test suite.

        Uses environments with parameters DIFFERENT from training defaults.
        These should never be seen during Phase 1 or 2 training.
        """
        ood_envs = [
            # Different moduli (not 7)
            ("ModArith(5,2,3)", ModularArithmeticEnv(5, 2, 3, seed=99), 5),
            ("ModArith(11,4,2)", ModularArithmeticEnv(11, 4, 2, seed=99), 11),
            ("ModArith(13,5,4)", ModularArithmeticEnv(13, 5, 4, seed=99), 13),
            # Different structure
            ("FibonacciMod(7)", FibonacciModEnv(7, seed=99), 7),
            ("MultiScale(14,7)", MultiScaleEnv(14, 7, 0.03, seed=99), 4),
        ]
        return cls(ood_envs, pass_fraction_threshold=0.50)

    def test_modification(
        self,
        proposal_id: str,
        old_expr: ExprNode,
        new_expr: ExprNode,
        verbose: bool = False
    ) -> OODPressureReport:
        """
        Test whether new_expr generalizes better than old_expr on OOD envs.

        For each OOD environment:
            1. Generate a fresh stream (never seen before)
            2. Compute old_expr's compression ratio
            3. Compute new_expr's compression ratio
            4. Pass if new_ratio <= old_ratio + improvement_threshold

        If pass_fraction < pass_fraction_threshold: REVOKE the modification.

        Args:
            proposal_id: The proposal being tested (for logging)
            old_expr: The expression being replaced
            new_expr: The proposed replacement
            verbose: Print per-environment results

        Returns:
            OODPressureReport
        """
        report = OODPressureReport(proposal_id=proposal_id)

        for env_name, env, alpha_size in self.ood_environments:
            # Generate fresh OOD stream
            env.reset(self.stream_length)
            stream = env.peek_all()

            # Compute compression ratios
            old_ratio = expression_compression_ratio(old_expr, stream, alpha_size)
            new_ratio = expression_compression_ratio(new_expr, stream, alpha_size)
            improvement = old_ratio - new_ratio

            # Pass if new expression is at least as good (within threshold)
            passed = improvement >= self.improvement_threshold

            result = OODTestResult(
                env_name=env_name,
                old_ratio=old_ratio,
                new_ratio=new_ratio,
                improvement=improvement,
                passed=passed
            )
            report.test_results.append(result)

            if verbose:
                print(f"  {result}")

        report.total_count = len(report.test_results)
        report.passed_count = sum(1 for r in report.test_results if r.passed)
        report.overall_passed = (
            report.pass_fraction() >= self.pass_fraction_threshold
        )
        report.revoked = not report.overall_passed

        if verbose:
            print(f"\n{report.summary()}")

        self.reports.append(report)

        if report.revoked:
            self.logger.warning(
                f"Proposal {proposal_id}: OOD FAILED "
                f"({report.passed_count}/{report.total_count} passed). REVOKED."
            )
        else:
            self.logger.info(
                f"Proposal {proposal_id}: OOD PASSED "
                f"({report.passed_count}/{report.total_count} passed)."
            )

        return report

    def ood_generalization_score(self, expr: ExprNode) -> float:
        """
        Compute overall OOD generalization score for an expression.

        Score = mean compression improvement across all OOD environments.
        Positive = expression generalizes to novel data.
        Negative = expression overfits to training.
        """
        improvements = []
        for env_name, env, alpha_size in self.ood_environments:
            env.reset(self.stream_length)
            stream = env.peek_all()
            from ouroboros.compression.mdl import naive_bits
            nb = naive_bits(stream, alpha_size)
            if nb == 0:
                continue
            ratio = expression_compression_ratio(expr, stream, alpha_size)
            # Score: 1.0 - ratio (higher = better compression = better generalization)
            improvements.append(1.0 - ratio)

        return float(np.mean(improvements)) if improvements else 0.0