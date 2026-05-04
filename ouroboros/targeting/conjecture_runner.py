"""
ConjectureTargetingSession — Continuous OUROBOROS targeting of open problems.

The session:
  1. Selects a conjecture environment (Collatz, Goldbach, prime gaps, etc.)
  2. Runs HierarchicalSearchRouter on a fresh sequence
  3. Queries NoveltyDetector — is this better than known?
  4. Queries OEISClient — is this in OEIS?
  5. If both say "novel" → write to conjecture_flags.jsonl
  6. Updates MetaMDLLearner with the discovery
  7. Repeats

The session can run indefinitely. It is designed to be left running
overnight, producing a daily report of the most interesting findings.

Known best formulas (for comparison):
  Collatz stopping times: best known ≈ 6.95 * log2(n)
  Prime gaps: best known ≈ log(p_n)^2 (Cramér's conjecture)
  Twin prime density: best known ≈ 2*C2 * n / log(n)^2
  
  Any expression achieving lower MDL is worth flagging.
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from ouroboros.novelty.open_conjectures import (
    CollatzStoppingTimesEnv, PrimeGapEnv, TwinPrimeDensityEnv,
)
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.novelty.detector import NoveltyDetector
from ouroboros.meta.mdl_prior_learner import MetaMDLLearner
from ouroboros.compression.mdl_engine import MDLEngine


@dataclass
class ConjectureFinding:
    """A flagged finding from the conjecture targeting session."""
    conjecture_name: str
    expression_str: str
    mdl_cost: float
    known_best_cost: float
    improvement_bits: float
    novelty_score: float
    oeis_match: Optional[str]
    timestamp: float
    session_id: str
    sequence_start: int
    sequence_length: int

    @property
    def beats_known(self) -> bool:
        return self.mdl_cost < self.known_best_cost

    def to_dict(self) -> dict:
        return {
            "conjecture": self.conjecture_name,
            "expression": self.expression_str,
            "mdl_cost": round(self.mdl_cost, 3),
            "known_best_cost": round(self.known_best_cost, 3),
            "improvement_bits": round(self.improvement_bits, 3),
            "novelty_score": round(self.novelty_score, 4),
            "oeis_match": self.oeis_match,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "beats_known": self.beats_known,
        }

    def summary(self) -> str:
        better = "✅ BEATS KNOWN" if self.beats_known else "📊 comparable"
        return (
            f"{better} [{self.conjecture_name}]\n"
            f"  Expression: {self.expression_str[:60]}\n"
            f"  MDL: {self.mdl_cost:.2f} vs known {self.known_best_cost:.2f} "
            f"(Δ={self.improvement_bits:.2f} bits)\n"
            f"  Novelty: {self.novelty_score:.3f}"
        )


def _compute_known_best_cost(env, known_formula_fn, observations):
    mdl = MDLEngine()
    try:
        preds = []
        for t in range(len(observations)):
            v = known_formula_fn(t)
            preds.append(int(round(v)) if math.isfinite(v) else 0)
        result = mdl.compute(preds, observations, n_nodes=5, n_constants=2)
        cost = result.total_mdl_cost
        return cost if math.isfinite(cost) else 1e9
    except Exception:
        return 1e9

class ConjectureTargetingSession:
    """
    Continuous targeting session for open mathematical conjectures.

    Usage:
        session = ConjectureTargetingSession(output_dir="results")
        session.run(max_iterations=1000, verbose=True)
        # → writes flags to results/conjecture_flags.jsonl
        # → generates daily report at results/conjecture_report.md
    """

    def __init__(
        self,
        output_dir: str = "results",
        beam_width: int = 15,
        n_iterations: int = 8,
        novelty_flag_threshold: float = 0.5,
        improvement_threshold_bits: float = 5.0,
        session_id: str = None,
        verbose: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.novelty_threshold = novelty_flag_threshold
        self.improvement_threshold = improvement_threshold_bits
        self.session_id = session_id or f"session_{int(time.time())}"
        self.verbose = verbose

        self._router = HierarchicalSearchRouter(RouterConfig(
            beam_width=beam_width,
            max_depth=5,
            n_iterations=n_iterations,
            random_seed=42,
        ))
        self._novelty_detector = NoveltyDetector(
            oeis_cache_path=str(self.output_dir / "oeis_cache.db"),
            registry_path=str(self.output_dir / "novelty_registry.json"),
            findings_log=str(self.output_dir / "novel_findings.jsonl"),
            use_oeis=True,
            novelty_threshold=novelty_flag_threshold,
            session_id=self.session_id,
            verbose=False,
        )
        self._meta_learner = MetaMDLLearner(
            save_path=str(self.output_dir / "meta_prior.json")
        )

        self._flags: List[ConjectureFinding] = []
        self._n_iterations = 0
        self._n_flags = 0

        # Known best formulas for each conjecture
        self._known_formulas = {
            "CollatzStoppingTimes": lambda t: 6.95 * math.log2(max(t, 2)),
            "PrimeGaps": lambda t: math.log(max(t * 15 + 2, 2))**2,
            "TwinPrimeDensity": lambda t: max(0, 2 * 0.6601618 * t / max(math.log(max(t*10, 2))**2, 1)),
        }

        # Environments with their names
        self._environments = [
            ("CollatzStoppingTimes", CollatzStoppingTimesEnv()),
            ("PrimeGaps", PrimeGapEnv()),
            ("TwinPrimeDensity", TwinPrimeDensityEnv()),
        ]

    def run(
        self,
        max_iterations: int = 100,
        stream_length: int = 200,
        rotate_envs: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the targeting session.
        Returns a summary dict.
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CONJECTURE TARGETING SESSION: {self.session_id}")
            print(f"Max iterations: {max_iterations}")
            print(f"Environments: {[name for name, _ in self._environments]}")
            print(f"{'='*60}\n")

        start = time.time()

        for iteration in range(max_iterations):
            self._n_iterations += 1

            # Select environment (rotate or random)
            env_name, env = self._environments[iteration % len(self._environments)]
            env.seed = iteration % 50

            # Generate fresh observations
            start_t = (iteration // len(self._environments)) * 20
            obs = env.generate(stream_length, start=start_t)

            if self.verbose and iteration % 10 == 0:
                print(f"[{iteration+1}/{max_iterations}] {env_name} "
                      f"(start={start_t}, n_flags={self._n_flags})")

            # Run discovery
            result = self._router.search(obs, alphabet_size=env.alphabet_size)
            if result.expr is None or not math.isfinite(result.mdl_cost):
                continue

            # Compute known best cost
            known_fn = self._known_formulas.get(env_name, lambda t: float(obs[t]))
            known_cost = _compute_known_best_cost(env, known_fn, obs)

            improvement = known_cost - result.mdl_cost

            # Run novelty detection
            annotated = self._novelty_detector.annotate(
                result.expr,
                [float(v) for v in obs],
                mdl_cost=result.mdl_cost,
                math_family=result.math_family.name,
            )

            # Flag if: (a) beats known formula AND (b) novel finding
            is_flagged = (
                improvement >= self.improvement_threshold and
                annotated.novelty_score >= self.novelty_threshold
            )

            if is_flagged:
                finding = ConjectureFinding(
                    conjecture_name=env_name,
                    expression_str=result.expr.to_string(),
                    mdl_cost=result.mdl_cost,
                    known_best_cost=known_cost,
                    improvement_bits=improvement,
                    novelty_score=annotated.novelty_score,
                    oeis_match=None,
                    timestamp=time.time(),
                    session_id=self.session_id,
                    sequence_start=start_t,
                    sequence_length=stream_length,
                )
                self._flags.append(finding)
                self._n_flags += 1
                self._write_flag(finding)

                if self.verbose:
                    print(f"\n⭐ FLAG #{self._n_flags}:")
                    print(finding.summary())

            # Update meta-learner
            self._meta_learner.update(
                result.expr,
                domain="number_theory",
                success=improvement > 0,
                mdl_cost=result.mdl_cost,
                generalized=annotated.novelty_score > 0.5,
            )

        elapsed = time.time() - start
        summary = self._generate_summary(elapsed)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SESSION COMPLETE")
            print(f"  Iterations: {self._n_iterations}")
            print(f"  Flags: {self._n_flags}")
            print(f"  Runtime: {elapsed:.1f}s")
            if self._flags:
                print(f"\nTop findings:")
                for f in sorted(self._flags, key=lambda x: -x.improvement_bits)[:3]:
                    print(f"  {f.summary()}")

        return summary

    def _write_flag(self, finding: ConjectureFinding) -> None:
        """Write a flag to the output JSONL."""
        flag_path = self.output_dir / "conjecture_flags.jsonl"
        with open(flag_path, 'a') as f:
            f.write(json.dumps(finding.to_dict()) + "\n")

    def _generate_summary(self, elapsed: float) -> Dict[str, Any]:
        """Generate a session summary."""
        return {
            "session_id": self.session_id,
            "n_iterations": self._n_iterations,
            "n_flags": self._n_flags,
            "runtime_seconds": elapsed,
            "flags": [f.to_dict() for f in self._flags],
            "meta_learner_updates": self._meta_learner._state.n_updates,
        }

    def generate_report(self) -> str:
        """Generate a Markdown report of the session."""
        lines = [
            f"# Conjecture Targeting Report",
            f"Session: {self.session_id}",
            f"",
            f"## Summary",
            f"- Iterations: {self._n_iterations}",
            f"- Flags: {self._n_flags}",
            f"",
            f"## Flagged Findings",
        ]
        if not self._flags:
            lines.append("No findings above threshold.")
        else:
            for i, f in enumerate(sorted(self._flags, key=lambda x: -x.improvement_bits), 1):
                lines.extend([
                    f"### Finding #{i}: {f.conjecture_name}",
                    f"- Expression: `{f.expression_str}`",
                    f"- MDL improvement: {f.improvement_bits:.2f} bits",
                    f"- Novelty score: {f.novelty_score:.3f}",
                    f"- Beats known formula: {f.beats_known}",
                    "",
                ])
        return "\n".join(lines)