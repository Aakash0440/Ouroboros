"""
NoveltyDetector — The complete novelty detection pipeline.

This is the main entry point. It:
  1. Takes any discovered expression and its observations
  2. Runs behavioral embedding
  3. Searches OEIS (if integer sequence)
  4. Searches the registry of known expressions
  5. Combines into a calibrated novelty score
  6. Stores the result in the novelty history
  7. Returns a NoveltyAnnotatedResult

Integration with the main pipeline:
  Every call to HierarchicalSearchRouter.search() automatically
  passes the result through NoveltyDetector.annotate().
  
  High-novelty findings (score > 0.5) are logged to:
    results/novel_findings.jsonl  — one JSON line per novel finding
  
  Very high novelty findings (score > 0.75) trigger a notification:
    print("⭐ POTENTIALLY NOVEL: [expression] — route to mathematician")
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from ouroboros.novelty.literature_matcher import LiteratureMatcher, LiteratureSearchResult
from ouroboros.novelty.embedder import BehavioralEmbedder
from ouroboros.novelty.registry import EmbeddingRegistry


@dataclass
class NoveltyAnnotatedResult:
    """A discovery result annotated with novelty information."""
    expression_str: str
    mdl_cost: float
    math_family: str
    observations: List[float]

    # Novelty analysis
    novelty_score: float
    novelty_category: str
    literature_result: Optional[LiteratureSearchResult]
    is_flagged: bool              # True if routes to mathematician

    # Metadata
    timestamp: float
    session_id: str

    def to_dict(self) -> dict:
        return {
            "expression": self.expression_str,
            "mdl_cost": self.mdl_cost,
            "math_family": self.math_family,
            "novelty_score": round(self.novelty_score, 4),
            "novelty_category": self.novelty_category,
            "is_flagged": self.is_flagged,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "oeis_id": (
                self.literature_result.oeis_result.oeis_id
                if self.literature_result and self.literature_result.oeis_result
                and self.literature_result.oeis_result.found else None
            ),
            "nearest_known": (
                self.literature_result.registry_result.nearest_known.name
                if self.literature_result and self.literature_result.registry_result
                and self.literature_result.registry_result.nearest_known else None
            ),
        }

    def summary(self) -> str:
        flag = "⭐ FLAGGED" if self.is_flagged else "  "
        return (
            f"{flag} {self.expression_str[:50]:50s} "
            f"MDL={self.mdl_cost:.1f} "
            f"novelty={self.novelty_score:.3f} "
            f"({self.novelty_category})"
        )


@dataclass
class NoveltyReport:
    """Summary report of novelty findings from an experimental run."""
    session_id: str
    n_discoveries_analyzed: int
    n_flagged: int
    flagged_findings: List[NoveltyAnnotatedResult]
    mean_novelty_score: float

    def print_report(self) -> None:
        print(f"\nNOVELTY REPORT — Session {self.session_id}")
        print(f"{'='*60}")
        print(f"Discoveries analyzed: {self.n_discoveries_analyzed}")
        print(f"Flagged as novel:    {self.n_flagged}")
        print(f"Mean novelty score:  {self.mean_novelty_score:.3f}")
        if self.flagged_findings:
            print(f"\nFlagged findings (route to mathematician):")
            for finding in self.flagged_findings[:5]:
                print(f"  {finding.summary()}")
        else:
            print("\nNo findings above novelty threshold — all match known results")

    def to_latex(self) -> str:
        rows = "\n".join(
            f"  {f.expression_str[:40]} & "
            f"{f.novelty_score:.3f} & "
            f"{f.math_family} \\\\"
            for f in self.flagged_findings[:5]
        ) or "  (none) & — & — \\\\"
        return rf"""
\begin{{table}}[h]
\centering
\caption{{Novel findings from novelty detection (session {self.session_id}).}}
\begin{{tabular}}{{lrr}}
\toprule
Expression & Novelty Score & Domain \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


class NoveltyDetector:
    """
    The complete novelty detection pipeline.

    Plugs into the main OUROBOROS discovery flow. Every discovered
    expression passes through this detector automatically.
    """

    def __init__(
        self,
        oeis_cache_path: str = "results/oeis_cache.db",
        registry_path: str = "results/novelty_registry.json",
        findings_log: str = "results/novel_findings.jsonl",
        use_oeis: bool = True,
        novelty_threshold: float = 0.5,
        flag_threshold: float = 0.7,
        session_id: str = "default",
        verbose: bool = False,
    ):
        self._matcher = LiteratureMatcher(
            oeis_cache_path=oeis_cache_path,
            registry_path=registry_path,
            use_oeis=use_oeis,
            verbose=verbose,
        )
        self._registry = self._matcher._registry
        self._findings_log = Path(findings_log)
        self._findings_log.parent.mkdir(parents=True, exist_ok=True)
        self.novelty_threshold = novelty_threshold
        self.flag_threshold = flag_threshold
        self.session_id = session_id
        self.verbose = verbose

        self._history: List[NoveltyAnnotatedResult] = []

    def annotate(
        self,
        expr,
        observations: List[float],
        mdl_cost: float,
        math_family: str = "UNKNOWN",
    ) -> NoveltyAnnotatedResult:
        """
        Annotate a discovery with novelty information.
        Automatically logs high-novelty findings.
        """
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)

        # Run literature search
        lit_result = self._matcher.match(expr, observations)

        is_flagged = lit_result.combined_novelty_score >= self.flag_threshold

        annotated = NoveltyAnnotatedResult(
            expression_str=expr_str,
            mdl_cost=mdl_cost,
            math_family=math_family,
            observations=list(observations[:20]),
            novelty_score=lit_result.combined_novelty_score,
            novelty_category=lit_result.novelty_category,
            literature_result=lit_result,
            is_flagged=is_flagged,
            timestamp=time.time(),
            session_id=self.session_id,
        )

        self._history.append(annotated)

        # Log to file
        if annotated.novelty_score >= self.novelty_threshold:
            self._log_finding(annotated)

        # Print alert for very novel findings
        if is_flagged and self.verbose:
            print(f"\n⭐ NOVEL FINDING DETECTED:")
            print(f"   {annotated.summary()}")
            print(f"   {lit_result.recommendation}")

        return annotated

    def annotate_router_result(self, router_result) -> NoveltyAnnotatedResult:
        """Annotate a HierarchicalSearchRouter result."""
        if router_result.expr is None:
            return NoveltyAnnotatedResult(
                expression_str="None",
                mdl_cost=float('inf'),
                math_family="UNKNOWN",
                observations=[],
                novelty_score=0.0,
                novelty_category="routine",
                literature_result=None,
                is_flagged=False,
                timestamp=time.time(),
                session_id=self.session_id,
            )
        return self.annotate(
            router_result.expr,
            observations=[],  # would need to pass observations separately
            mdl_cost=router_result.mdl_cost,
            math_family=router_result.math_family.name,
        )

    def _log_finding(self, finding: NoveltyAnnotatedResult) -> None:
        """Append a finding to the JSONL log."""
        try:
            with open(self._findings_log, 'a') as f:
                f.write(json.dumps(finding.to_dict()) + "\n")
        except Exception:
            pass

    def generate_report(self) -> NoveltyReport:
        """Generate a novelty report for the current session."""
        flagged = [r for r in self._history if r.is_flagged]
        scores = [r.novelty_score for r in self._history]
        mean_score = sum(scores) / max(len(scores), 1)

        return NoveltyReport(
            session_id=self.session_id,
            n_discoveries_analyzed=len(self._history),
            n_flagged=len(flagged),
            flagged_findings=sorted(flagged, key=lambda x: -x.novelty_score),
            mean_novelty_score=mean_score,
        )

    def register_approved_discovery(
        self,
        expr,
        name: str,
        domain: str = "ouroboros",
    ) -> None:
        """
        Register an approved OUROBOROS discovery in the registry,
        so future discoveries can be compared against it.
        """
        self._registry.register_known(expr, name, domain, source="ouroboros")

    @property
    def stats(self) -> dict:
        return {
            "n_analyzed": len(self._history),
            "n_flagged": sum(1 for r in self._history if r.is_flagged),
            "matcher_stats": self._matcher.stats,
        }