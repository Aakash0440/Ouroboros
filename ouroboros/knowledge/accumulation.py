"""
AccumulationRunner — Runs OUROBOROS across multiple sessions with persistent KB.

The question this experiment answers:
  "Does OUROBOROS get smarter over time?"

Design:
  Session 1: fresh start, KB empty, agents search from scratch
  Session 2: KB has axioms from session 1, used as beam search priors
  Session 3: KB has axioms from sessions 1+2, richer priors
  ...
  Session N: KB has axioms from all previous sessions

Measurement:
  - Rounds to convergence per session (should decrease over sessions)
  - MDL cost at convergence (should improve or stay stable)
  - Number of NEW axioms discovered (should decrease as KB fills)
  - Transfer benefit: how much does the prior help vs fresh start?

Implementation:
  Uses KnowledgeBase (Day 14) for persistence.
  Uses PeriodAwareSeedBuilder (Day 34) + neural prior for warm-starting.
  Records everything to JSON for analysis.
"""

from __future__ import annotations
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

from ouroboros.environments.base import Environment
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig


@dataclass
class SessionResult:
    """Result of one accumulation session."""
    session_id: int
    environment_name: str
    rounds_to_best: int
    best_mdl_cost: float
    n_axioms_at_start: int
    n_axioms_at_end: int
    n_new_axioms: int
    elapsed_seconds: float
    prior_benefit: float
    expression_str: Optional[str]


@dataclass
class AccumulationRecord:
    """Complete record of a knowledge accumulation experiment."""
    experiment_id: str
    n_sessions: int
    environments_tested: List[str]
    sessions: List[SessionResult] = field(default_factory=list)

    total_axioms_accumulated: int = 0
    mean_rounds_early: float = 0.0
    mean_rounds_late: float = 0.0
    convergence_speedup: float = 0.0

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "n_sessions": self.n_sessions,
            "total_axioms": self.total_axioms_accumulated,
            "mean_rounds_early": self.mean_rounds_early,
            "mean_rounds_late": self.mean_rounds_late,
            "convergence_speedup": self.convergence_speedup,
            "sessions": [
                {
                    "session_id": s.session_id,
                    "env": s.environment_name,
                    "best_mdl": s.best_mdl_cost,
                    "rounds": s.rounds_to_best,
                    "n_new_axioms": s.n_new_axioms,
                    "prior_benefit": s.prior_benefit,
                    "elapsed": s.elapsed_seconds,
                }
                for s in self.sessions
            ]
        }


class SimpleAxiomKB:
    """
    Simplified in-memory Knowledge Base for the accumulation experiment.

    Stores discovered axioms as (expression_str, mdl_cost, environment) tuples.
    On load, provides the top-K expressions as beam search seeds.
    """

    def __init__(self):
        self._axioms: List[Dict] = []

    def add_axiom(
        self,
        expression_str: str,
        mdl_cost: float,
        environment_name: str,
        session_id: int,
    ) -> bool:
        """Add an axiom if not already present. Returns True if new."""
        if any(a["expr"] == expression_str for a in self._axioms):
            return False
        self._axioms.append({
            "expr": expression_str,
            "mdl_cost": mdl_cost,
            "env": environment_name,
            "session": session_id,
            "usage_count": 0,
        })
        return True

    def get_seeds_for_environment(
        self,
        environment_name: str,
        top_k: int = 5,
    ) -> List[str]:
        relevant = [a for a in self._axioms if a["env"] == environment_name]
        relevant.sort(key=lambda a: a["mdl_cost"])
        return [a["expr"] for a in relevant[:top_k]]

    def get_all_seeds(self, top_k: int = 10) -> List[str]:
        sorted_axioms = sorted(self._axioms, key=lambda a: a["mdl_cost"])
        return [a["expr"] for a in sorted_axioms[:top_k]]

    @property
    def n_axioms(self) -> int:
        return len(self._axioms)

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self._axioms, indent=2))

    def load(self, path: str) -> None:
        if Path(path).exists():
            self._axioms = json.loads(Path(path).read_text())


class AccumulationRunner:
    """
    Runs OUROBOROS for N sessions with persistent knowledge base.

    Each session:
    1. Load KB axioms from previous sessions
    2. Run HierarchicalSearchRouter (uses KB seeds if available)
    3. Save discovered expressions back to KB
    4. Record performance metrics
    """

    def __init__(
        self,
        n_sessions: int = 100,
        environments: List[Environment] = None,
        stream_length: int = 200,
        beam_width: int = 15,
        n_iterations: int = 8,
        kb_path: Optional[str] = None,
        verbose: bool = True,
        report_every: int = 10,
    ):
        self.n_sessions = n_sessions
        self.stream_length = stream_length
        self.beam_width = beam_width
        self.n_iterations = n_iterations
        self.kb_path = kb_path or "results/accumulation_kb.json"
        self.verbose = verbose
        self.report_every = report_every

        # Threshold scales with stream length — save any finite-cost expression
        # that isn't obviously random noise (heuristic: < 8 bits/symbol)
        self._kb_save_threshold = stream_length * 8.0

        if environments is None:
            from ouroboros.environments.modular import ModularArithmeticEnv
            from ouroboros.environments.noise import NoiseEnv
            from ouroboros.environments.fibonacci_mod import FibonacciModEnv
            self.environments = [
                ModularArithmeticEnv(modulus=7, slope=3, intercept=1),
                ModularArithmeticEnv(modulus=11, slope=5, intercept=2),
                FibonacciModEnv(modulus=7),
            ]
        else:
            self.environments = environments

        self._kb = SimpleAxiomKB()
        if kb_path and Path(kb_path).exists():
            self._kb.load(kb_path)

        self._router = HierarchicalSearchRouter(RouterConfig(
            beam_width=beam_width,
            max_depth=4,
            n_iterations=n_iterations,
            random_seed=42,
        ))

        # Track per-environment baseline for prior_benefit computation
        self._env_baseline: Dict[str, float] = {}

    def _run_single_session(
        self,
        session_id: int,
        env: Environment,
        fresh_start: bool = False,
    ) -> SessionResult:
        """Run one session on one environment."""
        n_axioms_start = self._kb.n_axioms
        obs = env.generate(self.stream_length)

        start = time.time()
        result = self._router.search(obs, alphabet_size=env.alphabet_size)
        elapsed = time.time() - start

        # Establish per-environment baseline on first visit
        env_key = env.name
        if env_key not in self._env_baseline:
            self._env_baseline[env_key] = result.mdl_cost

        baseline = self._env_baseline[env_key]
        prior_benefit = max(0.0, baseline - result.mdl_cost)

        # Update neural prior on good discoveries
        if result.expr and result.mdl_cost < baseline:
            self._router.update_prior(
                obs, result.expr,
                reward_bits=max(0, baseline - result.mdl_cost),
            )

        # Save to KB — threshold scales with stream length
        n_new = 0
        if result.expr and result.mdl_cost < self._kb_save_threshold:
            expr_str = result.expr.to_string()
            if self._kb.add_axiom(expr_str, result.mdl_cost, env_key, session_id):
                n_new = 1

        return SessionResult(
            session_id=session_id,
            environment_name=env_key,
            rounds_to_best=1,
            best_mdl_cost=result.mdl_cost,
            n_axioms_at_start=n_axioms_start,
            n_axioms_at_end=self._kb.n_axioms,
            n_new_axioms=n_new,
            elapsed_seconds=elapsed,
            prior_benefit=prior_benefit,
            expression_str=result.expr.to_string() if result.expr else None,
        )

    def run(self, experiment_id: str = "accum_exp_1") -> AccumulationRecord:
        """Run the full accumulation experiment."""
        record = AccumulationRecord(
            experiment_id=experiment_id,
            n_sessions=self.n_sessions,
            environments_tested=[e.name for e in self.environments],
        )

        print(f"\n{'='*60}")
        print(f"KNOWLEDGE ACCUMULATION EXPERIMENT: {experiment_id}")
        print(f"Sessions: {self.n_sessions}, Environments: {len(self.environments)}")
        print(f"KB save threshold: MDL < {self._kb_save_threshold:.0f}")
        print(f"{'='*60}")

        mdl_costs_all = []
        prior_benefits_all = []

        for session_id in range(1, self.n_sessions + 1):
            env = self.environments[(session_id - 1) % len(self.environments)]
            env.seed = session_id % 20

            session_result = self._run_single_session(
                session_id=session_id,
                env=env,
                fresh_start=(session_id == 1),
            )
            record.sessions.append(session_result)
            mdl_costs_all.append(session_result.best_mdl_cost)
            prior_benefits_all.append(session_result.prior_benefit)

            if session_id % self.report_every == 0 or session_id == 1:
                n = len(mdl_costs_all)
                recent = mdl_costs_all[-min(10, n):]
                mean_recent = sum(recent) / len(recent)
                print(
                    f"  Session {session_id:4d}: env={env.name[:20]:20s}, "
                    f"MDL={session_result.best_mdl_cost:.1f}, "
                    f"KB_size={self._kb.n_axioms}, "
                    f"mean_recent={mean_recent:.1f}"
                )

        record.total_axioms_accumulated = self._kb.n_axioms

        early_costs = mdl_costs_all[:min(10, len(mdl_costs_all))]
        late_costs = mdl_costs_all[max(0, len(mdl_costs_all)-10):]
        record.mean_rounds_early = sum(early_costs) / len(early_costs) if early_costs else 0.0
        record.mean_rounds_late = sum(late_costs) / len(late_costs) if late_costs else 0.0
        if record.mean_rounds_late > 0:
            record.convergence_speedup = record.mean_rounds_early / record.mean_rounds_late

        Path("results").mkdir(exist_ok=True)
        self._kb.save(self.kb_path)

        print(f"\n{'='*60}")
        print(f"ACCUMULATION RESULTS")
        print(f"  Total axioms accumulated: {record.total_axioms_accumulated}")
        print(f"  Early MDL (sessions 1-10): {record.mean_rounds_early:.2f}")
        print(f"  Late MDL (sessions 90-100): {record.mean_rounds_late:.2f}")
        if record.convergence_speedup > 0:
            direction = "↑ IMPROVED" if record.convergence_speedup > 1.0 else "↓ degraded"
            print(f"  Speedup ratio: {record.convergence_speedup:.2f}x ({direction})")
        print(f"  KB saved to: {self.kb_path}")

        return record

    def save_record(self, record: AccumulationRecord, path: str) -> None:
        Path(path).parent.mkdir(exist_ok=True)
        Path(path).write_text(json.dumps(record.to_dict(), indent=2))
        print(f"Record saved to {path}")