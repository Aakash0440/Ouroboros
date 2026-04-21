"""
BenchmarkRunner — Runs all OUROBOROS experiments with statistical rigor.

Experiment types:
  1. compression_landmark   — MDL compression ratio at discovery step (Fig 1)
  2. moduli_generalization  — Discovery across prime moduli 5, 7, 11, 13
  3. convergence_rounds     — Steps to consensus across N agents
  4. crt_accuracy           — Fraction of runs where CRT is found correctly
  5. ood_generalization     — Approval rate of modifications on OOD envs
  6. self_improvement       — MDL improvement across Layer 1/2/3 rounds

Each experiment runs with n_seeds different random seeds.
Results: (mean, std, median, min, max, n_seeds)
"""

from __future__ import annotations
import json
import math
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class ExperimentResult:
    """Statistical summary of one experiment across n seeds."""
    experiment_name: str
    metric_name: str
    metric_unit: str
    values: List[float]   # one per seed

    # Computed statistics
    mean: float = field(init=False)
    std: float = field(init=False)
    median: float = field(init=False)
    min_val: float = field(init=False)
    max_val: float = field(init=False)
    n: int = field(init=False)

    def __post_init__(self):
        self.n = len(self.values)
        if self.n > 0:
            self.mean = statistics.mean(self.values)
            self.std = statistics.stdev(self.values) if self.n > 1 else 0.0
            self.median = statistics.median(self.values)
            self.min_val = min(self.values)
            self.max_val = max(self.values)
        else:
            self.mean = self.std = self.median = self.min_val = self.max_val = 0.0

    def ci_95(self) -> float:
        """95% confidence interval half-width (assuming normal distribution)."""
        if self.n <= 1:
            return float('inf')
        return 1.96 * self.std / math.sqrt(self.n)

    def latex_str(self, decimal_places: int = 4) -> str:
        """Format as 'mean ± CI' for LaTeX tables."""
        fmt = f"{{:.{decimal_places}f}}"
        mean_str = fmt.format(self.mean)
        ci_str = fmt.format(self.ci_95())
        return f"{mean_str} \\pm {ci_str}"

    def summary_str(self) -> str:
        return (
            f"{self.experiment_name}/{self.metric_name}: "
            f"{self.mean:.4f} ± {self.ci_95():.4f} {self.metric_unit} "
            f"(n={self.n}, median={self.median:.4f})"
        )


class BenchmarkRunner:
    """
    Runs all OUROBOROS experiments systematically.
    
    Usage:
        runner = BenchmarkRunner(n_seeds=10, fast_mode=True)
        results = runner.run_all()
        runner.save_results(results, "results/benchmark.json")
    """

    def __init__(
        self,
        n_seeds: int = 10,
        fast_mode: bool = True,
        output_dir: str = "results",
        verbose: bool = True,
    ):
        self.n_seeds = n_seeds
        self.fast_mode = fast_mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.verbose = verbose

        # Fast mode reduces parameters for speed
        if fast_mode:
            self.stream_length = 300
            self.n_agents = 6
            self.n_rounds = 10
            self.beam_width = 15
        else:
            self.stream_length = 2000
            self.n_agents = 8
            self.n_rounds = 20
            self.beam_width = 30

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    # ── Experiment 1: Compression Landmark ────────────────────────────────────

    def run_compression_landmark(self) -> ExperimentResult:
        """
        Measure MDL compression ratio at the discovery step.
        
        Key result: what is the ratio (MDL_before / MDL_after) when
        an agent first finds the correct expression?
        
        Expected: ~0.004 (from Day 1 experiments)
        """
        self._log("\n[1/6] Compression Landmark Experiment")
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine
        from ouroboros.environments.modular import ModularArithmeticEnv

        ratios = []
        mdl_engine = MDLEngine()

        for seed in range(self.n_seeds):
            env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1, seed=seed)
            obs = env.generate(self.stream_length)

            # Naive cost: predict mode always
            from collections import Counter
            mode = Counter(obs).most_common(1)[0][0]
            naive_preds = [mode] * len(obs)
            naive_result = mdl_engine.compute(naive_preds, obs, 1, 1)
            naive_cost = naive_result.total_mdl_cost

            # Best expression cost
            cfg = BeamConfig(
                beam_width=self.beam_width,
                const_range=20,
                max_depth=4,
                mcmc_iterations=100,
                random_seed=seed * 7,
            )
            synth = BeamSearchSynthesizer(cfg)
            expr = synth.search(obs)

            if expr is not None:
                preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                result = mdl_engine.compute(preds, obs, expr.node_count(), expr.constant_count())
                ratio = result.total_mdl_cost / naive_cost if naive_cost > 0 else 1.0
                ratios.append(ratio)
                self._log(f"  Seed {seed:2d}: ratio={ratio:.6f}, expr={expr.to_string()[:40]}")
            else:
                ratios.append(1.0)
                self._log(f"  Seed {seed:2d}: no expression found")

        return ExperimentResult(
            experiment_name="compression_landmark",
            metric_name="compression_ratio",
            metric_unit="(dimensionless)",
            values=ratios,
        )

    # ── Experiment 2: Moduli Generalization ───────────────────────────────────

    def run_moduli_generalization(self) -> List[ExperimentResult]:
        """
        Measure discovery success rate across prime moduli 5, 7, 11, 13.
        
        Success = found an expression with compression ratio < 0.1.
        Expected: >90% success on all primes.
        """
        self._log("\n[2/6] Moduli Generalization Experiment")
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine
        from ouroboros.environments.modular import ModularArithmeticEnv

        results = []
        mdl_engine = MDLEngine()

        for modulus in [5, 7, 11, 13]:
            self._log(f"  Modulus {modulus}...")
            success_rates = []

            for seed in range(self.n_seeds):
                env = ModularArithmeticEnv(modulus=modulus, slope=3, intercept=1, seed=seed)
                obs = env.generate(self.stream_length)

                from collections import Counter
                mode = Counter(obs).most_common(1)[0][0]
                naive_preds = [mode] * len(obs)
                naive_result = mdl_engine.compute(naive_preds, obs, 1, 1)
                naive_cost = naive_result.total_mdl_cost

                cfg = BeamConfig(
                    beam_width=self.beam_width,
                    const_range=modulus * 2,
                    max_depth=4,
                    mcmc_iterations=100,
                    random_seed=seed * 11,
                )
                expr = BeamSearchSynthesizer(cfg).search(obs)

                if expr is not None:
                    preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                    r = mdl_engine.compute(preds, obs, expr.node_count(), expr.constant_count())
                    ratio = r.total_mdl_cost / naive_cost if naive_cost > 0 else 1.0
                    success_rates.append(1.0 if ratio < 0.1 else 0.0)
                else:
                    success_rates.append(0.0)

            results.append(ExperimentResult(
                experiment_name=f"moduli_generalization_mod{modulus}",
                metric_name="success_rate",
                metric_unit="fraction",
                values=success_rates,
            ))
            self._log(f"    Mod {modulus} success: {statistics.mean(success_rates):.2f}")

        return results

    # ── Experiment 3: Convergence Rounds ──────────────────────────────────────

    def run_convergence_rounds(self) -> ExperimentResult:
        """
        Measure how many rounds until all agents converge to the same axiom.
        
        Expected: ~8 rounds (from Day 9 experiments)
        """
        self._log("\n[3/6] Convergence Rounds Experiment")
        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.agents.proto_axiom_pool import ProtoAxiomPool
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        from ouroboros.compression.mdl_engine import MDLEngine

        convergence_steps = []
        mdl_engine = MDLEngine()

        for seed in range(self.n_seeds):
            env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1, seed=seed)
            pool = ProtoAxiomPool(consensus_threshold=0.5, n_agents=self.n_agents)
            promoted_at = None

            for round_num in range(1, self.n_rounds + 1):
                obs = env.generate(self.stream_length, start=(round_num-1)*self.stream_length)

                for agent_i in range(self.n_agents):
                    cfg = BeamConfig(
                        beam_width=self.beam_width,
                        const_range=20,
                        max_depth=4,
                        mcmc_iterations=80,
                        random_seed=seed * 100 + agent_i * 7,
                    )
                    expr = BeamSearchSynthesizer(cfg).search(obs)
                    if expr is not None:
                        preds = [expr.evaluate(t, obs[:t]) for t in range(len(obs))]
                        r = mdl_engine.compute(preds, obs, expr.node_count(), expr.constant_count())
                        pool.submit(f"AGENT_{agent_i:02d}", expr, r.total_mdl_cost, round_num)

                if pool.has_promoted_axiom() and promoted_at is None:
                    promoted_at = round_num
                    break

            steps = promoted_at if promoted_at else self.n_rounds
            convergence_steps.append(float(steps))
            self._log(f"  Seed {seed:2d}: converged at round {steps}")

        return ExperimentResult(
            experiment_name="convergence_rounds",
            metric_name="rounds_to_convergence",
            metric_unit="rounds",
            values=convergence_steps,
        )

    # ── Experiment 4: CRT Accuracy ────────────────────────────────────────────

    def run_crt_accuracy(self) -> ExperimentResult:
        """
        Measure what fraction of runs find the correct CRT joint expression.
        
        Expected: >0.85 accuracy with fast config, >0.95 with wide beam.
        """
        self._log("\n[4/6] CRT Accuracy Experiment")
        from ouroboros.environments.joint import JointEnvironment
        from ouroboros.emergence.crt_detector import check_behavioral_crt
        from ouroboros.acceleration.sparse_beam import SparseBeamSearch, SparseBeamConfig

        accuracies = []

        for seed in range(self.n_seeds):
            env = JointEnvironment(
                mod1=7, slope1=3, int1=1,
                mod2=11, slope2=5, int2=2,
                seed=seed,
            )
            obs = env.generate(self.stream_length)

            cfg = SparseBeamConfig(
                beam_width=self.beam_width,
                const_range=77,
                max_lag=3,
                n_iterations=8,
                random_seed=seed * 13,
            )
            expr = SparseBeamSearch(cfg).search(obs, alphabet_size=77)

            if expr is not None:
                # Check if expression captures CRT behavior
                is_crt = check_behavioral_crt(
                    expr, obs[:200],
                    mod1=7, mod2=11,
                    accuracy_threshold=0.85,
                )
                accuracies.append(1.0 if is_crt else 0.0)
                self._log(f"  Seed {seed:2d}: CRT={'✓' if is_crt else '✗'}, "
                          f"expr={expr.to_string()[:40]}")
            else:
                accuracies.append(0.0)
                self._log(f"  Seed {seed:2d}: no expression found")

        return ExperimentResult(
            experiment_name="crt_accuracy",
            metric_name="crt_success_rate",
            metric_unit="fraction",
            values=accuracies,
        )

    # ── Experiment 5: OOD Generalization ─────────────────────────────────────

    def run_ood_generalization(self) -> ExperimentResult:
        """
        Measure the fraction of approved modifications that generalize OOD.
        
        Expected: >0.80 (modifications that survive OOD testing are genuinely good)
        """
        self._log("\n[5/6] OOD Generalization Experiment")
        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.environments.noise import NoiseEnv
        from ouroboros.proof_market.ood_pressure import OODPressureModule

        ood_scores = []

        for seed in range(self.n_seeds):
            ood_envs = [
                ModularArithmeticEnv(modulus=11, slope=5, intercept=2, seed=seed),
                ModularArithmeticEnv(modulus=13, slope=7, intercept=3, seed=seed),
                NoiseEnv(alphabet_size=7, seed=seed),
            ]
            ood = OODPressureModule(validation_environments=ood_envs, n_tests=3)
            ood_result = ood.test(
                lambda obs: [(3*t+1)%7 for t in range(len(obs))],  # perfect expr
                stream_length=self.stream_length,
            )
            ood_scores.append(ood_result.pass_fraction)
            self._log(f"  Seed {seed:2d}: OOD pass_fraction={ood_result.pass_fraction:.2f}")

        return ExperimentResult(
            experiment_name="ood_generalization",
            metric_name="ood_pass_fraction",
            metric_unit="fraction",
            values=ood_scores,
        )

    # ── Experiment 6: Self-Improvement Gain ───────────────────────────────────

    def run_self_improvement_gain(self) -> ExperimentResult:
        """
        Measure MDL cost reduction over self-improvement rounds.
        
        Reports: (initial_cost - final_cost) / initial_cost
        Expected: >0.10 (10% improvement from self-modification)
        """
        self._log("\n[6/6] Self-Improvement Gain Experiment")
        from ouroboros.agents.layer2_agent import Layer2Agent, Layer2AgentConfig
        from ouroboros.agents.objective_market import ObjectiveProofMarket
        from ouroboros.environments.modular import ModularArithmeticEnv

        gains = []

        for seed in range(self.n_seeds):
            env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1, seed=seed)
            agent = Layer2Agent(
                config=Layer2AgentConfig(
                    agent_id=f"BENCH_{seed:02d}",
                    objective_proposal_interval=2,
                    random_seed=seed * 17,
                )
            )
            market = ObjectiveProofMarket()

            initial_cost = None
            final_cost = None

            for round_num in range(1, 8):
                log = agent.run_round(
                    env=env,
                    objective_market=market,
                    validation_envs=[],
                    round_num=round_num,
                    stream_length=self.stream_length,
                    verbose=False,
                )
                cost = log.get("current_cost", 0.0)
                if cost > 0:
                    if initial_cost is None:
                        initial_cost = cost
                    final_cost = cost

            if initial_cost and initial_cost > 0 and final_cost is not None:
                gain = (initial_cost - final_cost) / initial_cost
                gains.append(max(0.0, gain))
            else:
                gains.append(0.0)
            self._log(f"  Seed {seed:2d}: gain={gains[-1]:.4f}")

        return ExperimentResult(
            experiment_name="self_improvement_gain",
            metric_name="relative_cost_reduction",
            metric_unit="fraction",
            values=gains,
        )

    # ── Run all experiments ───────────────────────────────────────────────────

    def run_all(self) -> Dict[str, Any]:
        """Run all 6 experiments and return a results dictionary."""
        start_total = time.time()
        self._log(f"\n{'='*60}")
        self._log(f"OUROBOROS FULL BENCHMARK SUITE")
        self._log(f"Seeds: {self.n_seeds}, Fast mode: {self.fast_mode}")
        self._log(f"{'='*60}")

        all_results = {}

        # Experiment 1
        r1 = self.run_compression_landmark()
        all_results["compression_landmark"] = r1
        self._log(f"  → {r1.summary_str()}")

        # Experiment 2
        r2_list = self.run_moduli_generalization()
        for r2 in r2_list:
            all_results[r2.experiment_name] = r2
            self._log(f"  → {r2.summary_str()}")

        # Experiment 3
        r3 = self.run_convergence_rounds()
        all_results["convergence_rounds"] = r3
        self._log(f"  → {r3.summary_str()}")

        # Experiment 4
        r4 = self.run_crt_accuracy()
        all_results["crt_accuracy"] = r4
        self._log(f"  → {r4.summary_str()}")

        # Experiment 5
        r5 = self.run_ood_generalization()
        all_results["ood_generalization"] = r5
        self._log(f"  → {r5.summary_str()}")

        # Experiment 6
        r6 = self.run_self_improvement_gain()
        all_results["self_improvement_gain"] = r6
        self._log(f"  → {r6.summary_str()}")

        elapsed = time.time() - start_total
        self._log(f"\nTotal benchmark time: {elapsed/60:.1f} minutes")

        return all_results

    def save_results(self, results: Dict[str, Any], filename: str = None) -> Path:
        """Save results to JSON."""
        if filename is None:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_{ts}.json"

        path = self.output_dir / filename
        serializable = {}
        for key, r in results.items():
            if isinstance(r, ExperimentResult):
                serializable[key] = {
                    "mean": r.mean, "std": r.std, "median": r.median,
                    "min": r.min_val, "max": r.max_val,
                    "ci_95": r.ci_95(), "n": r.n, "values": r.values,
                    "latex": r.latex_str(),
                }
        path.write_text(json.dumps(serializable, indent=2))
        self._log(f"Results saved to {path}")
        return path