"""
Phase1Runner — unified interface for running all Phase 1 experiments.

Encapsulates:
    - Environment setup
    - Agent creation (BaseAgent, SynthesisAgent, or HierarchicalAgent)
    - Society simulation (checkpoints, logging)
    - Axiom pool management (ProtoAxiomPool or ScaleAxiomPool)
    - Results collection and serialization

Usage:
    runner = Phase1Runner.for_modular_arithmetic(7, 3, 1)
    results = runner.run(stream_length=2000, eval_interval=200)
    runner.save_results('experiments/phase1/results/modular_run.json')
    runner.plot_all('experiments/phase1/results/')

This is the class you'll use repeatedly in Days 5–10 and Phase 2.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Type
from dataclasses import dataclass, field
import numpy as np

from ouroboros.core.config import OuroborosConfig
from ouroboros.environment.base import ObservationEnvironment
from ouroboros.environment.structured import (
    BinaryRepeatEnv, ModularArithmeticEnv, FibonacciModEnv,
    PrimeSequenceEnv, MultiScaleEnv, NoiseEnv
)
from ouroboros.agents.base_agent import BaseAgent
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.agents.hierarchical_agent import HierarchicalAgent
from ouroboros.emergence.proto_axiom_pool import ProtoAxiomPool
from ouroboros.emergence.scale_axiom_pool import ScaleAxiomPool
from ouroboros.compression.mdl import naive_bits
from ouroboros.utils.logger import MetricsWriter, get_logger
from ouroboros.utils.visualize import (
    plot_compression_curves, plot_environment_comparison,
    plot_discovery_event
)


@dataclass
class Phase1Results:
    """
    Complete results from a Phase 1 run.

    Fields:
        environment_name: Name of the environment
        stream_length: Total observations
        num_agents: Number of agents
        agent_type: 'base', 'synthesis', or 'hierarchical'
        final_ratios: Per-agent final compression ratios
        mean_ratio: Mean across agents
        best_ratio: Best (lowest) ratio
        axioms_promoted: List of promoted axiom dicts
        discovery_step: Step at which first axiom was promoted (or None)
        discovery_expression: Discovered expression string (or None)
        elapsed_seconds: Total run time
        run_dir: Where metrics were saved
    """
    environment_name: str
    stream_length: int
    num_agents: int
    agent_type: str
    final_ratios: Dict[int, float] = field(default_factory=dict)
    mean_ratio: float = 1.0
    best_ratio: float = 1.0
    axioms_promoted: List[Dict] = field(default_factory=list)
    discovery_step: Optional[int] = None
    discovery_expression: Optional[str] = None
    elapsed_seconds: float = 0.0
    run_dir: str = ''

    def to_dict(self) -> Dict:
        return {
            'environment': self.environment_name,
            'stream_length': self.stream_length,
            'num_agents': self.num_agents,
            'agent_type': self.agent_type,
            'final_ratios': self.final_ratios,
            'mean_ratio': round(self.mean_ratio, 4),
            'best_ratio': round(self.best_ratio, 4),
            'axioms_promoted': self.axioms_promoted,
            'discovery_step': self.discovery_step,
            'discovery_expression': self.discovery_expression,
            'elapsed_seconds': round(self.elapsed_seconds, 2),
            'run_dir': self.run_dir,
        }


class Phase1Runner:
    """
    Runs complete Phase 1 experiments.

    Args:
        environment: Observation environment
        num_agents: Number of agents
        agent_type: 'base' | 'synthesis' | 'hierarchical'
        config: OuroborosConfig (uses defaults if None)
        scales: For hierarchical agent (ignored otherwise)
        run_dir: Directory for metrics and plots
        seed: Base random seed
    """

    def __init__(
        self,
        environment: ObservationEnvironment,
        environment_name: str,
        num_agents: int = 8,
        agent_type: str = 'synthesis',
        config: Optional[OuroborosConfig] = None,
        scales: Optional[List[int]] = None,
        run_dir: str = 'experiments/phase1/runs/unnamed',
        seed: int = 42
    ):
        self.environment = environment
        self.environment_name = environment_name
        self.num_agents = num_agents
        self.agent_type = agent_type
        self.config = config or OuroborosConfig()
        self.scales = scales or [1, 4, 16, 64]
        self.run_dir = run_dir
        self.seed = seed
        self.logger = get_logger('Phase1Runner')

        Path(run_dir).mkdir(parents=True, exist_ok=True)
        self._agents: List[BaseAgent] = []
        self._results: Optional[Phase1Results] = None

    # ─── Factory methods for common setups ────────────────────────────────

    @classmethod
    def for_modular_arithmetic(
        cls,
        modulus: int = 7,
        slope: int = 3,
        intercept: int = 1,
        num_agents: int = 8,
        run_dir: Optional[str] = None,
        **kwargs
    ) -> 'Phase1Runner':
        env = ModularArithmeticEnv(modulus, slope, intercept, seed=42)
        name = f"ModularArith({modulus},{slope},{intercept})"
        rdir = run_dir or f"experiments/phase1/runs/modular_{modulus}_{slope}_{intercept}"
        return cls(env, name, num_agents, 'synthesis', run_dir=rdir, **kwargs)

    @classmethod
    def for_multiscale(
        cls,
        slow_period: int = 28,
        fast_period: int = 7,
        num_agents: int = 6,
        run_dir: Optional[str] = None,
        **kwargs
    ) -> 'Phase1Runner':
        env = MultiScaleEnv(slow_period, fast_period, 0.03, seed=42)
        name = f"MultiScale(slow={slow_period},fast={fast_period})"
        scales = [1, 4, slow_period // 4, slow_period // 2]
        scales = sorted(set(max(1, s) for s in scales))
        rdir = run_dir or f"experiments/phase1/runs/multiscale_{slow_period}_{fast_period}"
        return cls(env, name, num_agents, 'hierarchical', scales=scales, run_dir=rdir, **kwargs)

    @classmethod
    def for_noise_baseline(cls, num_agents: int = 4, **kwargs) -> 'Phase1Runner':
        env = NoiseEnv(4, seed=42)
        kwargs.setdefault('run_dir', 'experiments/phase1/runs/noise_baseline')
        return cls(env, "Noise", num_agents, 'synthesis', **kwargs)

    # ─── Agent creation ───────────────────────────────────────────────────

    def _create_agents(self, alphabet_size: int) -> List[BaseAgent]:
        cfg = self.config.synthesis
        agents = []
        for i in range(self.num_agents):
            s = self.seed + i * 13
            if self.agent_type == 'base':
                agent = BaseAgent(i, alphabet_size, seed=s)
            elif self.agent_type == 'synthesis':
                agent = SynthesisAgent(
                    i, alphabet_size,
                    beam_width=cfg.beam_width,               # ← cfg IS already self.config.compression
                    max_depth=cfg.max_depth,
                    const_range=cfg.const_range,
                    seed=s
                )
            elif self.agent_type == 'hierarchical':
                agent = HierarchicalAgent(
                    i, alphabet_size,
                    scales=self.scales,
                    beam_width=cfg.beam_width,
                    max_depth=cfg.max_depth,
                    const_range=cfg.const_range,
                    seed=s
                )
            else:
                raise ValueError(f"Unknown agent_type: {self.agent_type}")
            agents.append(agent)
        return agents

    # ─── Main run ─────────────────────────────────────────────────────────

    def run(
        self,
        stream_length: int = 2000,
        eval_interval: int = 200,
        consensus_threshold: float = 0.5,
        verbose: bool = True
    ) -> Phase1Results:
        """
        Run the full Phase 1 experiment.

        Args:
            stream_length: Total symbols to generate
            eval_interval: Checkpoint interval
            consensus_threshold: Axiom promotion threshold
            verbose: Print progress

        Returns:
            Phase1Results with all metrics
        """
        start_time = time.time()

        alpha = self.environment.alphabet_size
        self.environment.reset(stream_length)
        stream = self.environment.peek_all()
        nb = naive_bits(stream, alpha)

        self._agents = self._create_agents(alpha)

        # Axiom pool
        pool = ProtoAxiomPool(
            self.num_agents, consensus_threshold, alpha, fingerprint_length=80
        )

        results = Phase1Results(
            environment_name=self.environment_name,
            stream_length=stream_length,
            num_agents=self.num_agents,
            agent_type=self.agent_type,
            run_dir=self.run_dir
        )

        checkpoints = list(range(eval_interval, stream_length + eval_interval, eval_interval))

        with MetricsWriter(self.run_dir) as writer:
            for step in checkpoints:
                data = stream[:min(step, stream_length)]
                pool.clear_submissions()

                if verbose:
                    print(f"  Step {step}/{stream_length} ...", end='\r')

                for agent in self._agents:
                    agent.observation_history = list(data)
                    agent.search_and_update()
                    ratio = agent.measure_compression_ratio()

                    writer.write(
                        step=step,
                        agent_id=agent.agent_id,
                        compression_ratio=ratio,
                        context_length=agent.program.context_length,
                        expression=getattr(agent, 'expression_string',
                                          lambda: f"ngram(k={agent.program.context_length})")(),
                    )

                    # Submit to axiom pool
                    if hasattr(agent, 'best_expression') and agent.best_expression:
                        pool.submit(agent.agent_id, agent.best_expression, ratio * nb, step)

                # Detect consensus
                new_axioms = pool.detect_consensus(step, self.environment_name, nb)
                for ax in new_axioms:
                    if results.discovery_step is None:
                        results.discovery_step = step
                        results.discovery_expression = ax.expression.to_string()
                    results.axioms_promoted.append({
                        'axiom_id': ax.axiom_id,
                        'expression': ax.expression.to_string(),
                        'step': step,
                        'confidence': round(ax.confidence, 4),
                        'support': len(ax.supporting_agents),
                    })

        # Final ratios
        for agent in self._agents:
            agent.observation_history = list(stream)
            ratio = agent.measure_compression_ratio()
            results.final_ratios[agent.agent_id] = round(ratio, 4)

        results.mean_ratio = round(float(np.mean(list(results.final_ratios.values()))), 4)
        results.best_ratio = round(float(min(results.final_ratios.values())), 4)
        results.elapsed_seconds = time.time() - start_time

        if verbose:
            print(f"\n  ✅ {self.environment_name}: mean={results.mean_ratio:.4f}  "
                  f"best={results.best_ratio:.4f}  "
                  f"axioms={len(results.axioms_promoted)}  "
                  f"time={results.elapsed_seconds:.1f}s")

        self._results = results
        return results

    def save_results(self, path: str) -> None:
        """Save results as JSON."""
        if self._results is None:
            raise RuntimeError("No results yet. Call run() first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._results.to_dict(), f, indent=2)
        print(f"Results saved to {path}")

    def plot_all(self, results_dir: str) -> List[str]:
        """Generate all plots. Returns list of saved paths."""
        if self._results is None:
            raise RuntimeError("No results yet. Call run() first.")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        plots = []

        # Compression curves
        p1 = plot_compression_curves(
            self.run_dir,
            title=f'{self.environment_name} — Compression Over Time',
            save_path=f"{results_dir}/compression_curves.png"
        )
        plots.append(p1)

        # Discovery event (if axiom was found)
        if self._results.discovery_step and self._agents:
            # Find which agent made the first discovery
            best_agent_id = min(
                self._results.final_ratios,
                key=self._results.final_ratios.get
            )
            p2 = plot_discovery_event(
                self.run_dir,
                agent_id=best_agent_id,
                discovery_step=self._results.discovery_step,
                expression_found=self._results.discovery_expression or '',
                save_path=f"{results_dir}/discovery_event.png"
            )
            plots.append(p2)

        return plots