# ouroboros/agents/society.py

"""
AgentSociety — manages a group of agents on one environment.

Phase 1: just runs agents in parallel, collects metrics.
Phase 2: adds proof market interactions.
Phase 3: adds specialization roles.

This is the main object you interact with in experiments.
You don't call individual agents directly — you call society.run_episode().
"""

from typing import List, Dict, Optional
import numpy as np
from rich.table import Table
from rich.console import Console

from ouroboros.agents.base_agent import BaseAgent
from ouroboros.environmentss.base import ObservationEnvironment
from ouroboros.utils.logger import MetricsWriter, get_logger

console = Console()


class AgentSociety:
    """
    A society of agents compressing the same environment.

    All agents see the same stream.
    Each agent independently searches for the best program.
    The society tracks collective metrics.

    Args:
        num_agents: Number of agents (default 8)
        environment: The observation environment
        alphabet_size: Symbol count (must match environment)
        max_context_length: Maximum n-gram k for all agents
        lambda_weight: MDL regularization weight
        seed: Base random seed (agents get seed, seed+17, seed+34, ...)
    """

    def __init__(
        self,
        num_agents: int,
        environment: ObservationEnvironment,
        alphabet_size: int,
        max_context_length: int = 8,
        lambda_weight: float = 1.0,
        seed: int = 42,
    ):
        self.environment = environment
        self.alphabet_size = alphabet_size
        self.logger = get_logger('AgentSociety')

        self.agents: List[BaseAgent] = [
            BaseAgent(
                agent_id=i,
                alphabet_size=alphabet_size,
                max_context_length=max_context_length,
                lambda_weight=lambda_weight,
                seed=seed,
            )
            for i in range(num_agents)
        ]

    def run_episode(
        self,
        stream_length: int = 2000,
        eval_interval: int = 200,
        writer: Optional[MetricsWriter] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Run one full episode: all agents compress the stream.

        Evaluation at every eval_interval steps:
        - Each agent searches for a better program on the prefix so far
        - Compression ratio is measured and logged

        Args:
            stream_length: Total observation stream length
            eval_interval: Steps between evaluations
            writer: Optional metrics writer
            verbose: If True, print agent status at each eval

        Returns:
            Summary dict with final compression ratios per agent
            plus 'mean_ratio' and 'best_ratio'
        """
        # Generate the stream
        self.environment.reset(stream_length)
        full_stream = self.environment.peek_all()

        checkpoints = list(range(eval_interval, stream_length + eval_interval, eval_interval))

        for step in checkpoints:
            prefix = full_stream[:step]

            for agent in self.agents:
                # Update agent's history to this prefix
                agent.set_history(prefix)
                cost = agent.search_and_update()
                ratio = agent.measure_compression_ratio()

                if writer:
                    writer.write(
                        step=step,
                        agent_id=agent.agent_id,
                        compression_ratio=ratio,
                        mdl_cost=cost,
                        context_k=agent.program.context_length,
                        table_entries=agent.program.num_entries,
                    )

            if verbose:
                ratios = [a.latest_ratio() for a in self.agents]
                self.logger.info(
                    f"Step {step}: "
                    f"mean_ratio={np.mean(ratios):.4f}, "
                    f"best={min(ratios):.4f}"
                )

        # Final results
        results: Dict = {}
        for agent in self.agents:
            # Restore full history for final measurement
            agent.set_history(full_stream)
            ratio = agent.measure_compression_ratio()
            results[f'agent_{agent.agent_id}'] = ratio

        ratios = list(results.values())
        results['mean_ratio'] = float(np.mean(ratios))
        results['best_ratio'] = float(min(ratios))
        results['worst_ratio'] = float(max(ratios))
        results['stream_length'] = stream_length
        results['num_agents'] = len(self.agents)

        return results

    def print_status(self, title: str = "Agent Society Status") -> None:
        """Print agent status as a Rich table."""
        table = Table(title=title)
        table.add_column("ID",     style="cyan",    width=4)
        table.add_column("k",      style="green",   width=4)
        table.add_column("Entries",style="yellow",  width=8)
        table.add_column("Prog Bits", style="magenta", width=10)
        table.add_column("Ratio",  style="red",     width=8)
        table.add_column("Searches", style="blue",  width=9)
        table.add_column("Obs",    style="white",   width=7)

        for agent in self.agents:
            ratio = agent.latest_ratio()
            color = "green" if ratio < 0.5 else "yellow" if ratio < 0.85 else "red"
            table.add_row(
                str(agent.agent_id),
                str(agent.program.context_length),
                str(agent.program.num_entries),
                f"{agent.program.description_bits:.0f}",
                f"[{color}]{ratio:.4f}[/{color}]",
                str(agent.search_count),
                str(len(agent.observation_history)),
            )

        console.print(table)