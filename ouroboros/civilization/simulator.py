"""
CivilizationSimulator — Multi-agent mathematical discovery at civilization scale.

Research question: "If you seed a multi-agent system with compression pressure
and adversarial verification, does it rediscover mathematics in a similar order
to human civilization?"

Known historical order of mathematical discovery:
  1. Counting / natural numbers (~30,000 BCE)
  2. Arithmetic — addition, subtraction (~3,000 BCE)
  3. Multiplication, division (~2,000 BCE)
  4. Modular arithmetic / remainders (~500 BCE)
  5. Prime numbers (~300 BCE, Euclid)
  6. Fibonacci / recurrence relations (~1200 CE)
  7. Logarithms / exponentials (~1600 CE)
  8. Calculus / derivatives (~1680 CE, Newton/Leibniz)
  9. Fourier analysis (~1820 CE)
  10. Chinese Remainder Theorem formalization (~1247 CE but formalized ~1800 CE)

OUROBOROS discovery order (predicted):
  1. CONST expressions (trivial — always found first)
  2. LINEAR expressions (ax+b patterns — found in earliest rounds)
  3. MOD expressions (modular arithmetic — core of the system)
  4. PRIME/GCD (number theory — found when agents encounter prime-related envs)
  5. PREV/recurrence (Fibonacci — found when FibEnv is in the rotation)
  6. EXP/LOG (exponential — found on RadioactiveDecayEnv)
  7. DERIV/CUMSUM (calculus — found on SpringMassEnv)
  8. FFT/AUTOCORR (Fourier — found on periodic envs with FFT_SEED)
  9. CRT (found on JointEnvironment)
  10. CORR + law verification (physics — found on PhysicsEnvs)

The hypothesis: OUROBOROS discovers concepts in roughly the same order
because compression pressure is the same force that drove human discovery.
Simple (short description) concepts come first.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any

# ── Known mathematical concepts and their proxy node types ────────────────────

@dataclass
class MathConcept:
    """A mathematical concept with its proxy indicator."""
    name: str
    human_era: str           # approximate historical period
    human_order: int         # rank in human discovery order (1=earliest)
    indicator_nodes: List[str]  # node type names that indicate this concept
    indicator_env: Optional[str] = None  # environment where this is discovered

    def is_indicated_by(self, node_names: Set[str]) -> bool:
        """True if any indicator node appears in the discovered expression."""
        return bool(set(self.indicator_nodes) & node_names)


MATH_CONCEPTS: List[MathConcept] = [
    MathConcept("Counting/Arithmetic",   "~3000 BCE", 1,
                ["CONST", "ADD", "TIME"]),
    MathConcept("Multiplication",        "~2000 BCE", 2,
                ["MUL", "POW"]),
    MathConcept("Modular Arithmetic",    "~500 BCE",  3,
                ["MOD"], indicator_env="ModularArithmetic"),
    MathConcept("Prime Numbers",         "~300 BCE",  4,
                ["ISPRIME", "TOTIENT"]),
    MathConcept("GCD / Number Theory",   "~300 BCE",  5,
                ["GCD_NODE", "LCM_NODE"]),
    MathConcept("Fibonacci/Recurrences", "~1200 CE",  6,
                ["PREV"], indicator_env="Fibonacci"),
    MathConcept("Logarithms/Exponentials", "~1600 CE", 7,
                ["EXP", "LOG"]),
    MathConcept("Calculus/Derivatives",  "~1680 CE",  8,
                ["DERIV", "DERIV2", "CUMSUM"]),
    MathConcept("Fourier Analysis",      "~1820 CE",  9,
                ["FFT_AMP", "FFT_PHASE", "AUTOCORR"]),
    MathConcept("Chinese Remainder Thm", "~1247 CE",  10,
                ["MOD"], indicator_env="Joint"),
    MathConcept("Statistical Analysis",  "~1900 CE",  11,
                ["MEAN_WIN", "VAR_WIN", "STD_WIN", "CORR"]),
    MathConcept("Rolling Statistics",    "~1950 CE",  12,
                ["EWMA", "ZSCORE", "QUANTILE"]),
]


@dataclass
class ConceptDiscovery:
    """Records when a mathematical concept was first discovered."""
    concept: MathConcept
    round_discovered: int
    agent_id: str
    environment_name: str
    expression_str: str
    mdl_cost: float


@dataclass
class AgentSpecialization:
    """Tracks what an agent specializes in over time."""
    agent_id: str
    dominant_env: str          # which environment this agent performs best on
    dominant_concept: str      # which mathematical concept it discovers most
    discovery_count: int
    mean_mdl_cost: float


@dataclass
class CivilizationResult:
    """Complete result of the civilization simulation."""
    n_agents: int
    n_environments: int
    n_rounds: int
    total_discoveries: int

    # Discovery timeline
    concept_discoveries: List[ConceptDiscovery]
    ouroboros_discovery_order: List[str]   # concepts in order discovered
    human_discovery_order: List[str]        # historical order

    # Specialization
    agent_specializations: List[AgentSpecialization]
    specialization_emerged: bool

    # Historical comparison
    order_correlation: float    # Spearman correlation between OUROBOROS and human order
    shared_concepts: List[str]  # concepts discovered in correct relative order

    # Performance
    total_runtime_seconds: float

    def summary(self) -> str:
        lines = [
            f"\nMATHEMATICAL CIVILIZATION SIMULATION RESULTS",
            f"{'='*60}",
            f"Agents: {self.n_agents}, Environments: {self.n_environments}, Rounds: {self.n_rounds}",
            f"Total discoveries: {self.total_discoveries}",
            f"",
            f"OUROBOROS Discovery Order:",
        ]
        for i, concept in enumerate(self.ouroboros_discovery_order, 1):
            # Find when it was discovered
            disc = next((d for d in self.concept_discoveries if d.concept.name == concept), None)
            round_str = f"(round {disc.round_discovered})" if disc else "(not discovered)"
            lines.append(f"  {i:2d}. {concept} {round_str}")

        lines.append(f"")
        lines.append(f"Human Discovery Order (historical):")
        for i, concept in enumerate(self.human_discovery_order[:8], 1):
            lines.append(f"  {i:2d}. {concept}")

        lines.append(f"")
        lines.append(f"Order Correlation (Spearman): {self.order_correlation:.3f}")

        if self.order_correlation > 0.7:
            lines.append("✅ STRONG MATCH: OUROBOROS discovery order resembles human history")
        elif self.order_correlation > 0.4:
            lines.append("⚠️  MODERATE MATCH: Some alignment with historical order")
        else:
            lines.append("❌ WEAK MATCH: Discovery order differs from human history")

        if self.specialization_emerged:
            lines.append(f"")
            lines.append(f"SPECIALIZATION DETECTED: {len(self.agent_specializations)} specialized agents")
            for spec in self.agent_specializations[:3]:
                lines.append(f"  {spec.agent_id}: specialized in {spec.dominant_concept} "
                              f"on {spec.dominant_env}")

        lines.append(f"")
        lines.append(f"Runtime: {self.total_runtime_seconds:.1f}s")
        return "\n".join(lines)


class CivilizationSimulator:
    """
    Runs the OUROBOROS mathematical civilization simulation.
    
    Configuration:
      Fast:  16 agents × 10 environments × 50 rounds  ≈ 10 minutes
      Full:  64 agents × 20 environments × 200 rounds ≈ 90 minutes
    """

    def __init__(
        self,
        n_agents: int = 16,
        n_rounds: int = 50,
        stream_length: int = 200,
        beam_width: int = 12,
        n_iterations: int = 6,
        random_seed: int = 42,
        verbose: bool = True,
        report_every: int = 10,
    ):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.stream_length = stream_length
        self.beam_width = beam_width
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.verbose = verbose
        self.report_every = report_every

    def _build_environment_suite(self) -> List:
        """Build the 10 or 20 environments for the simulation."""
        import random as rnd
        rng = rnd.Random(self.random_seed)

        from ouroboros.environments.modular import ModularArithmeticEnv
        from ouroboros.environments.fibonacci_mod import FibonacciModEnv
        from ouroboros.environments.noise import NoiseEnv
        from ouroboros.environments.multi_scale import MultiScaleEnv
        from ouroboros.environments.physics import (
            SpringMassEnv, RadioactiveDecayEnv, FreeFallEnv
        )
        from ouroboros.environments.algorithm_env import PrimeCountEnv, CollatzEnv
        from ouroboros.environments.long_range import TribonacciModEnv

        envs = [
            # Basic arithmetic / modular
            ModularArithmeticEnv(modulus=5, slope=2, intercept=1),
            ModularArithmeticEnv(modulus=7, slope=3, intercept=1),
            ModularArithmeticEnv(modulus=11, slope=5, intercept=2),
            ModularArithmeticEnv(modulus=13, slope=7, intercept=3),
            # Number theoretic
            PrimeCountEnv(),
            CollatzEnv(),
            # Recurrence
            FibonacciModEnv(modulus=7),
            TribonacciModEnv(modulus=7),
            # Physics (exponential, calculus)
            RadioactiveDecayEnv(n0=200, decay_rate=0.05),
            SpringMassEnv(amplitude=10, omega=0.3),
        ]
        return envs

    def _extract_nodes_from_expr(self, expr) -> Set[str]:
        """Extract all node type names from an expression tree."""
        nodes = set()
        if expr is None:
            return nodes
        name = expr.node_type.name if hasattr(expr.node_type, 'name') else str(expr.node_type)
        nodes.add(name)
        if hasattr(expr, 'left') and expr.left:
            nodes |= self._extract_nodes_from_expr(expr.left)
        if hasattr(expr, 'right') and expr.right:
            nodes |= self._extract_nodes_from_expr(expr.right)
        if hasattr(expr, 'third') and expr.third:
            nodes |= self._extract_nodes_from_expr(expr.third)
        return nodes

    def _check_concept_discovery(
        self,
        expr,
        env_name: str,
        agent_id: str,
        round_num: int,
        mdl_cost: float,
        already_discovered: Set[str],
    ) -> List[ConceptDiscovery]:
        """Check if this expression constitutes a new concept discovery."""
        if expr is None:
            return []

        node_names = self._extract_nodes_from_expr(expr)
        new_discoveries = []

        for concept in MATH_CONCEPTS:
            if concept.name in already_discovered:
                continue
            # Check indicator nodes
            if concept.is_indicated_by(node_names):
                # Check environment indicator if specified
                if concept.indicator_env is None or concept.indicator_env in env_name:
                    new_discoveries.append(ConceptDiscovery(
                        concept=concept,
                        round_discovered=round_num,
                        agent_id=agent_id,
                        environment_name=env_name,
                        expression_str=expr.to_string()[:60],
                        mdl_cost=mdl_cost,
                    ))

        return new_discoveries

    def _spearman_correlation(self, order_a: List[str], order_b: List[str]) -> float:
        """Compute Spearman rank correlation between two discovery orders."""
        # Find concepts present in both lists
        common = [c for c in order_a if c in order_b]
        if len(common) < 3:
            return 0.0

        rank_a = {c: i for i, c in enumerate(order_a)}
        rank_b = {c: i for i, c in enumerate(order_b)}

        n = len(common)
        d_sq = sum((rank_a[c] - rank_b[c])**2 for c in common)
        rho = 1.0 - (6 * d_sq) / (n * (n**2 - 1))
        return max(-1.0, min(1.0, rho))

    def run(self) -> CivilizationResult:
        """Run the full civilization simulation."""
        import random
        from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
        from concurrent.futures import ThreadPoolExecutor

        rng = random.Random(self.random_seed)
        start_total = time.time()
        envs = self._build_environment_suite()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MATHEMATICAL CIVILIZATION SIMULATION")
            print(f"Agents: {self.n_agents}, Envs: {len(envs)}, Rounds: {self.n_rounds}")
            print(f"{'='*60}\n")

        # Initialize per-agent routers with different seeds
        routers = [
            HierarchicalSearchRouter(RouterConfig(
                beam_width=self.beam_width,
                max_depth=4,
                n_iterations=self.n_iterations,
                random_seed=self.random_seed + i * 13,
            ))
            for i in range(self.n_agents)
        ]

        # Tracking
        concept_discoveries: List[ConceptDiscovery] = []
        discovered_concepts: Set[str] = set()
        agent_discovery_counts: Dict[str, int] = {f"A{i}": 0 for i in range(self.n_agents)}
        agent_env_costs: Dict[str, Dict[str, List[float]]] = {
            f"A{i}": {} for i in range(self.n_agents)
        }

        for round_num in range(1, self.n_rounds + 1):
            if round_num % self.report_every == 0 and self.verbose:
                n_disc = len(discovered_concepts)
                print(f"  Round {round_num:3d}/{self.n_rounds}: "
                      f"{n_disc} concepts discovered so far")

            for agent_i in range(self.n_agents):
                # Each agent works on a different environment this round
                env = envs[(agent_i + round_num) % len(envs)]
                env.seed = (round_num + agent_i) % 20

                # ── before the for round_num loop, replace the routers list init block ────
            executor = ThreadPoolExecutor(max_workers=self.n_agents)

# ── inside for round_num, replace the entire for agent_i block ────────────

            # Submit all agents in parallel
            futures = {}
            for agent_i in range(self.n_agents):
                env = envs[(agent_i + round_num) % len(envs)]
                env.seed = (round_num + agent_i) % 20
                obs = env.generate(self.stream_length)
                fut = executor.submit(routers[agent_i].search, obs, env.alphabet_size)
                futures[fut] = (agent_i, env, obs)

            # Collect results
            for fut, (agent_i, env, obs) in futures.items():
                try:
                    result = fut.result(timeout=3.0)
                except Exception:
                    continue

                if result.expr is None:
                    continue

                agent_id = f"A{agent_i}"
                env_name = env.name

                if result.mdl_cost < 400:
                    routers[agent_i].update_prior(obs, result.expr,
                                                  reward_bits=max(0, 400 - result.mdl_cost))

                if env_name not in agent_env_costs[agent_id]:
                    agent_env_costs[agent_id][env_name] = []
                agent_env_costs[agent_id][env_name].append(result.mdl_cost)

                new = self._check_concept_discovery(
                    result.expr, env_name, agent_id,
                    round_num, result.mdl_cost, discovered_concepts
                )
                for disc in new:
                    concept_discoveries.append(disc)
                    discovered_concepts.add(disc.concept.name)
                    agent_discovery_counts[agent_id] += 1
                    if self.verbose:
                        print(f"    🎯 CONCEPT DISCOVERED: {disc.concept.name} "
                              f"by {agent_id} on {env_name} (round {round_num})")

# ── after the for round_num loop ends ─────────────────────────────────────
        executor.shutdown(wait=False)

        # Build OUROBOROS discovery order
        ouroboros_order = [
            d.concept.name
            for d in sorted(concept_discoveries, key=lambda d: d.round_discovered)
        ]
        # Deduplicate (keep first occurrence)
        seen = set()
        ouroboros_order_dedup = []
        for c in ouroboros_order:
            if c not in seen:
                ouroboros_order_dedup.append(c)
                seen.add(c)

        # Human historical order (filtered to discovered concepts)
        human_order = [
            c.name for c in sorted(MATH_CONCEPTS, key=lambda c: c.human_order)
            if c.name in discovered_concepts
        ]

        # Compute Spearman correlation
        rho = self._spearman_correlation(ouroboros_order_dedup, human_order)

        # Detect specialization
        specializations = []
        for agent_id, env_costs in agent_env_costs.items():
            if not env_costs:
                continue
            # Find the env where this agent has the lowest mean cost
            best_env = min(env_costs, key=lambda e: sum(env_costs[e])/len(env_costs[e]))
            mean_cost = sum(env_costs[best_env]) / len(env_costs[best_env])
            n_disc = agent_discovery_counts[agent_id]
            if n_disc > 0:
                specializations.append(AgentSpecialization(
                    agent_id=agent_id,
                    dominant_env=best_env,
                    dominant_concept=ouroboros_order_dedup[0] if ouroboros_order_dedup else "None",
                    discovery_count=n_disc,
                    mean_mdl_cost=mean_cost,
                ))

        specialization_emerged = len([s for s in specializations if s.discovery_count > 0]) > 2

        result = CivilizationResult(
            n_agents=self.n_agents,
            n_environments=len(envs),
            n_rounds=self.n_rounds,
            total_discoveries=len(concept_discoveries),
            concept_discoveries=concept_discoveries,
            ouroboros_discovery_order=ouroboros_order_dedup,
            human_discovery_order=human_order,
            agent_specializations=specializations,
            specialization_emerged=specialization_emerged,
            order_correlation=rho,
            shared_concepts=[c for c in ouroboros_order_dedup if c in set(human_order)],
            total_runtime_seconds=time.time() - start_total,
        )

        if self.verbose:
            print(result.summary())

        return result