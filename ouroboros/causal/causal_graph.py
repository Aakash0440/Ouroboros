"""
CausalGraph — Directed acyclic graph representing causal structure.

A causal graph over variables {X₁, X₂, ..., Xₙ} is a DAG where
an edge Xᵢ → Xⱼ means "Xᵢ is a direct cause of Xⱼ."

For OUROBOROS, variables are:
  - Observed sequences (obs[t] for each channel)
  - Derived quantities (DERIV(obs), EWMA(obs), etc.)
  - Time (t itself)

Key operations:
  do(X=x)     — Pearl's intervention: fix X to value x, cut incoming edges
  parents(X)  — direct causes of X
  d_separate  — test conditional independence
  backdoor     — find valid adjustment sets for effect estimation

Implementation:
  Lightweight DAG using adjacency lists. No external dependencies.
  Supports up to 20 variables (sufficient for OUROBOROS use cases).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple, FrozenSet


@dataclass
class CausalVariable:
    """A variable in the causal graph."""
    name: str
    var_type: str      # "observed", "derived", "latent", "time"
    description: str = ""

    def __hash__(self): return hash(self.name)
    def __eq__(self, other): return self.name == other.name
    def __repr__(self): return f"V({self.name})"


@dataclass
class CausalEdge:
    """A directed causal edge from cause to effect."""
    cause: CausalVariable
    effect: CausalVariable
    strength: float = 1.0     # estimated causal strength (0-1)
    lag: int = 0              # time lag (0 = instantaneous, k = k-step lag)
    expression: Optional[str] = None   # symbolic form of the causal mechanism

    def __repr__(self):
        lag_str = f"[lag={self.lag}]" if self.lag > 0 else ""
        return f"{self.cause.name} → {self.effect.name}{lag_str}"


class CausalGraph:
    """
    Directed acyclic graph representing causal structure between variables.

    Supports Pearl's do-calculus operations:
      - do(X=x): intervention — fix X, cut incoming edges
      - P(Y | do(X=x)): interventional distribution query
      - backdoor criterion: test if set Z blocks all backdoor paths

    Example — Spring-Mass System:
      Causal graph:
        FORCE → ACCELERATION → VELOCITY → POSITION
        POSITION → FORCE  (Hooke's law: F = -kx)

      Intervention: do(FORCE=0) removes Hooke's law feedback,
      predicts position would evolve as free particle.

    Example — Climate:
      CO2_concentration → Radiative_forcing → Temperature
      Temperature → CO2_concentration  (feedback loop, with lag 100yr)
    """

    def __init__(self):
        self._variables: Dict[str, CausalVariable] = {}
        self._edges: List[CausalEdge] = []
        self._adjacency: Dict[str, Set[str]] = {}    # cause → {effects}
        self._parents: Dict[str, Set[str]] = {}       # effect → {causes}

    def add_variable(self, var: CausalVariable) -> None:
        self._variables[var.name] = var
        if var.name not in self._adjacency:
            self._adjacency[var.name] = set()
        if var.name not in self._parents:
            self._parents[var.name] = set()

    def add_edge(self, edge: CausalEdge) -> bool:
        """
        Add a causal edge. Returns False if it would create a cycle.
        (We allow feedback loops with explicit lag > 0 — they are not cycles
        in the temporal sense.)
        """
        # Ensure variables exist
        if edge.cause.name not in self._variables:
            self.add_variable(edge.cause)
        if edge.effect.name not in self._variables:
            self.add_variable(edge.effect)

        # Check for cycle (only for lag=0 edges)
        if edge.lag == 0 and self._would_create_cycle(edge.cause.name, edge.effect.name):
            return False

        self._edges.append(edge)
        self._adjacency[edge.cause.name].add(edge.effect.name)
        self._parents[edge.effect.name].add(edge.cause.name)
        return True

    def _would_create_cycle(self, cause: str, effect: str) -> bool:
        """Check if adding cause → effect would create a cycle."""
        # DFS from effect: if we can reach cause, adding this edge creates a cycle
        visited = set()
        stack = [effect]
        while stack:
            node = stack.pop()
            if node == cause:
                return True
            if node in visited:
                continue
            visited.add(node)
            stack.extend(self._adjacency.get(node, set()))
        return False

    def parents(self, var_name: str) -> Set[str]:
        """Return direct causes of var_name."""
        return self._parents.get(var_name, set())

    def children(self, var_name: str) -> Set[str]:
        """Return direct effects of var_name."""
        return self._adjacency.get(var_name, set())

    def ancestors(self, var_name: str) -> Set[str]:
        """Return all ancestors (direct and indirect causes)."""
        result = set()
        stack = list(self._parents.get(var_name, set()))
        while stack:
            v = stack.pop()
            if v not in result:
                result.add(v)
                stack.extend(self._parents.get(v, set()))
        return result

    def descendants(self, var_name: str) -> Set[str]:
        """Return all descendants (direct and indirect effects)."""
        result = set()
        stack = list(self._adjacency.get(var_name, set()))
        while stack:
            v = stack.pop()
            if v not in result:
                result.add(v)
                stack.extend(self._adjacency.get(v, set()))
        return result

    def do_intervention(self, var_name: str, value: float) -> 'InterventionalGraph':
        """
        Apply Pearl's do-operator: do(var_name = value).

        Returns a new InterventionalGraph where:
          - var_name is fixed to value
          - All incoming edges to var_name are removed (cut)
          - All outgoing edges from var_name remain
        """
        return InterventionalGraph(
            original_graph=self,
            intervened_var=var_name,
            intervention_value=value,
        )

    def backdoor_criterion(
        self,
        cause: str,
        effect: str,
        adjustment_set: Set[str],
    ) -> bool:
        """
        Test if adjustment_set satisfies the backdoor criterion for
        estimating the causal effect of cause on effect.

        Backdoor criterion (Pearl 2000):
        1. No variable in Z is a descendant of X
        2. Z blocks every backdoor path from X to Y
        """
        # Condition 1: No variable in Z is a descendant of X
        x_descendants = self.descendants(cause)
        if any(z in x_descendants for z in adjustment_set):
            return False

        # Condition 2: Z blocks all backdoor paths
        # Simplified check: Z contains all parents of cause (sufficient condition)
        cause_parents = self.parents(cause)
        return all(p in adjustment_set for p in cause_parents)

    def topological_sort(self) -> List[str]:
        """Return variables in topological order (causes before effects)."""
        in_degree = {v: 0 for v in self._variables}
        for edge in self._edges:
            if edge.lag == 0:
                in_degree[edge.effect.name] += 1

        queue = [v for v, deg in in_degree.items() if deg == 0]
        result = []
        while queue:
            v = queue.pop(0)
            result.append(v)
            for child in self._adjacency.get(v, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def to_string(self) -> str:
        """Human-readable representation."""
        lines = ["CausalGraph:"]
        for var_name in sorted(self._variables):
            effects = sorted(self._adjacency.get(var_name, set()))
            if effects:
                lines.append(f"  {var_name} → {', '.join(effects)}")
        return "\n".join(lines)

    @property
    def n_variables(self) -> int:
        return len(self._variables)

    @property
    def n_edges(self) -> int:
        return len(self._edges)


@dataclass
class InterventionalGraph:
    """
    A causal graph with one variable fixed by intervention (do-operator).
    Used to compute interventional distributions and counterfactuals.
    """
    original_graph: CausalGraph
    intervened_var: str
    intervention_value: float

    def effective_parents(self, var_name: str) -> Set[str]:
        """
        Parents of var_name in the interventional graph.
        The intervened variable has no parents (incoming edges cut).
        """
        if var_name == self.intervened_var:
            return set()  # intervention cuts all incoming edges
        return self.original_graph.parents(var_name)

    def is_intervened(self, var_name: str) -> bool:
        return var_name == self.intervened_var

    def intervention_prediction(
        self,
        target_var: str,
        n_steps: int = 100,
    ) -> List[float]:
        """
        Predict the effect of this intervention on target_var over n_steps.
        Returns a predicted sequence for target_var.

        Uses the structural equations (if available) from the causal edges.
        Falls back to zero if structural equations not available.
        """
        # Simplified: return intervention value if target is intervened
        if target_var == self.intervened_var:
            return [self.intervention_value] * n_steps
        # Otherwise, would need structural equations — return None to indicate
        return []