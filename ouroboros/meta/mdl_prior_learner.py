"""
MetaMDLLearner — Learns the MDL prior from successful discoveries.

The MDL objective currently uses a fixed prior:
  description_bits(expression) = n_nodes * bits_per_node + n_constants * bits_per_constant

This prior assigns the same cost to DERIV(SIN(t)) and MEAN_WIN(CUMSUM(t), 20)
regardless of which is more common in real scientific laws.

The meta-learner observes successful discoveries and updates the prior:
  After discovering 100 physics laws:
    → DERIV and DERIV2 are common → reduce their description cost
    → TOTIENT and GCD_NODE are rare in physics → increase their cost
    → Constants near π, e, √2 are common → reduce cost for these values
    → Deep expressions (depth > 4) rarely succeed → increase depth penalty

This is Bayesian model selection with a learned prior.
It's the difference between a fixed MDL and an adaptive MDL that improves
with experience.

Technical approach:
  Maintain counts of each node type in successful vs failed expressions.
  Update description_bits[node_type] proportionally to
    -log(P(node_type | success)) using Laplace smoothing.

  For constants: maintain a distribution over constant values and
  reduce cost for frequently-successful values.

Stability guarantee:
  Prior updates are bounded: description_bits stays in [0.5, 12.0].
  This prevents collapse to zero-cost for common nodes.
"""

from __future__ import annotations
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ouroboros.nodes.extended_nodes import ExtNodeType, NODE_SPECS


# Default description bits (from the original grammar)
DEFAULT_BITS: Dict[str, float] = {
    "CONST": 6.0, "TIME": 2.0, "PREV": 3.0,
    "ADD": 3.0, "SUB": 3.0, "MUL": 3.0, "DIV": 4.0,
    "MOD": 4.0, "POW": 5.0, "IF": 5.0, "EQ": 4.0, "LT": 4.0,
    "SIN": 4.0, "COS": 4.0, "EXP": 4.0, "LOG": 4.0, "SQRT": 4.0,
}
for nt, spec in NODE_SPECS.items():
    DEFAULT_BITS[nt.name] = spec.description_bits

# Bounds for description bits
MIN_BITS = 0.5
MAX_BITS = 12.0

# Learning rate for prior updates
PRIOR_LR = 0.02
LAPLACE_ALPHA = 2.0  # Laplace smoothing


@dataclass
class PriorState:
    """Current state of the learned prior."""
    description_bits: Dict[str, float]       # node_type → bits
    constant_distribution: Dict[int, float]   # constant_value → frequency
    depth_penalty: float                       # extra bits per depth level
    n_updates: int
    n_successes_seen: int
    domain_priors: Dict[str, Dict[str, float]]  # domain → {node_type → bits}

    def get_bits(self, node_name: str, domain: str = "general") -> float:
        """Get description bits for a node in a specific domain."""
        if domain in self.domain_priors and node_name in self.domain_priors[domain]:
            return self.domain_priors[domain][node_name]
        return self.description_bits.get(node_name, 4.0)

    def to_dict(self) -> dict:
        return {
            "description_bits": self.description_bits,
            "constant_distribution": {str(k): v for k, v in self.constant_distribution.items()},
            "depth_penalty": self.depth_penalty,
            "n_updates": self.n_updates,
            "n_successes_seen": self.n_successes_seen,
        }

    @classmethod
    def default(cls) -> 'PriorState':
        return cls(
            description_bits=dict(DEFAULT_BITS),
            constant_distribution={},
            depth_penalty=0.5,
            n_updates=0,
            n_successes_seen=0,
            domain_priors={},
        )


@dataclass
class PriorUpdate:
    """One update to the MDL prior."""
    expression_str: str
    domain: str
    success: bool             # True if expression promoted through proof market
    mdl_cost: float
    node_counts: Counter      # which nodes appeared in this expression
    depth: int
    generalized: bool         # True if expression worked on OOD data


class MetaMDLLearner:
    """
    Learns the MDL prior from successful discoveries.

    Maintains separate priors for each domain (physics, number_theory, etc.)
    so that physics expressions get physics-appropriate costs and
    number-theory expressions get number-theory-appropriate costs.

    Usage:
        learner = MetaMDLLearner()

        # After a successful discovery in physics:
        learner.update(expr, domain="physics", success=True,
                       mdl_cost=45.0, generalized=True)

        # Get updated description bits for beam search:
        bits = learner.get_description_bits("DERIV", domain="physics")
        # → probably lower than default after many physics discoveries
    """

    def __init__(
        self,
        save_path: Optional[str] = None,
        learning_rate: float = PRIOR_LR,
        min_updates_for_domain: int = 5,
    ):
        self._state = PriorState.default()
        self._save_path = save_path
        self.lr = learning_rate
        self.min_updates = min_updates_for_domain

        # Track per-domain node counts
        self._domain_success_counts: Dict[str, Counter] = defaultdict(Counter)
        self._domain_total_counts: Dict[str, Counter] = defaultdict(Counter)
        self._domain_update_counts: Dict[str, int] = defaultdict(int)

        if save_path and Path(save_path).exists():
            self._load(save_path)

    def update(
        self,
        expr,
        domain: str,
        success: bool,
        mdl_cost: float,
        generalized: bool = False,
    ) -> PriorUpdate:
        """
        Update the prior based on a discovery outcome.

        expr: the discovered expression
        domain: mathematical domain ("physics", "number_theory", etc.)
        success: True if expression was approved (promoted through proof market)
        mdl_cost: the MDL cost of the expression
        generalized: True if expression worked on out-of-distribution data
        """
        # Extract node counts from expression
        node_counts = self._count_nodes(expr)
        depth = expr.depth() if hasattr(expr, 'depth') else 0

        update = PriorUpdate(
            expression_str=expr.to_string() if hasattr(expr, 'to_string') else str(expr),
            domain=domain,
            success=success,
            mdl_cost=mdl_cost,
            node_counts=node_counts,
            depth=depth,
            generalized=generalized,
        )

        # Update counts
        for node_name, count in node_counts.items():
            self._domain_total_counts[domain][node_name] += count
            if success:
                weight = 2.0 if generalized else 1.0
                self._domain_success_counts[domain][node_name] += count * weight
        self._domain_update_counts[domain] += 1
        self._state.n_updates += 1
        if success:
            self._state.n_successes_seen += 1

        # Update depth penalty
        if success:
            # Successful deep expressions → reduce depth penalty slightly
            if depth > 4:
                self._state.depth_penalty = max(0.1, self._state.depth_penalty * 0.99)
        else:
            # Failed shallow expressions → increase penalty slightly
            if depth <= 2 and not success:
                pass  # depth probably not the issue

        # Update domain-specific priors
        if self._domain_update_counts[domain] >= self.min_updates:
            self._update_domain_prior(domain)

        # Update global prior
        self._update_global_prior(node_counts, success, generalized)

        # Save periodically
        if self._state.n_updates % 50 == 0 and self._save_path:
            self._save(self._save_path)

        return update

    def _update_domain_prior(self, domain: str) -> None:
        """Update description bits for a specific domain."""
        success_counts = self._domain_success_counts[domain]
        total_counts = self._domain_total_counts[domain]

        if domain not in self._state.domain_priors:
            self._state.domain_priors[domain] = dict(DEFAULT_BITS)

        domain_bits = self._state.domain_priors[domain]
        total_successes = sum(success_counts.values()) + LAPLACE_ALPHA * len(DEFAULT_BITS)
        total_total = sum(total_counts.values()) + LAPLACE_ALPHA * len(DEFAULT_BITS)

        for node_name in set(list(success_counts.keys()) + list(DEFAULT_BITS.keys())):
            n_success = success_counts.get(node_name, 0) + LAPLACE_ALPHA
            n_total = total_counts.get(node_name, 0) + LAPLACE_ALPHA

            # P(success | used this node)
            p_success_given_node = n_success / max(n_total, 1)

            # Reduce bits for nodes that often appear in successes
            # Increase bits for nodes that rarely appear in successes
            target_bits = DEFAULT_BITS.get(node_name, 4.0)
            if p_success_given_node > 0.5:
                target_bits *= 0.9  # reduce cost
            elif p_success_given_node < 0.2:
                target_bits *= 1.1  # increase cost

            # Apply learning rate update
            current = domain_bits.get(node_name, DEFAULT_BITS.get(node_name, 4.0))
            new_bits = current + self.lr * (target_bits - current)
            domain_bits[node_name] = max(MIN_BITS, min(MAX_BITS, new_bits))

    def _update_global_prior(
        self,
        node_counts: Counter,
        success: bool,
        generalized: bool,
    ) -> None:
        """Update global description bits."""
        weight = (2.0 if generalized else 1.0) if success else -0.5
        for node_name, count in node_counts.items():
            if node_name not in self._state.description_bits:
                self._state.description_bits[node_name] = DEFAULT_BITS.get(node_name, 4.0)
            current = self._state.description_bits[node_name]
            # Move toward lower cost for successful nodes, higher for failed
            delta = -0.1 * weight * count * self.lr
            new_bits = current + delta
            self._state.description_bits[node_name] = max(MIN_BITS, min(MAX_BITS, new_bits))

    def _count_nodes(self, expr) -> Counter:
        """Count node types in an expression tree."""
        counts = Counter()
        if expr is None:
            return counts
        node_name = expr.node_type.name if hasattr(expr.node_type, 'name') else str(expr.node_type)
        counts[node_name] += 1
        for child_attr in ['left', 'right', 'third']:
            child = getattr(expr, child_attr, None)
            if child:
                counts.update(self._count_nodes(child))
        return counts

    def get_description_bits(self, node_name: str, domain: str = "general") -> float:
        """Get the current learned description bits for a node in a domain."""
        return self._state.get_bits(node_name, domain)

    def get_category_weights_for_router(self, domain: str) -> Dict:
        """
        Convert learned priors into category weights for HierarchicalSearchRouter.
        Nodes with lower bits get higher sampling weight.
        """
        from ouroboros.nodes.extended_nodes import NodeCategory
        category_weights = {}
        for cat in NodeCategory:
            # Average bits for this category in this domain
            cat_nodes = [
                nt.name for nt, spec in NODE_SPECS.items()
                if spec.category == cat
            ]
            if not cat_nodes:
                category_weights[cat] = 1.0
                continue
            avg_bits = sum(
                self.get_description_bits(n, domain) for n in cat_nodes
            ) / len(cat_nodes)
            # Lower bits → higher weight
            max_bits = MAX_BITS
            weight = max_bits / max(avg_bits, MIN_BITS)
            category_weights[cat] = max(0.1, min(10.0, weight))
        return category_weights

    def prior_summary(self, domain: str = None) -> str:
        """Print a summary of learned description bits."""
        bits = (self._state.domain_priors.get(domain, self._state.description_bits)
                if domain else self._state.description_bits)

        changed = {
            name: (bits[name], DEFAULT_BITS.get(name, 4.0))
            for name in bits
            if abs(bits.get(name, 0) - DEFAULT_BITS.get(name, 4.0)) > 0.05
        }

        if not changed:
            return f"Prior unchanged ({self._state.n_updates} updates)"

        lines = [f"Learned Prior ({domain or 'global'}, n={self._state.n_updates}):"]
        for name, (learned, default) in sorted(changed.items(), key=lambda x: x[1][0]-x[1][1]):
            direction = "↓ cheaper" if learned < default else "↑ more expensive"
            lines.append(f"  {name}: {default:.2f} → {learned:.2f} {direction}")
        return "\n".join(lines)

    def _save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self._state.to_dict(), indent=2))

    def _load(self, path: str) -> None:
        try:
            data = json.loads(Path(path).read_text())
            self._state.description_bits = data.get("description_bits", DEFAULT_BITS)
            self._state.depth_penalty = data.get("depth_penalty", 0.5)
            self._state.n_updates = data.get("n_updates", 0)
            self._state.n_successes_seen = data.get("n_successes_seen", 0)
        except Exception:
            pass
