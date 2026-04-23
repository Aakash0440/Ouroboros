"""
NeuralNodePrior — Lightweight learned weights for node type selection.

Despite the name, this is NOT a neural network in the deep learning sense.
It is a small learned probability table: P(node_type | environment_features).

Why "neural"? The update rule is gradient-like — when a node type leads to
a lower MDL cost, its weight increases. When it leads to higher cost, it
decreases. This is the core insight of neural guidance without the training
overhead.

Architecture: 
  Input: 6 sequence statistics (entropy, autocorr_lag1, autocorr_lag7, 
         deriv_variance, monotonicity, unique_ratio)
  Output: probability weight per ExtNodeType (40 weights)
  
  The "network" is just a dot product: weight[nt] = dot(stats, feature_vector[nt])
  where feature_vector[nt] is learned from experience.

Training:
  After each successful search (MDL cost < baseline), the node types
  used in the winning expression get their feature vectors updated:
    feature_vector[nt] += learning_rate * stats
  
  This is a form of online learning with O(6 * 40) = O(240) parameters.
  Updates are fast. No GPU needed.

Effect:
  After 50 successful discoveries:
  - On periodic environments (high autocorr): SIN/COS/FFT_AMP get high weights
  - On number-theoretic environments (low unique_ratio): GCD/ISPRIME get high weights
  - On exponential environments (high monotonicity): EXP/EWMA get high weights
"""

from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ouroboros.nodes.extended_nodes import ExtNodeType, NodeCategory, NODE_SPECS
from ouroboros.nodes.extended_nodes import ExtExprNode


# Feature dimension (6 statistics)
N_FEATURES = 6

# Number of extended node types
N_NODES = len(ExtNodeType)

# Learning rate for online updates
LEARNING_RATE = 0.05

# Default weights — uniform start, all 1.0
DEFAULT_WEIGHT = 1.0


@dataclass
class NodePriorStats:
    """Statistics for tracking prior quality."""
    n_updates: int = 0
    n_queries: int = 0
    total_reward: float = 0.0
    node_usage: Dict[str, int] = field(default_factory=dict)


class NeuralNodePrior:
    """
    Lightweight learned prior over node types.
    
    The prior is a 40×6 matrix W where:
      weights[nt][i] = how much feature i predicts usefulness of node nt
    
    Usage:
        prior = NeuralNodePrior()
        stats = [entropy, ac1, ac7, deriv_var, monotone, unique_ratio]
        weights = prior.get_weights(stats)
        # weights: Dict[ExtNodeType, float] — sampling weights
        
        # After a successful search:
        prior.update(stats, successful_expr, reward=improvement_in_bits)
    """

    def __init__(self, learning_rate: float = LEARNING_RATE, seed: int = 42):
        self.lr = learning_rate
        self._rng = random.Random(seed)
        self.stats = NodePriorStats()

        # Weight matrix: for each node type, a 6-dimensional feature weight vector
        self._weights: Dict[str, List[float]] = {}
        for nt in ExtNodeType:
            # Initialize with small random weights around DEFAULT_WEIGHT
            self._weights[nt.name] = [
                DEFAULT_WEIGHT + self._rng.gauss(0, 0.05)
                for _ in range(N_FEATURES)
            ]

        # Category-level weights (faster to update than per-node)
        self._category_weights: Dict[str, List[float]] = {}
        for cat in NodeCategory:
            self._category_weights[cat.name] = [1.0] * N_FEATURES

    def _features_from_stats(self, sequence_stats: List[float]) -> List[float]:
        """
        Normalize sequence statistics to [0, 1] range for the prior.
        
        Expected order: [entropy, autocorr_lag1, autocorr_lag7,
                         deriv_variance_normalized, monotonicity, unique_ratio]
        """
        if len(sequence_stats) < N_FEATURES:
            sequence_stats = sequence_stats + [0.5] * (N_FEATURES - len(sequence_stats))
        
        feats = []
        for i, v in enumerate(sequence_stats[:N_FEATURES]):
            # Clamp to [0, 1] — most stats are already in this range
            if i == 3:  # deriv_variance can be unbounded
                v = min(1.0, math.log1p(abs(v)) / 10.0)
            feats.append(max(0.0, min(1.0, float(v))))
        return feats

    def get_weights(
        self,
        sequence_stats: List[float],
        allowed_types: Optional[List[ExtNodeType]] = None,
    ) -> Dict[str, float]:
        """
        Get sampling weights for all (or specified) node types.
        
        Returns dict: node_type.name → float weight (higher = more likely to sample)
        """
        self.stats.n_queries += 1
        feats = self._features_from_stats(sequence_stats)

        weights = {}
        node_types = allowed_types or list(ExtNodeType)
        for nt in node_types:
            w_vec = self._weights.get(nt.name, [DEFAULT_WEIGHT] * N_FEATURES)
            # Dot product: sum of feature_i * weight_i
            score = sum(f * w for f, w in zip(feats, w_vec))
            # Softplus to keep positive
            weights[nt.name] = math.log1p(math.exp(max(-20, min(20, score))))

        return weights

    def update(
        self,
        sequence_stats: List[float],
        successful_expr: ExtExprNode,
        reward: float,
    ) -> None:
        """
        Update weights based on a successful expression.
        
        reward: improvement in MDL bits (positive = good)
        """
        if reward <= 0:
            return

        feats = self._features_from_stats(sequence_stats)
        used_nodes = self._collect_node_types(successful_expr)
        self.stats.n_updates += 1
        self.stats.total_reward += reward

        # Normalize reward
        norm_reward = min(1.0, reward / 100.0)

        for nt_name in used_nodes:
            if nt_name not in self._weights:
                continue
            w_vec = self._weights[nt_name]
            # Gradient ascent: increase weights for features that were present
            for i in range(N_FEATURES):
                w_vec[i] += self.lr * norm_reward * feats[i]
            # Also update category weights
            try:
                nt = ExtNodeType[nt_name]
                cat = NODE_SPECS[nt].category if nt in NODE_SPECS else None
                if cat and cat.name in self._category_weights:
                    cat_w = self._category_weights[cat.name]
                    for i in range(N_FEATURES):
                        cat_w[i] += self.lr * 0.5 * norm_reward * feats[i]
            except (KeyError, AttributeError):
                pass

            self.stats.node_usage[nt_name] = self.stats.node_usage.get(nt_name, 0) + 1

    def _collect_node_types(self, expr: ExtExprNode) -> List[str]:
        """Collect all ExtNodeType names in an expression tree."""
        result = []
        if hasattr(expr.node_type, 'name'):
            name = expr.node_type.name
            if hasattr(ExtNodeType, name):
                result.append(name)
        if expr.left:
            result.extend(self._collect_node_types(expr.left))
        if expr.right:
            result.extend(self._collect_node_types(expr.right))
        if expr.third:
            result.extend(self._collect_node_types(expr.third))
        return result

    def get_category_weights(
        self, sequence_stats: List[float]
    ) -> Dict[str, float]:
        """Get weights at the category level (faster than per-node)."""
        feats = self._features_from_stats(sequence_stats)
        weights = {}
        for cat_name, w_vec in self._category_weights.items():
            score = sum(f * w for f, w in zip(feats, w_vec))
            weights[cat_name] = max(0.1, math.log1p(math.exp(score)))
        return weights

    def save(self, path: str) -> None:
        """Save learned weights to JSON."""
        data = {
            "weights": self._weights,
            "category_weights": self._category_weights,
            "stats": {
                "n_updates": self.stats.n_updates,
                "n_queries": self.stats.n_queries,
                "total_reward": self.stats.total_reward,
                "node_usage": self.stats.node_usage,
            }
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """Load learned weights from JSON."""
        data = json.loads(Path(path).read_text())
        self._weights = data.get("weights", {})
        self._category_weights = data.get("category_weights", {})
        s = data.get("stats", {})
        self.stats.n_updates = s.get("n_updates", 0)
        self.stats.n_queries = s.get("n_queries", 0)
        self.stats.total_reward = s.get("total_reward", 0.0)
        self.stats.node_usage = s.get("node_usage", {})

    def top_nodes_for_stats(
        self, sequence_stats: List[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Return top-k node types for given sequence statistics."""
        weights = self.get_weights(sequence_stats)
        sorted_weights = sorted(weights.items(), key=lambda x: -x[1])
        return sorted_weights[:top_k]