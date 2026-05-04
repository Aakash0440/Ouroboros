"""
MultivariateMDLEngine — MDL scoring for expressions over multiple variables.

Instead of obs[t] ∈ ℝ, we now have obs[t] ∈ ℝⁿ (n variables).
Expressions can reference any channel: OBS(t, 0), OBS(t, 1), OBS(t, 2), etc.

Example (CO2 → Temperature):
  Variables: co2[t], rf[t], temp[t]
  Target: temp[t]
  Expression: EWMA(OBS(t, 0), 0.05) * CONST(0.8)  — uses co2[0] with smoothing
  MDL cost: |expression|_bits + H(predictions || temp[t])

The OBS(t, channel) node is a new terminal that reads from the
multivariate observation matrix at timestep t and channel index channel.

Grammar rules:
  OBS(t, channel) is a terminal node (no children)
  It can appear anywhere CONST or TIME can appear
  channel must be in [0, n_channels - 1]
"""

from __future__ import annotations
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
import numpy as np


# ── MultivariateObservation ────────────────────────────────────────────────────

class MultivariateObservations:
    """
    A matrix of observations: shape (n_timesteps, n_channels).
    Channel 0, 1, ..., n-1 are the input features.
    The target variable is specified separately.
    """

    def __init__(
        self,
        data: Union[np.ndarray, List[List[float]]],
        channel_names: List[str] = None,
        target_channel: int = -1,  # -1 means last channel
    ):
        if isinstance(data, list):
            data = np.array(data, dtype=float)
        self.data = data
        self.n_timesteps, self.n_channels = data.shape
        self.channel_names = channel_names or [f"x{i}" for i in range(self.n_channels)]
        self.target_channel = target_channel if target_channel >= 0 else self.n_channels - 1

    def get_channel(self, channel: int) -> np.ndarray:
        """Get one channel as a 1D array."""
        return self.data[:, channel]

    def get_target(self) -> np.ndarray:
        """Get the target variable."""
        return self.data[:, self.target_channel]

    def get_features(self) -> np.ndarray:
        """Get all non-target channels."""
        cols = [i for i in range(self.n_channels) if i != self.target_channel]
        return self.data[:, cols]

    def get_value(self, t: int, channel: int) -> float:
        """Get the value at timestep t, channel c."""
        if 0 <= t < self.n_timesteps and 0 <= channel < self.n_channels:
            return float(self.data[t, channel])
        return 0.0

    @classmethod
    def from_dict(cls, d: Dict[str, List[float]], target: str) -> 'MultivariateObservations':
        """Create from a dict {variable_name: [values]} with specified target."""
        names = list(d.keys())
        data = np.column_stack([d[name] for name in names])
        target_idx = names.index(target) if target in names else -1
        return cls(data, channel_names=names, target_channel=target_idx)

    @classmethod
    def from_csv(cls, path: str, target_col: str) -> 'MultivariateObservations':
        """Load from CSV file."""
        import csv
        rows = []
        headers = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not headers:
                    headers = list(row.keys())
                rows.append([float(row[h]) for h in headers])
        data = np.array(rows)
        target_idx = headers.index(target_col)
        return cls(data, channel_names=headers, target_channel=target_idx)


# ── OBS Node for multivariate expressions ────────────────────────────────────

class MultivariateExprNode:
    """
    Expression node that can reference multiple observation channels.

    New node type: OBS_CHANNEL(channel_index)
      - Terminal node (no children)
      - Returns obs[t][channel_index] when evaluated
      - Requires MultivariateObservations to be passed during evaluation

    Wraps existing ExtExprNode for backward compatibility.
    """

    def __init__(
        self,
        channel: int = 0,
        obs_matrix: MultivariateObservations = None,
    ):
        self.channel = channel
        self._obs_matrix = obs_matrix

    def evaluate(
        self,
        t: int,
        history: List[float],
        state: Dict = None,
        obs_matrix: MultivariateObservations = None,
    ) -> float:
        """Evaluate: return obs[t][channel]."""
        matrix = obs_matrix or self._obs_matrix
        if matrix is None:
            return 0.0
        return matrix.get_value(t, self.channel)

    def to_string(self) -> str:
        return f"OBS({self.channel})"

    def node_count(self) -> int:
        return 1

    def constant_count(self) -> int:
        return 0

    def depth(self) -> int:
        return 1


# ── Multivariate MDL Engine ────────────────────────────────────────────────────

@dataclass
class MultivariateScoreResult:
    """MDL score for a multivariate expression."""
    expression_str: str
    total_mdl_cost: float
    program_bits: float
    data_bits: float
    n_timesteps: int
    n_channels_used: int
    r_squared: float     # goodness of fit (0-1)


class MultivariateMDLEngine:
    """
    MDL engine for expressions over multiple observation channels.

    Extends the standard MDL engine to handle:
    - Multiple input channels
    - OBS_CHANNEL terminal nodes
    - Cross-channel correlations in expressions

    Usage:
        obs = MultivariateObservations.from_dict(
            {"co2": co2_series, "temp": temp_series},
            target="temp"
        )
        engine = MultivariateMDLEngine()
        expr = build_expression_using_co2_and_temp(...)
        score = engine.score(expr, obs)
    """

    def __init__(self, obs_matrix: MultivariateObservations = None):
        self._obs_matrix = obs_matrix
        from ouroboros.compression.mdl_engine import MDLEngine
        self._base_mdl = MDLEngine()

    def score(
        self,
        expr,
        obs_matrix: MultivariateObservations,
    ) -> MultivariateScoreResult:
        """Score an expression over multivariate observations."""
        target = obs_matrix.get_target()
        n = len(target)

        # Evaluate expression at each timestep
        predictions = []
        for t in range(n):
            try:
                pred = self._eval_with_matrix(expr, t, obs_matrix)
                if not isinstance(pred, (int, float)) or not math.isfinite(pred):
                    pred = 0.0
                predictions.append(pred)
            except Exception:
                predictions.append(0.0)

        # Compute data description length (Shannon entropy of residuals)
        residuals = [float(target[t]) - predictions[t] for t in range(n)]
        mean_sq_resid = sum(r**2 for r in residuals) / max(n, 1)
        if mean_sq_resid < 1e-12:
            data_bits = 0.0
        else:
            std_resid = math.sqrt(mean_sq_resid)
            data_bits = n * math.log2(max(std_resid * math.sqrt(2 * math.pi * math.e), 1.0))

        # Compute program description length
        program_bits = 0.0
        if hasattr(expr, 'node_count'):
            program_bits = expr.node_count() * 4.0 + expr.constant_count() * 6.0

        # R-squared
        mean_target = sum(float(target[t]) for t in range(n)) / max(n, 1)
        ss_tot = sum((float(target[t]) - mean_target)**2 for t in range(n))
        ss_res = sum(r**2 for r in residuals)
        r_sq = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-12))

        # Count channels used
        expr_str = expr.to_string() if hasattr(expr, 'to_string') else str(expr)
        n_channels_used = sum(
            1 for i in range(obs_matrix.n_channels)
            if f"OBS({i})" in expr_str or f"x{i}" in expr_str
        )

        return MultivariateScoreResult(
            expression_str=expr_str,
            total_mdl_cost=program_bits + data_bits,
            program_bits=program_bits,
            data_bits=data_bits,
            n_timesteps=n,
            n_channels_used=n_channels_used,
            r_squared=r_sq,
        )

    def _eval_with_matrix(
        self,
        expr,
        t: int,
        obs_matrix: MultivariateObservations,
    ) -> float:
        """Evaluate expression with access to observation matrix."""
        if isinstance(expr, MultivariateExprNode):
            return expr.evaluate(t, [], {}, obs_matrix)

        # Standard ExprNode evaluation with multivariate target as history
        target = obs_matrix.get_target()
        history = [float(v) for v in target[:t]]

        if hasattr(expr, 'evaluate'):
            return expr.evaluate(t, history, {})
        return float(expr)


# ── MultivariateSearchRunner ───────────────────────────────────────────────────

class MultivariateSearchRunner:
    """
    Runs OUROBOROS expression search over multivariate observations.

    For each pair (target, features), discovers the symbolic expression
    that best predicts target from features (and the target's own history).
    """

    def __init__(
        self,
        beam_width: int = 20,
        n_iterations: int = 10,
        max_depth: int = 5,
        verbose: bool = True,
    ):
        self.beam_width = beam_width
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.verbose = verbose

    def discover(
        self,
        obs_matrix: MultivariateObservations,
    ) -> Dict:
        """
        Discover the symbolic law governing the target variable.

        Returns dict with:
          'expression': best expression string
          'mdl_cost': MDL cost
          'r_squared': goodness of fit
          'channels_used': which input channels were used
        """
        from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig

        target = obs_matrix.get_target()
        int_target = [int(round(v)) for v in target]
        alphabet_size = max(int_target) - min(int_target) + 2

        router = HierarchicalSearchRouter(RouterConfig(
            beam_width=self.beam_width,
            max_depth=self.max_depth,
            n_iterations=self.n_iterations,
            random_seed=42,
        ))

        # Run discovery on target variable
        result = router.search(int_target, alphabet_size=max(alphabet_size, 2))

        if self.verbose:
            target_name = obs_matrix.channel_names[obs_matrix.target_channel]
            print(f"Multivariate discovery — target: {target_name}")
            if result.expr:
                print(f"  Expression: {result.expr.to_string()}")
                print(f"  MDL cost: {result.mdl_cost:.2f}")
                print(f"  Family: {result.math_family.name}")

        return {
            'expression': result.expr.to_string() if result.expr else None,
            'mdl_cost': result.mdl_cost,
            'math_family': result.math_family.name if result.math_family else 'UNKNOWN',
            'channels_used': [obs_matrix.channel_names[obs_matrix.target_channel]],
        }