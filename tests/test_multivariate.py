"""Tests for multivariate MDL engine."""
import pytest
import math
import numpy as np
from ouroboros.multivariate.mdl_engine import (
    MultivariateObservations, MultivariateExprNode,
    MultivariateMDLEngine, MultivariateScoreResult,
)


class TestMultivariateObservations:
    def test_from_dict(self):
        obs = MultivariateObservations.from_dict(
            {"co2": [1.0, 2.0, 3.0], "temp": [14.0, 14.1, 14.2]},
            target="temp"
        )
        assert obs.n_channels == 2
        assert obs.n_timesteps == 3
        assert obs.target_channel == 1

    def test_get_channel(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        obs = MultivariateObservations(data, target_channel=1)
        ch0 = obs.get_channel(0)
        assert list(ch0) == [1.0, 2.0, 3.0]

    def test_get_target(self):
        data = np.array([[1.0, 10.0], [2.0, 20.0]])
        obs = MultivariateObservations(data, target_channel=1)
        target = obs.get_target()
        assert list(target) == [10.0, 20.0]

    def test_get_value(self):
        data = np.array([[1.0, 5.0], [2.0, 6.0]])
        obs = MultivariateObservations(data)
        assert obs.get_value(0, 0) == 1.0
        assert obs.get_value(1, 1) == 6.0

    def test_out_of_bounds_returns_zero(self):
        data = np.array([[1.0, 5.0]])
        obs = MultivariateObservations(data)
        assert obs.get_value(100, 0) == 0.0  # out of bounds

    def test_channel_names(self):
        obs = MultivariateObservations.from_dict(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]}, target="b"
        )
        assert obs.channel_names == ["a", "b"]

    def test_default_target_is_last(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        obs = MultivariateObservations(data)
        assert obs.target_channel == 2  # last channel

    def test_get_features_excludes_target(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        obs = MultivariateObservations(data, target_channel=2)
        features = obs.get_features()
        assert features.shape[1] == 2  # 2 feature channels


class TestMultivariateExprNode:
    def test_basic_evaluation(self):
        data = np.array([[10.0, 20.0], [11.0, 21.0]])
        obs = MultivariateObservations(data)
        node = MultivariateExprNode(channel=0, obs_matrix=obs)
        val = node.evaluate(0, [])
        assert val == 10.0

    def test_second_channel(self):
        data = np.array([[10.0, 20.0], [11.0, 21.0]])
        obs = MultivariateObservations(data)
        node = MultivariateExprNode(channel=1, obs_matrix=obs)
        val = node.evaluate(0, [])
        assert val == 20.0

    def test_to_string(self):
        node = MultivariateExprNode(channel=2)
        assert node.to_string() == "OBS(2)"

    def test_node_count(self):
        node = MultivariateExprNode(channel=0)
        assert node.node_count() == 1

    def test_no_obs_matrix_returns_zero(self):
        node = MultivariateExprNode(channel=0)
        val = node.evaluate(0, [])
        assert val == 0.0


class TestMultivariateMDLEngine:
    def test_score_obs_node(self):
        data = np.array([[float(t), float(t*2)] for t in range(50)])
        obs = MultivariateObservations(data, target_channel=1)
        engine = MultivariateMDLEngine()
        # OBS(0) * 2 should perfectly predict target (which is 2*x0)
        node = MultivariateExprNode(channel=0, obs_matrix=obs)
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        score = engine.score(node, obs)
        assert isinstance(score, MultivariateScoreResult)
        assert math.isfinite(score.total_mdl_cost)

    def test_r_squared_perfect_prediction(self):
        # When expression perfectly predicts target, R² ≈ 1
        data = np.array([[float(t), float(t)] for t in range(50)])
        obs = MultivariateObservations(data, target_channel=1)
        engine = MultivariateMDLEngine()
        node = MultivariateExprNode(channel=0, obs_matrix=obs)
        score = engine.score(node, obs)
        assert score.r_squared > 0.9  # should be near 1

    def test_score_result_fields(self):
        data = np.array([[1.0, 2.0] for _ in range(20)])
        obs = MultivariateObservations(data, target_channel=1)
        engine = MultivariateMDLEngine()
        node = MultivariateExprNode(channel=0, obs_matrix=obs)
        score = engine.score(node, obs)
        assert score.n_timesteps == 20
        assert 0.0 <= score.r_squared <= 1.0

    def test_from_dict_workflow(self):
        import math
        obs = MultivariateObservations.from_dict({
            "time": [float(t) for t in range(50)],
            "sin_signal": [math.sin(t * 0.3) for t in range(50)],
        }, target="sin_signal")
        assert obs.n_channels == 2
        assert obs.n_timesteps == 50