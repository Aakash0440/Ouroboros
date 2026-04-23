"""Tests for communication experiment and statistical analysis."""
import pytest
import statistics
from ouroboros.experiments.communication_experiment import (
    AgentState, ExperimentRun, _compute_herding_index,
)
from ouroboros.experiments.statistical_tests import (
    mann_whitney_u, bootstrap_ci, EffectSizeResult, StatisticalTester,
    MannWhitneyResult, BootstrapCI,
)


class TestHerdingIndex:
    def _make_state(self, centroid):
        return AgentState("A0", "const(1)", 50.0, None, centroid)

    def test_identical_agents_herd_100pct(self):
        states = [self._make_state([1,2,3,4,5]) for _ in range(5)]
        h = _compute_herding_index(states)
        assert h == pytest.approx(1.0)

    def test_all_different_herd_0pct(self):
        states = [self._make_state([i, i*2, i*3]) for i in range(1, 6)]
        h = _compute_herding_index(states)
        assert h == pytest.approx(0.0)

    def test_half_same_half_different(self):
        same = [self._make_state([1, 2, 3]) for _ in range(3)]
        diff = [self._make_state([i*10, i*20, i*30]) for i in range(1, 3)]
        h = _compute_herding_index(same + diff)
        assert 0.0 < h < 1.0

    def test_single_agent_returns_zero(self):
        states = [self._make_state([1, 2, 3])]
        h = _compute_herding_index(states)
        assert h == 0.0

    def test_none_centroids_excluded(self):
        states = [
            self._make_state(None),
            self._make_state(None),
        ]
        h = _compute_herding_index(states)
        assert h == 0.0


class TestMannWhitneyU:
    def test_identical_distributions(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = mann_whitney_u(x, x)
        assert result.p_value > 0.05  # not significant

    def test_clearly_different_distributions(self):
        x = [1.0, 2.0, 3.0]
        y = [100.0, 200.0, 300.0]
        result = mann_whitney_u(x, y)
        assert result.u_statistic < x.__len__() * y.__len__() / 2

    def test_empty_returns_p1(self):
        result = mann_whitney_u([], [1.0, 2.0])
        assert result.p_value == 1.0

    def test_n_values_correct(self):
        result = mann_whitney_u([1.0, 2.0, 3.0], [4.0, 5.0])
        assert result.n1 == 3
        assert result.n2 == 2

    def test_significant_flag(self):
        # Clearly different distributions with enough samples should be significant
        x = list(range(1, 11))
        y = list(range(100, 110))
        result = mann_whitney_u(x, y)
        assert result.significant  # p < 0.05


class TestBootstrapCI:
    def test_ci_contains_zero_for_same_distributions(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 2.0, 3.0, 4.0, 5.0]
        ci = bootstrap_ci(x, y, n_bootstrap=500, seed=42)
        assert ci.includes_zero

    def test_ci_excludes_zero_for_different(self):
        x = [1.0, 1.5, 2.0, 1.2, 1.8]
        y = [10.0, 10.5, 11.0, 10.2, 10.8]
        ci = bootstrap_ci(x, y, n_bootstrap=1000, seed=42)
        assert not ci.includes_zero

    def test_observed_diff_correct(self):
        x = [5.0, 5.0]
        y = [3.0, 3.0]
        ci = bootstrap_ci(x, y, n_bootstrap=100)
        assert ci.observed_diff == pytest.approx(2.0)

    def test_n_bootstrap_recorded(self):
        ci = bootstrap_ci([1.0], [2.0], n_bootstrap=777)
        assert ci.n_bootstrap == 777


class TestEffectSize:
    def test_zero_effect(self):
        r = EffectSizeResult.from_groups([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert r.cohens_d == pytest.approx(0.0, abs=0.01)

    def test_large_effect(self):
        x = [1.0, 1.2, 0.8, 1.1]
        y = [10.0, 10.2, 9.8, 10.1]
        r = EffectSizeResult.from_groups(x, y)
        assert r.cohens_d > 0.8
        assert r.interpretation == "large"

    def test_small_effect(self):
        x = [0.0, 0.3, 0.1, 0.2]
        y = [0.2, 0.5, 0.3, 0.4]
        r = EffectSizeResult.from_groups(x, y)
        assert r.interpretation in ("negligible", "small")

    def test_insufficient_data(self):
        r = EffectSizeResult.from_groups([1.0], [2.0])
        assert "insufficient" in r.interpretation


class TestStatisticalTester:
    def _make_run(self, condition, seed, consensus, cost, herding, unique):
        from ouroboros.experiments.communication_experiment import AgentState
        state = AgentState("A0", "expr", cost, consensus, None)
        return ExperimentRun(
            condition=condition, seed=seed, n_rounds=10,
            agent_states=[state],
            rounds_to_consensus=consensus,
            final_best_cost=cost,
            herding_index=herding,
            unique_expressions=unique,
        )

    def test_analysis_returns_result(self):
        solo_runs = [self._make_run("SOLO", i, i+5, 50.0-i, 0.1, 3) for i in range(5)]
        comm_runs = [self._make_run("COMM", i, i+3, 48.0-i, 0.4, 2) for i in range(5)]
        tester = StatisticalTester()
        result = tester.analyze(solo_runs, comm_runs, n_rounds=10)
        assert result.n_solo_runs == 5
        assert result.n_comm_runs == 5
        assert len(result.solo_consensus_rounds) == 5

    def test_summary_str_is_string(self):
        solo_runs = [self._make_run("SOLO", i, None, 50.0, 0.1, 3) for i in range(3)]
        comm_runs = [self._make_run("COMM", i, 5, 45.0, 0.3, 2) for i in range(3)]
        tester = StatisticalTester()
        result = tester.analyze(solo_runs, comm_runs, n_rounds=10)
        s = result.summary()
        assert isinstance(s, str) and len(s) > 100
        assert "Mann-Whitney" in s
        assert "CONCLUSION" in s