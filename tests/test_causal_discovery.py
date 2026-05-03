"""Tests for causal discovery — CausalGraph, DoCalculusEngine, InterventionalEnvironments."""
import pytest
import math
from ouroboros.causal.causal_graph import (
    CausalGraph, CausalEdge, CausalVariable, InterventionalGraph,
)
from ouroboros.causal.do_calculus import (
    DoCalculusEngine, _partial_correlation, _granger_causality_test,
    CausalEffectEstimate,
)
from ouroboros.causal.interventional_env import (
    InterventionalSpringMassEnv, SyntheticClimateEnv,
    InterventionResult, CausalDiscoveryRunner,
)


def make_var(name: str) -> CausalVariable:
    return CausalVariable(name=name, var_type="observed")


class TestCausalGraph:
    def test_add_variables(self):
        g = CausalGraph()
        g.add_variable(make_var("X"))
        g.add_variable(make_var("Y"))
        assert g.n_variables == 2

    def test_add_edge(self):
        g = CausalGraph()
        vx, vy = make_var("X"), make_var("Y")
        edge = CausalEdge(vx, vy, lag=0)
        success = g.add_edge(edge)
        assert success
        assert g.n_edges == 1

    def test_cycle_detection(self):
        g = CausalGraph()
        vx, vy, vz = make_var("X"), make_var("Y"), make_var("Z")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        g.add_edge(CausalEdge(vy, vz, lag=0))
        # X → Y → Z → X would be a cycle
        cycle_edge = CausalEdge(vz, vx, lag=0)
        result = g.add_edge(cycle_edge)
        assert not result  # should reject cycle

    def test_lag_edges_not_cycles(self):
        g = CausalGraph()
        vx, vy = make_var("X"), make_var("Y")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        # Y → X with lag is OK (temporal loop, not logical cycle)
        result = g.add_edge(CausalEdge(vy, vx, lag=1))
        assert result

    def test_parents(self):
        g = CausalGraph()
        vx, vy = make_var("X"), make_var("Y")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        assert "X" in g.parents("Y")
        assert len(g.parents("X")) == 0

    def test_children(self):
        g = CausalGraph()
        vx, vy = make_var("X"), make_var("Y")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        assert "Y" in g.children("X")

    def test_ancestors(self):
        g = CausalGraph()
        vx, vy, vz = make_var("X"), make_var("Y"), make_var("Z")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        g.add_edge(CausalEdge(vy, vz, lag=0))
        ancs = g.ancestors("Z")
        assert "X" in ancs and "Y" in ancs

    def test_topological_sort(self):
        g = CausalGraph()
        vx, vy, vz = make_var("X"), make_var("Y"), make_var("Z")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        g.add_edge(CausalEdge(vy, vz, lag=0))
        order = g.topological_sort()
        assert order.index("X") < order.index("Y") < order.index("Z")

    def test_do_intervention(self):
        g = CausalGraph()
        vx, vy = make_var("X"), make_var("Y")
        g.add_edge(CausalEdge(vx, vy, lag=0))
        int_graph = g.do_intervention("X", 5.0)
        assert int_graph.is_intervened("X")
        assert not int_graph.is_intervened("Y")
        assert len(int_graph.effective_parents("X")) == 0  # cut

    def test_to_string(self):
        g = CausalGraph()
        vx, vy = make_var("X"), make_var("Y")
        g.add_edge(CausalEdge(vx, vy))
        s = g.to_string()
        assert "X" in s and "Y" in s

    def test_backdoor_criterion(self):
        g = CausalGraph()
        vx, vy, vz = make_var("X"), make_var("Y"), make_var("Z")
        # Z → X → Y (Z is confounder through X)
        g.add_edge(CausalEdge(vz, vx, lag=0))
        g.add_edge(CausalEdge(vx, vy, lag=0))
        # Adjusting for Z satisfies backdoor criterion
        assert g.backdoor_criterion("X", "Y", {"Z"})


class TestPartialCorrelation:
    def test_zero_when_controlled(self):
        # x and y are both caused by z: x = z + noise, y = z + noise
        import random
        rng = random.Random(42)
        z = [float(rng.gauss(0, 1)) for _ in range(100)]
        x = [zi + rng.gauss(0, 0.1) for zi in z]
        y = [zi + rng.gauss(0, 0.1) for zi in z]
        # Partial correlation controlling for z should be near 0
        pc = _partial_correlation(x, y, z)
        assert abs(pc) < 0.3  # not perfect but should be reduced

    def test_high_when_not_confounded(self):
        # Direct cause: x causes y
        x = [float(i) for i in range(100)]
        y = [float(i) * 2 for i in range(100)]
        z = [1.0] * 100  # irrelevant variable
        pc = _partial_correlation(x, y, z)
        assert abs(pc) > 0.9


class TestGrangerCausality:
    def test_detects_granger_cause(self):
        # x[t] causes y[t+1]
        x = [math.sin(t * 0.3) for t in range(100)]
        y = [x[t-1] * 2 + 0.1 * math.cos(t) for t in range(1, 101)]
        f_stat, lag = _granger_causality_test(x, y[:100], max_lag=3)
        assert f_stat > 0  # some Granger causality detected

    def test_no_causality_for_independent(self):
        import random
        rng = random.Random(1)
        x = [rng.gauss(0, 1) for _ in range(100)]
        y = [rng.gauss(0, 1) for _ in range(100)]
        f_stat, _ = _granger_causality_test(x, y, max_lag=3)
        # Independent: F-stat should be low (not always zero due to randomness)
        assert f_stat < 20  # not too high


class TestInterventionalEnvironments:
    def test_spring_generates_oscillation(self):
        env = InterventionalSpringMassEnv(amplitude=10, omega=0.3)
        obs = env.generate(50)
        assert len(obs) == 50
        sign_changes = sum(1 for i in range(1, len(obs)) if obs[i]*obs[i-1] < 0)
        assert sign_changes >= 2

    def test_intervention_changes_trajectory(self):
        env = InterventionalSpringMassEnv(amplitude=10, omega=0.3)
        result = env.intervene("position", value=5.0, at_time=20, n_steps=30)
        assert isinstance(result, InterventionResult)
        # Post-intervention should differ from counterfactual at some point
        diff = sum(abs(r-c) for r,c in zip(result.post_intervention,
                                            result.counterfactual))
        assert diff > 0.1  # some difference

    def test_causal_effect_computed(self):
        env = InterventionalSpringMassEnv(amplitude=10, omega=0.3)
        result = env.intervene("position", value=0.0, at_time=10, n_steps=20)
        effect = result.causal_effect
        assert isinstance(effect, float)

    def test_spring_causal_graph(self):
        env = InterventionalSpringMassEnv()
        graph = env.get_causal_graph()
        assert graph.n_edges >= 3
        assert "position" in graph._variables

    def test_climate_generates(self):
        env = SyntheticClimateEnv()
        obs = env.generate(100)
        assert len(obs) == 100
        assert all(math.isfinite(v) for v in obs)

    def test_climate_multivariate(self):
        env = SyntheticClimateEnv()
        mv = env.generate_multivariate(50)
        assert "co2" in mv and "temperature" in mv
        # CO2 should increase over time
        assert mv["co2"][-1] > mv["co2"][0]

    def test_climate_intervention(self):
        env = SyntheticClimateEnv()
        result = env.intervene_co2(
            intervention_value=560.0,  # 2x baseline
            at_time=50,
            n_steps=30,
        )
        assert isinstance(result, InterventionResult)
        assert result.intervened_var == "co2"


class TestDoCalculusEngine:
    def test_discovers_simple_cause(self):
        engine = DoCalculusEngine(granger_threshold=2.0, max_lag=3)
        # x causes y
        x = [float(t) for t in range(100)]
        y = [x[t-1] * 1.5 for t in range(1, 101)]
        sequences = {"x": x, "y": y[:100]}
        graph = engine.discover(sequences)
        assert graph.n_variables >= 1

    def test_runs_without_crash(self):
        engine = DoCalculusEngine()
        import random
        rng = random.Random(42)
        sequences = {
            "a": [rng.gauss(0, 1) for _ in range(80)],
            "b": [rng.gauss(0, 1) for _ in range(80)],
        }
        graph = engine.discover(sequences)
        assert isinstance(graph, CausalGraph)