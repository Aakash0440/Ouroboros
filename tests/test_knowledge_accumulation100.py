"""Tests for 100-session knowledge accumulation experiment."""
import pytest
import math
from ouroboros.knowledge.experiment100 import (
    ExperimentPhase, GrowthModelFit, Experiment100Result,
    fit_logarithmic_growth, KnowledgeAccumulationExperiment100,
)
from ouroboros.knowledge.accumulation import SessionResult


def make_session(i: int, cost: float = 100.0, benefit: float = 5.0, n_axioms: int = 5) -> SessionResult:
    return SessionResult(
        session_id=i, environment_name="test",
        rounds_to_best=3, best_mdl_cost=cost,
        n_axioms_at_start=max(0, n_axioms-1), n_axioms_at_end=n_axioms,
        n_new_axioms=1, elapsed_seconds=1.0,
        prior_benefit=benefit, expression_str=None,
    )


class TestFitLogarithmicGrowth:
    def test_logarithmic_fit_perfect(self):
        import math
        a, b = 3.0, 1.0
        times = list(range(1, 21))
        values = [a * math.log(t) + b for t in times]
        result = fit_logarithmic_growth(times, values)
        assert result.r_squared > 0.99
        assert abs(result.a - a) < 0.1
        assert abs(result.b - b) < 0.1

    def test_fit_returns_model(self):
        times = [1, 2, 3, 4, 5]
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        result = fit_logarithmic_growth(times, values)
        assert isinstance(result, GrowthModelFit)
        assert result.model_type == "logarithmic"

    def test_predict_nonnegative(self):
        result = fit_logarithmic_growth([1,2,3], [1.0, 2.0, 3.0])
        for t in range(1, 20):
            assert result.predict(t) >= 0

    def test_too_short_returns_default(self):
        result = fit_logarithmic_growth([1, 2], [1.0, 2.0])
        assert isinstance(result, GrowthModelFit)

    def test_plateau_detected_flat_growth(self):
        # Very flat: growth rate near 0 early
        times = list(range(1, 21))
        values = [1.0] * 20  # constant — no growth
        result = fit_logarithmic_growth(times, values)
        # With flat data, a ≈ 0, so derivative ≈ 0 everywhere → early plateau
        if result.plateau_session:
            assert result.plateau_session >= 1


class TestExperimentPhase:
    def test_mean_mdl(self):
        sessions = [make_session(i, cost=float(100 - i*2)) for i in range(10)]
        phase = ExperimentPhase("early", (1, 10), sessions)
        assert 80 <= phase.mean_mdl <= 100

    def test_mean_axioms(self):
        sessions = [make_session(i, n_axioms=i+1) for i in range(5)]
        phase = ExperimentPhase("test", (1, 5), sessions)
        assert phase.mean_axioms > 0

    def test_n_sessions(self):
        sessions = [make_session(i) for i in range(7)]
        phase = ExperimentPhase("test", (1, 7), sessions)
        assert phase.n_sessions == 7


class TestGrowthModelFit:
    def test_logarithmic_predict(self):
        model = GrowthModelFit("logarithmic", a=2.0, b=1.0, r_squared=0.95, plateau_session=None)
        assert model.predict(1) == pytest.approx(1.0)  # 2*log(1)+1 = 1
        assert model.predict(10) > model.predict(1)

    def test_plateau_predict(self):
        model = GrowthModelFit("plateau", a=0.0, b=10.0, r_squared=0.8, plateau_session=20)
        assert model.predict(5) == pytest.approx(10.0)
        assert model.predict(100) == pytest.approx(10.0)

    def test_description_is_string(self):
        model = GrowthModelFit("logarithmic", 2.0, 1.0, 0.95, 30)
        s = model.description()
        assert isinstance(s, str) and "log" in s.lower()


class TestExperiment100Result:
    def _make_result(self) -> Experiment100Result:
        early = ExperimentPhase("early", (1, 10), [make_session(i, 200-i) for i in range(10)])
        late = ExperimentPhase("late", (90, 100), [make_session(i, 150-i) for i in range(10)])
        model = GrowthModelFit("logarithmic", 3.0, 1.0, 0.92, None)
        return Experiment100Result(
            n_sessions=30, total_axioms_final=15,
            phases={"early": early, "late": late},
            growth_model=model,
            early_mean_mdl=early.mean_mdl,
            late_mean_mdl=late.mean_mdl,
            mdl_improvement=early.mean_mdl - late.mean_mdl,
            improvement_pct=5.0,
            early_mean_benefit=2.0, late_mean_benefit=8.0,
            benefit_growth=6.0,
            knowledge_accumulated=True,
        )

    def test_summary_is_string(self):
        result = self._make_result()
        s = result.summary()
        assert isinstance(s, str) and len(s) > 100

    def test_latex_section_has_numbers(self):
        result = self._make_result()
        latex = result.latex_section()
        assert "axiom" in latex.lower()
        assert "30" in latex  # n_sessions

    def test_knowledge_accumulated_field(self):
        result = self._make_result()
        assert isinstance(result.knowledge_accumulated, bool)


class TestKnowledgeAccumulationExperiment100:
    def test_tiny_run_produces_result(self):
        exp = KnowledgeAccumulationExperiment100(
            n_sessions=5, stream_length=80,
            beam_width=5, n_iterations=2,
            verbose=False, report_every=10,
        )
        result = exp.run()
        assert isinstance(result, Experiment100Result)
        assert result.n_sessions == 5

    def test_growth_model_fitted(self):
        exp = KnowledgeAccumulationExperiment100(
            n_sessions=8, stream_length=80,
            beam_width=5, n_iterations=2,
            verbose=False, report_every=10,
        )
        result = exp.run()
        assert isinstance(result.growth_model, GrowthModelFit)
        assert result.growth_model.model_type == "logarithmic"

    def test_phases_computed(self):
        exp = KnowledgeAccumulationExperiment100(
            n_sessions=8, stream_length=80,
            beam_width=5, n_iterations=2,
            verbose=False, report_every=10,
        )
        result = exp.run()
        assert len(result.phases) >= 2  # at least early and late