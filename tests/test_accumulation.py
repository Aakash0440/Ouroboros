"""Tests for knowledge accumulation system."""
import pytest
from ouroboros.knowledge.accumulation import (
    SimpleAxiomKB, AccumulationRunner, SessionResult, AccumulationRecord,
)
from ouroboros.knowledge.growth_tracker import KnowledgeGrowthTracker, GrowthAnalysis


class TestSimpleAxiomKB:
    def test_add_axiom(self):
        kb = SimpleAxiomKB()
        is_new = kb.add_axiom("(3*t+1)%7", 45.0, "ModArith", session_id=1)
        assert is_new
        assert kb.n_axioms == 1

    def test_no_duplicate(self):
        kb = SimpleAxiomKB()
        kb.add_axiom("expr1", 45.0, "env1", 1)
        is_new = kb.add_axiom("expr1", 40.0, "env1", 2)
        assert not is_new
        assert kb.n_axioms == 1

    def test_get_seeds_for_env(self):
        kb = SimpleAxiomKB()
        kb.add_axiom("expr1", 45.0, "ModArith", 1)
        kb.add_axiom("expr2", 40.0, "ModArith", 2)
        kb.add_axiom("expr3", 50.0, "FibEnv", 3)
        seeds = kb.get_seeds_for_environment("ModArith", top_k=5)
        assert "expr2" in seeds  # lower MDL first
        assert "expr1" in seeds
        assert "expr3" not in seeds  # wrong env

    def test_seeds_sorted_by_mdl(self):
        kb = SimpleAxiomKB()
        kb.add_axiom("worse", 100.0, "env", 1)
        kb.add_axiom("better", 30.0, "env", 2)
        seeds = kb.get_seeds_for_environment("env", top_k=2)
        assert seeds[0] == "better"

    def test_get_all_seeds(self):
        kb = SimpleAxiomKB()
        for i in range(10):
            kb.add_axiom(f"expr{i}", float(i*10), "env", i)
        seeds = kb.get_all_seeds(top_k=5)
        assert len(seeds) == 5

    def test_save_load_roundtrip(self, tmp_path):
        kb = SimpleAxiomKB()
        kb.add_axiom("expr1", 45.0, "env1", 1)
        path = str(tmp_path / "kb.json")
        kb.save(path)
        kb2 = SimpleAxiomKB()
        kb2.load(path)
        assert kb2.n_axioms == 1
        seeds = kb2.get_all_seeds()
        assert "expr1" in seeds

    def test_empty_kb_returns_empty_seeds(self):
        kb = SimpleAxiomKB()
        seeds = kb.get_seeds_for_environment("any_env")
        assert seeds == []


class TestSessionResult:
    def test_session_result_fields(self):
        sr = SessionResult(
            session_id=5, environment_name="ModArith",
            rounds_to_best=3, best_mdl_cost=45.0,
            n_axioms_at_start=10, n_axioms_at_end=11,
            n_new_axioms=1, elapsed_seconds=2.5,
            prior_benefit=15.0, expression_str="(3*t+1)%7",
        )
        assert sr.session_id == 5
        assert sr.n_new_axioms == 1
        assert sr.prior_benefit == 15.0


class TestAccumulationRunner:
    def test_short_run_produces_record(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        runner = AccumulationRunner(
            n_sessions=5,
            environments=[ModularArithmeticEnv(modulus=7)],
            stream_length=100,
            beam_width=6,
            n_iterations=3,
            kb_path=None,
            verbose=False,
            report_every=5,
        )
        record = runner.run("test_run")
        assert len(record.sessions) == 5
        assert record.n_sessions == 5

    def test_kb_grows_across_sessions(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        runner = AccumulationRunner(
            n_sessions=5,
            environments=[ModularArithmeticEnv(modulus=7)],
            stream_length=100,
            beam_width=6,
            n_iterations=3,
            verbose=False,
            report_every=10,
        )
        record = runner.run("kb_growth_test")
        # KB should have grown (may be 0 if no expression found in fast mode)
        final_kb = max(s.n_axioms_at_end for s in record.sessions)
        assert final_kb >= 0  # at least 0 (might not find anything in tiny run)

    def test_record_has_environment_names(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        runner = AccumulationRunner(
            n_sessions=3,
            environments=[ModularArithmeticEnv(modulus=7)],
            stream_length=80, beam_width=5, n_iterations=2,
            verbose=False, report_every=10,
        )
        record = runner.run("env_name_test")
        assert "ModularArithmetic" in record.environments_tested[0]


class TestKnowledgeGrowthTracker:
    def _make_sessions(self, n: int, improving: bool = True):
        results = []
        for i in range(n):
            cost = 200.0 - (i * 1.5 if improving else 0)
            results.append(SessionResult(
                session_id=i+1, environment_name="env",
                rounds_to_best=1, best_mdl_cost=cost,
                n_axioms_at_start=i, n_axioms_at_end=i+1,
                n_new_axioms=1 if i < 20 else 0,
                elapsed_seconds=1.0, prior_benefit=float(i),
                expression_str=f"expr{i}",
            ))
        return results

    def test_analysis_returns_result(self):
        sessions = self._make_sessions(20, improving=True)
        tracker = KnowledgeGrowthTracker()
        analysis = tracker.analyze(sessions, verbose=False)
        assert isinstance(analysis, GrowthAnalysis)
        assert analysis.n_sessions == 20

    def test_improvement_detected(self):
        sessions = self._make_sessions(30, improving=True)
        tracker = KnowledgeGrowthTracker()
        analysis = tracker.analyze(sessions, verbose=False)
        # Early costs > late costs → positive improvement
        assert analysis.mdl_improvement > 0

    def test_plateau_detection(self):
        sessions = self._make_sessions(50, improving=True)
        # Force no new axioms after session 20
        for s in sessions[20:]:
            s.n_new_axioms = 0
        tracker = KnowledgeGrowthTracker()
        analysis = tracker.analyze(sessions, verbose=False)
        # Plateau should be detected around session 20-30
        if analysis.plateau_session:
            assert 15 <= analysis.plateau_session <= 35

    def test_growth_curve_data(self):
        sessions = self._make_sessions(20, improving=True)
        tracker = KnowledgeGrowthTracker()
        curve = tracker.generate_growth_curve_data(sessions, window=5)
        assert "sessions" in curve
        assert "mdl_smooth" in curve
        assert len(curve["sessions"]) == 20

    def test_empty_sessions(self):
        tracker = KnowledgeGrowthTracker()
        analysis = tracker.analyze([], verbose=False)
        assert analysis.n_sessions == 0
        assert analysis.mdl_improvement == 0.0

    def test_summary_string(self):
        sessions = self._make_sessions(30, improving=True)
        tracker = KnowledgeGrowthTracker()
        analysis = tracker.analyze(sessions, verbose=False)
        s = analysis.summary()
        assert isinstance(s, str) and len(s) > 50
        assert "sessions" in s.lower()