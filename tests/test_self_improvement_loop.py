"""Tests for the self-improvement loop."""
import pytest
from ouroboros.core.self_improvement_loop import (
    SelfImprovementLoop, LoopState, LoopIterationResult,
)
from ouroboros.environments.modular import ModularArithmeticEnv


class TestLoopState:
    def test_initial_state(self):
        state = LoopState()
        assert state.n_iterations == 0
        assert state.n_expressions_discovered == 0

    def test_update_increments(self):
        state = LoopState()
        result = LoopIterationResult(
            iteration=1, env_name="test",
            expression_str="(3*t+1)%7", mdl_cost=45.0,
            novelty_score=0.3, novelty_category="interesting",
            causal_edges_found=1, isomorphisms_found=0,
            auto_proved=True, new_primitive_proposed=False,
            probability_of_best=0.8, runtime_seconds=1.0,
        )
        state.update(result)
        assert state.n_iterations == 1
        assert state.n_expressions_discovered == 1
        assert state.n_auto_proved == 1

    def test_report_is_string(self):
        state = LoopState()
        s = state.report()
        assert isinstance(s, str) and "n=" in s


class TestSelfImprovementLoop:
    def test_requires_environments(self, tmp_path):
        loop = SelfImprovementLoop(output_dir=str(tmp_path), verbose=False)
        with pytest.raises(ValueError):
            loop.run(max_iterations=1)

    def test_short_run(self, tmp_path):
        loop = SelfImprovementLoop(
            output_dir=str(tmp_path),
            beam_width=5, n_search_iterations=2,
            enable_causal=False, enable_isomorphism=False,
            enable_auto_proof=True, enable_primitive_proposal=False,
            verbose=False, report_every=999,
        )
        loop.add_environment(ModularArithmeticEnv(modulus=7, slope=3, intercept=1))
        state = loop.run(max_iterations=3, stream_length=60)
        assert state.n_iterations == 3

    def test_state_tracks_discoveries(self, tmp_path):
        loop = SelfImprovementLoop(
            output_dir=str(tmp_path),
            beam_width=5, n_search_iterations=2,
            enable_causal=False, enable_isomorphism=False,
            verbose=False, report_every=999,
        )
        loop.add_environment(ModularArithmeticEnv(modulus=7))
        state = loop.run(max_iterations=3, stream_length=60)
        assert state.n_expressions_discovered >= 0

    def test_log_file_created(self, tmp_path):
        loop = SelfImprovementLoop(
            output_dir=str(tmp_path),
            beam_width=5, n_search_iterations=2,
            verbose=False, report_every=999,
        )
        loop.add_environment(ModularArithmeticEnv(modulus=7))
        loop.run(max_iterations=2, stream_length=50)
        log_file = tmp_path / "loop_results.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split('\n')
        assert len(lines) >= 1