"""Unit tests for Phase2Runner."""
import pytest
from ouroboros.core.phase2_runner import Phase2Runner, Phase2Results, is_converged
from ouroboros.agents.self_modifying_agent import SelfModifyingAgent
from ouroboros.compression.program_synthesis import build_linear_modular


class TestIsConverged:
    def test_empty_agents_not_converged(self):
        assert not is_converged([])

    def test_high_ratio_not_converged(self):
        agents = [SelfModifyingAgent(0, 7), SelfModifyingAgent(1, 7)]
        for a in agents:
            a.compression_ratios = [0.8]
        assert not is_converged(agents, threshold=0.05)

    def test_low_ratio_same_expr_converged(self):
        agents = [SelfModifyingAgent(0, 7), SelfModifyingAgent(1, 7)]
        expr = build_linear_modular(3, 1, 7)
        for a in agents:
            a.compression_ratios = [0.004]
            a.best_expression = expr
            a._using_symbolic = True
        assert is_converged(agents, threshold=0.05)


class TestPhase2Runner:
    def test_factory_for_modular(self):
        runner = Phase2Runner.for_modular_arithmetic(7, 3, 1, num_agents=2)
        assert runner.num_agents == 2
        assert '7' in runner.environment_name

    def test_run_returns_results(self):
        runner = Phase2Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir='experiments/phase2/runs/test_p2_tmp'
        )
        results = runner.run(num_rounds=3, verbose=False)
        assert isinstance(results, Phase2Results)
        assert results.num_rounds == 3
        assert len(results.final_compression_ratios) == 2

    def test_approval_rate_in_range(self):
        runner = Phase2Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir='experiments/phase2/runs/test_rate_tmp'
        )
        results = runner.run(num_rounds=3, verbose=False)
        assert 0.0 <= results.approval_rate() <= 1.0

    def test_results_to_dict(self):
        runner = Phase2Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir='experiments/phase2/runs/test_dict_tmp'
        )
        results = runner.run(num_rounds=2, verbose=False)
        d = results.to_dict()
        assert 'environment' in d
        assert 'converged' in d
        assert 'approval_rate' in d

    def test_save_results(self, tmp_path):
        runner = Phase2Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir=str(tmp_path / 'runs')
        )
        runner.run(num_rounds=2, verbose=False)
        out = str(tmp_path / 'results.json')
        runner.save_results(out)
        import os
        assert os.path.exists(out)