"""Unit tests for Phase1Runner."""
import pytest
from ouroboros.core.phase1_runner import Phase1Runner, Phase1Results
from ouroboros.environmentss.structured import BinaryRepeatEnv


class TestPhase1Runner:
    def test_factory_modular(self):
        runner = Phase1Runner.for_modular_arithmetic(7, 3, 1, num_agents=2)
        assert runner.num_agents == 2
        assert runner.agent_type == 'synthesis'
        assert 'Modular' in runner.environment_name

    def test_factory_noise_baseline(self):
        runner = Phase1Runner.for_noise_baseline(num_agents=2)
        assert runner.num_agents == 2
        assert 'Noise' in runner.environment_name

    def test_factory_multiscale(self):
        runner = Phase1Runner.for_multiscale(28, 7, num_agents=2)
        assert runner.agent_type == 'hierarchical'
        assert len(runner.scales) > 0

    def test_run_returns_results(self):
        runner = Phase1Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir='experiments/phase1/runs/test_runner_tmp'
        )
        results = runner.run(stream_length=200, eval_interval=100, verbose=False)
        assert isinstance(results, Phase1Results)
        assert len(results.final_ratios) == 2
        assert 0.0 <= results.mean_ratio <= 1.5
        assert results.elapsed_seconds > 0

    def test_results_ratios_in_range(self):
        runner = Phase1Runner.for_noise_baseline(
            num_agents=2,
            run_dir='experiments/phase1/runs/test_noise_tmp'
        )
        results = runner.run(stream_length=200, eval_interval=100, verbose=False)
        for ratio in results.final_ratios.values():
            assert 0.0 <= ratio <= 2.0

    def test_noise_no_axioms(self):
        runner = Phase1Runner.for_noise_baseline(
            num_agents=3,
            run_dir='experiments/phase1/runs/test_noise_ax_tmp'
        )
        results = runner.run(stream_length=300, eval_interval=150, verbose=False)
        assert len(results.axioms_promoted) == 0, (
            f"Noise produced axioms: {results.axioms_promoted}"
        )

    def test_results_to_dict(self):
        runner = Phase1Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir='experiments/phase1/runs/test_dict_tmp'
        )
        results = runner.run(stream_length=200, eval_interval=100, verbose=False)
        d = results.to_dict()
        assert 'environment' in d
        assert 'mean_ratio' in d
        assert 'best_ratio' in d
        assert 'elapsed_seconds' in d
        assert isinstance(d['final_ratios'], dict)

    def test_save_results_creates_file(self, tmp_path):
        runner = Phase1Runner.for_modular_arithmetic(
            7, 3, 1, num_agents=2,
            run_dir=str(tmp_path / 'runs')
        )
        results = runner.run(stream_length=200, eval_interval=100, verbose=False)
        out = str(tmp_path / 'results.json')
        runner.save_results(out)
        import os
        assert os.path.exists(out)

    def test_agents_created_correctly(self):
        from ouroboros.agents.synthesis_agent import SynthesisAgent
        from ouroboros.agents.base_agent import BaseAgent
        from ouroboros.agents.hierarchical_agent import HierarchicalAgent

        for agent_type, expected_class in [
            ('base', BaseAgent),
            ('synthesis', SynthesisAgent),
            ('hierarchical', HierarchicalAgent),
        ]:
            runner = Phase1Runner(
                BinaryRepeatEnv(), "BinaryRepeat",
                num_agents=2, agent_type=agent_type,
                scales=[1, 4],
                run_dir=f'experiments/phase1/runs/test_{agent_type}_tmp'
            )
            runner.run(stream_length=100, eval_interval=50, verbose=False)
            for agent in runner._agents:
                assert isinstance(agent, expected_class), (
                    f"Expected {expected_class}, got {type(agent)}"
                )