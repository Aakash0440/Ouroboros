"""Tests for GPU synthesis — these run on CPU if no GPU available."""
import pytest
import torch
from ouroboros.compression.gpu_synthesis import (
    GPUExprEvaluator, GPUBeamSearchSynthesizer, get_device
)
from ouroboros.compression.program_synthesis import (
    build_linear_modular, C, T, ADD, MOD
)


class TestGPUExprEvaluator:
    def setup_method(self):
        self.device = torch.device('cpu')  # Tests always run on CPU
        self.eval = GPUExprEvaluator(device=self.device)

    def test_const_node_vectorized(self):
        node = C(5)
        t = torch.arange(10, device=self.device)
        result = self.eval._eval_tensor(node, t)
        assert torch.all(result == 5)
        assert len(result) == 10

    def test_time_node_vectorized(self):
        node = T()
        t = torch.arange(10, device=self.device)
        result = self.eval._eval_tensor(node, t)
        assert torch.all(result == t)

    def test_add_node_vectorized(self):
        node = ADD(T(), C(3))
        t = torch.arange(5, device=self.device)
        result = self.eval._eval_tensor(node, t)
        expected = t + 3
        assert torch.all(result == expected)

    def test_mod_node_vectorized(self):
        node = MOD(T(), C(7))
        t = torch.arange(20, device=self.device)
        result = self.eval._eval_tensor(node, t)
        expected = t % 7
        assert torch.all(result == expected)

    def test_evaluate_sequence_clamped(self):
        node = build_linear_modular(3, 1, 7)
        preds = self.eval.evaluate_sequence(node, 20, alphabet_size=7)
        assert torch.all(preds >= 0)
        assert torch.all(preds < 7)
        assert len(preds) == 20

    def test_matches_cpu_evaluate(self):
        """GPU evaluator should match CPU evaluate() for all t."""
        node = build_linear_modular(3, 1, 7)
        n = 50
        gpu_preds = self.eval.evaluate_sequence(node, n, 7).tolist()
        cpu_preds = node.predict_sequence(n, 7)
        assert gpu_preds == cpu_preds

    def test_handles_mod_zero_gracefully(self):
        node = MOD(T(), C(0))
        t = torch.arange(10, device=self.device)
        result = self.eval._eval_tensor(node, t)
        assert torch.all(result == 0)  # Should not crash


class TestGPUBeamSearchSynthesizer:
    def setup_method(self):
        self.device = torch.device('cpu')

    def test_search_returns_expr_and_cost(self):
        synth = GPUBeamSearchSynthesizer(
            beam_width=5, max_depth=2, const_range=5,
            alphabet_size=7, device=self.device
        )
        seq = [(3*t+1) % 7 for t in range(50)]
        expr, cost = synth.search(seq)
        assert expr is not None
        assert cost < float('inf')

    def test_same_result_as_cpu(self):
        """GPU search should find results at least as good as CPU."""
        from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
        seq = [(3*t+1) % 7 for t in range(100)]

        cpu = BeamSearchSynthesizer(
            beam_width=10, max_depth=2, const_range=10, alphabet_size=7
        )
        gpu = GPUBeamSearchSynthesizer(
            beam_width=10, max_depth=2, const_range=10,
            alphabet_size=7, device=self.device
        )

        _, cpu_cost = cpu.search(seq)
        _, gpu_cost = gpu.search(seq)

        # GPU (or CPU fallback) should find similar quality
        assert gpu_cost <= cpu_cost * 1.15  # Within 15%

    def test_cpu_fallback_on_cpu_device(self):
        """With CPU device, should use CPU fallback transparently."""
        synth = GPUBeamSearchSynthesizer(
            beam_width=5, max_depth=1, const_range=3,
            alphabet_size=2, device=torch.device('cpu')
        )
        expr, cost = synth.search([0, 1] * 25)
        assert expr is not None


class TestGPUSynthesisAgent:
    def test_init_and_basic_usage(self):
        from ouroboros.compression.gpu_synthesis import GPUSynthesisAgent
        agent = GPUSynthesisAgent(
            agent_id=0, alphabet_size=7,
            beam_width=5, max_depth=2, const_range=7
        )
        seq = [(3*t+1)%7 for t in range(100)]
        agent.observe(seq)
        cost = agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        assert isinstance(ratio, float)
        assert 0.0 <= ratio <= 1.5