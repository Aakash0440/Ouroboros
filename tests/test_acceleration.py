"""Tests for GPU/NumPy acceleration and large alphabet environments."""
import pytest
import numpy as np
from ouroboros.acceleration.batch_evaluator import (
    CompiledExpr, compile_expr, batch_mdl_score,
    build_history_matrix, GPUBatchEvaluator,
)
from ouroboros.acceleration.sparse_beam import SparseBeamSearch, SparseBeamConfig
from ouroboros.acceleration.alphabet_scaler import AlphabetScaler, ScalerConfig
from ouroboros.environments.large_alphabet import CRTLargeEnv, TripleCRTEnv
from ouroboros.synthesis.expr_node import ExprNode, NodeType


def make_const_expr(v: int) -> ExprNode:
    return ExprNode(NodeType.CONST, value=v)

def make_time_expr() -> ExprNode:
    return ExprNode(NodeType.TIME)

def make_mod_expr(slope: int, intercept: int, modulus: int) -> ExprNode:
    return ExprNode(NodeType.MOD,
        left=ExprNode(NodeType.ADD,
            left=ExprNode(NodeType.MUL,
                left=make_const_expr(slope),
                right=make_time_expr(),
            ),
            right=make_const_expr(intercept),
        ),
        right=make_const_expr(modulus),
    )


class TestCompiledExpr:
    def test_const_evaluates_correctly(self):
        expr = make_const_expr(7)
        compiled = compile_expr(expr)
        t = np.arange(10, dtype=np.float64)
        hist = np.zeros((10, 3), dtype=np.int64)
        result = compiled.evaluate_batch(t, hist)
        assert all(result == 7)

    def test_time_evaluates_correctly(self):
        expr = make_time_expr()
        compiled = compile_expr(expr)
        t = np.arange(10, dtype=np.float64)
        hist = np.zeros((10, 3), dtype=np.int64)
        result = compiled.evaluate_batch(t, hist)
        assert list(result) == list(range(10))

    def test_add_evaluates_correctly(self):
        expr = ExprNode(NodeType.ADD, left=make_const_expr(3), right=make_time_expr())
        compiled = compile_expr(expr)
        t = np.arange(5, dtype=np.float64)
        hist = np.zeros((5, 3), dtype=np.int64)
        result = compiled.evaluate_batch(t, hist)
        assert list(result) == [3, 4, 5, 6, 7]

    def test_mod_evaluates_correctly(self):
        # (3 * t + 1) % 7
        expr = make_mod_expr(3, 1, 7)
        compiled = compile_expr(expr)
        t = np.arange(14, dtype=np.float64)
        hist = np.zeros((14, 3), dtype=np.int64)
        result = compiled.evaluate_batch(t, hist)
        expected = [(3*i + 1) % 7 for i in range(14)]
        assert list(result) == expected

    def test_clamp_range(self):
        expr = make_const_expr(200)
        compiled = compile_expr(expr)
        t = np.arange(5, dtype=np.float64)
        hist = np.zeros((5, 3), dtype=np.int64)
        result = compiled.evaluate_batch(t, hist, clamp_range=10)
        assert all(0 <= v <= 9 for v in result)


class TestBuildHistoryMatrix:
    def test_shape(self):
        obs = list(range(20))
        t_arr, hist = build_history_matrix(obs, max_lag=3)
        assert t_arr.shape == (20,)
        assert hist.shape == (20, 3)

    def test_lag1_is_prev_obs(self):
        obs = [10, 20, 30, 40, 50]
        _, hist = build_history_matrix(obs, max_lag=2)
        # hist[t, 0] = obs[t-1], hist[1, 0] should be obs[0]=10
        assert hist[1, 0] == 10
        assert hist[2, 0] == 20

    def test_zero_padding_at_start(self):
        obs = [1, 2, 3, 4, 5]
        _, hist = build_history_matrix(obs, max_lag=3)
        # hist[0, :] should be all zeros (before sequence start)
        assert all(hist[0, :] == 0)


class TestBatchMDLScore:
    def test_perfect_prediction_scores_low(self):
        obs = [(3*t + 1) % 7 for t in range(200)]
        expr = make_mod_expr(3, 1, 7)
        wrong = make_mod_expr(5, 2, 7)
        costs = batch_mdl_score([expr, wrong], obs, alphabet_size=7)
        assert costs[0] < costs[1], "Perfect prediction should score lower"

    def test_returns_list_of_floats(self):
        obs = [1, 2, 3, 4, 5] * 20
        exprs = [make_const_expr(i) for i in range(5)]
        costs = batch_mdl_score(exprs, obs, alphabet_size=10)
        assert len(costs) == 5
        assert all(isinstance(c, float) for c in costs)

    def test_single_candidate(self):
        obs = [7] * 50
        expr = make_const_expr(7)
        costs = batch_mdl_score([expr], obs, alphabet_size=13)
        assert len(costs) == 1
        assert costs[0] < float('inf')


class TestSparseBeamSearch:
    def test_returns_expr_on_modular(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
        obs = env.generate(200)
        cfg = SparseBeamConfig(beam_width=10, n_iterations=5)
        searcher = SparseBeamSearch(cfg)
        result = searcher.search(obs, alphabet_size=7)
        assert result is not None

    def test_warm_start_seeds_generated(self):
        searcher = SparseBeamSearch()
        seeds = searcher._warm_start_seeds(77)
        assert len(seeds) > 0
        # All seeds should be valid ExprNodes
        for seed in seeds:
            assert isinstance(seed, ExprNode)

    def test_mutation_returns_different_expr(self):
        searcher = SparseBeamSearch(SparseBeamConfig(random_seed=1))
        expr = make_mod_expr(3, 1, 7)
        mutations = [searcher._mutate(expr) for _ in range(20)]
        # At least some mutations should differ from original
        orig_str = expr.to_string()
        mutated_strs = [m.to_string() for m in mutations]
        assert any(s != orig_str for s in mutated_strs)


class TestAlphabetScaler:
    def test_small_uses_standard(self):
        scaler = AlphabetScaler()
        strategy = scaler._choose_strategy(alphabet_size=13, stream_length=500)
        assert strategy == "standard"

    def test_medium_uses_sparse_numpy(self):
        scaler = AlphabetScaler()
        strategy = scaler._choose_strategy(alphabet_size=77, stream_length=500)
        assert strategy == "sparse_numpy"

    def test_large_uses_sparse_gpu(self):
        scaler = AlphabetScaler()
        strategy = scaler._choose_strategy(alphabet_size=221, stream_length=500)
        assert strategy == "sparse_gpu"

    def test_search_returns_result_small(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        obs = ModularArithmeticEnv(modulus=7).generate(100)
        scaler = AlphabetScaler()
        result = scaler.search(obs, alphabet_size=7)
        assert result is not None


class TestLargeAlphabetEnvs:
    def test_crt_large_generates_correctly(self):
        env = CRTLargeEnv(mod1=13, mod2=17)
        seq = env.generate(100)
        assert len(seq) == 100
        assert all(0 <= v < 13 * 17 for v in seq)
        assert env.alphabet_size == 221

    def test_crt_large_is_deterministic(self):
        env = CRTLargeEnv()
        assert env.generate(50) == env.generate(50)

    def test_triple_crt_alphabet_size(self):
        env = TripleCRTEnv(mod1=7, mod2=11, mod3=13)
        assert env.alphabet_size == 7 * 11 * 13

    def test_triple_crt_values_in_range(self):
        env = TripleCRTEnv(mod1=7, mod2=11, mod3=13)
        seq = env.generate(100)
        assert all(0 <= v < env.alphabet_size for v in seq)

    def test_gpu_evaluator_availability(self):
        evaluator = GPUBatchEvaluator()
        # Must not crash even if GPU unavailable
        assert isinstance(evaluator.using_gpu, bool)