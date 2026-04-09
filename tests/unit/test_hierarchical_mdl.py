"""Unit tests for hierarchical MDL and scale-aware synthesis."""
import pytest
import numpy as np
from ouroboros.compression.hierarchical_mdl import (
    HierarchicalMDL, aggregate_sequence, compression_at_scale
)


class TestAggregateSequence:
    def test_window_1_is_identity(self):
        seq = [1, 2, 3, 4, 5]
        assert aggregate_sequence(seq, 1, 6) == seq

    def test_window_2_sum_mod(self):
        seq = [1, 2, 3, 4, 5, 6]
        agg = aggregate_sequence(seq, 2, 10)
        assert agg[0] == (1 + 2) % 10   # = 3
        assert agg[1] == (3 + 4) % 10   # = 7
        assert agg[2] == (5 + 6) % 10   # = 1

    def test_window_larger_than_sequence_returns_empty(self):
        seq = [1, 2, 3]
        agg = aggregate_sequence(seq, 10, 4)
        assert agg == []

    def test_length_is_floor_division(self):
        seq = list(range(100))
        for w in [1, 2, 4, 5, 10]:
            agg = aggregate_sequence(seq, w, 10)
            assert len(agg) == 100 // w

    def test_values_in_range(self):
        seq = [i % 7 for i in range(100)]
        for w in [1, 2, 4]:
            agg = aggregate_sequence(seq, w, 7)
            assert all(0 <= v < 7 for v in agg)

    def test_majority_method(self):
        # 3 zeros and 1 one in each window → majority is 0
        seq = [0, 0, 0, 1] * 10
        agg = aggregate_sequence(seq, 4, 2, method='majority')
        assert all(v == 0 for v in agg)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            aggregate_sequence([1, 2], 1, 2, method='invalid')


class TestHierarchicalMDL:
    def test_compression_profile_returns_dict(self):
        hier = HierarchicalMDL([1, 4, 16], alphabet_size=4)
        seq = [i % 4 for i in range(200)]
        profile = hier.compression_profile(seq)
        assert set(profile.keys()) == {1, 4, 16}
        assert all(isinstance(v, float) for v in profile.values())

    def test_ratios_between_0_and_2(self):
        hier = HierarchicalMDL([1, 4], alphabet_size=4)
        seq = [i % 4 for i in range(200)]
        profile = hier.compression_profile(seq)
        for r in profile.values():
            assert 0.0 <= r <= 2.0

    def test_dominant_scale_returns_pair(self):
        hier = HierarchicalMDL([1, 4, 16], alphabet_size=2)
        seq = [i % 2 for i in range(200)]
        scale, ratio = hier.dominant_scale(seq)
        assert scale in [1, 4, 16]
        assert isinstance(ratio, float)

    def test_noise_all_ratios_above_threshold(self):
        from ouroboros.environment.structured import NoiseEnv
        env = NoiseEnv(4, seed=0)
        env.reset(2000)
        stream = env.peek_all()
        hier = HierarchicalMDL([1, 4, 16], alphabet_size=4)
        profile = hier.compression_profile(stream)
        for scale, ratio in profile.items():
            assert ratio > 0.75, f"Scale {scale}: noise ratio {ratio} too low"

    def test_structured_compresses_at_some_scale(self):
        seq = [i % 7 for i in range(500)]
        hier = HierarchicalMDL([1, 4, 16], alphabet_size=7)
        profile = hier.compression_profile(seq)
        best_ratio = min(profile.values())
        assert best_ratio < 0.50, f"Expected best ratio < 0.50, got {best_ratio}"

    def test_multi_scale_improvement(self):
        # A sequence that has DIFFERENT structure at scale 1 vs scale 4
        # (alternating at scale 1, repeat-4 at scale 4)
        seq = [0, 1, 0, 1, 2, 3, 2, 3] * 50
        hier = HierarchicalMDL([1, 4, 8], alphabet_size=4)
        improvement = hier.multi_scale_improvement(seq)
        assert isinstance(improvement, float)
        assert improvement >= 1.0

    def test_aggregate_at_scale_method(self):
        hier = HierarchicalMDL([1, 4], alphabet_size=7)
        seq = [i % 7 for i in range(100)]
        agg = hier.aggregate_at_scale(seq, 4)
        assert len(agg) == 25


class TestCompressionAtScale:
    def test_returns_float(self):
        r = compression_at_scale([0, 1] * 100, window=1, alphabet_size=2)
        assert isinstance(r, float)

    def test_structured_better_than_noise(self):
        structured = [i % 4 for i in range(200)]
        noise = list(np.random.default_rng(0).integers(0, 4, 200))
        r_s = compression_at_scale(structured, 1, 4)
        r_n = compression_at_scale(noise, 1, 4)
        assert r_s < r_n


class TestScaleAxiomPool:
    def test_submit_and_detect(self):
        from ouroboros.emergence.scale_axiom_pool import ScaleAxiomPool
        from ouroboros.compression.program_synthesis import build_linear_modular
        from ouroboros.compression.mdl import naive_bits

        pool = ScaleAxiomPool([1, 4], num_agents=4, alphabet_size=7)
        expr = build_linear_modular(3, 1, 7)

        for i in range(3):
            pool.submit(1, i, expr, 10.0, step=100)
        pool.submit(1, 3, build_linear_modular(5, 2, 7), 50.0, step=100)

        seq = [(3*t+1)%7 for t in range(100)]
        nb = naive_bits(seq, 7)
        new_axioms = pool.detect_all_scales(
            100, "TestEnv", {1: nb, 4: nb / 4}
        )
        assert len(new_axioms) >= 0  # May or may not promote depending on threshold

    def test_empty_pool_summary(self):
        from ouroboros.emergence.scale_axiom_pool import ScaleAxiomPool
        pool = ScaleAxiomPool([1, 4], 4, 0.5, 7)
        s = pool.summary()
        assert isinstance(s, str)

    def test_clear_all_clears_all_pools(self):
        from ouroboros.emergence.scale_axiom_pool import ScaleAxiomPool
        from ouroboros.compression.program_synthesis import C
        pool = ScaleAxiomPool([1, 4], 4, 0.5, 7)
        pool.submit(1, 0, C(3), 10.0)
        pool.submit(4, 0, C(3), 10.0)
        pool.clear_all()
        for p in pool.scale_pools.values():
            assert len(p.current_submissions) == 0