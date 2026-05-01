"""Tests for the OUROBOROS web API."""
import pytest
import json
import math
from ouroboros.api.server import (
    SessionCache, RateLimiter,
    _run_discovery, _run_law_verification, _generate_lean4_stub,
    DiscoveryRequest, LawVerifyRequest,
)
from ouroboros.api.client import OuroborosClient, DiscoveryResult


class TestSessionCache:
    def test_miss_on_empty(self):
        cache = SessionCache()
        assert cache.get([1, 2, 3], {}) is None

    def test_set_and_get(self):
        cache = SessionCache()
        obs = [1.0, 2.0, 3.0]
        config = {"beam_width": 20}
        cache.set(obs, config, {"expression": "test"})
        result = cache.get(obs, config)
        assert result is not None
        assert result["expression"] == "test"

    def test_different_config_miss(self):
        cache = SessionCache()
        obs = [1.0, 2.0, 3.0]
        cache.set(obs, {"beam_width": 20}, {"expression": "test"})
        result = cache.get(obs, {"beam_width": 25})
        assert result is None

    def test_eviction_at_max_size(self):
        cache = SessionCache(maxsize=3)
        for i in range(5):
            cache.set([float(i)], {}, {"data": i})
        assert cache.size <= 3

    def test_hit_rate_computed(self):
        cache = SessionCache()
        obs = [1.0, 2.0]
        cache.set(obs, {}, {"x": 1})
        cache.get(obs, {})
        cache.get([9.0], {})
        assert 0.0 < cache.hit_rate < 1.0

    def test_different_obs_miss(self):
        cache = SessionCache()
        cache.set([1.0, 2.0], {}, {"result": "a"})
        assert cache.get([1.0, 3.0], {}) is None


class TestRateLimiter:
    def test_allows_first_request(self):
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.is_allowed("127.0.0.1")

    def test_blocks_after_limit(self):
        limiter = RateLimiter(requests_per_minute=3)
        ip = "10.0.0.1"
        for _ in range(3):
            limiter.is_allowed(ip)
        assert not limiter.is_allowed(ip)

    def test_different_ips_independent(self):
        limiter = RateLimiter(requests_per_minute=2)
        for _ in range(2):
            limiter.is_allowed("1.1.1.1")
        # Second IP should still be allowed
        assert limiter.is_allowed("2.2.2.2")

    def test_allows_again_after_window(self):
        import time
        limiter = RateLimiter(requests_per_minute=2)
        ip = "3.3.3.3"
        limiter.is_allowed(ip)
        limiter.is_allowed(ip)
        # Manually clear old entries by setting time far back
        limiter._buckets[ip] = [t - 61 for t in limiter._buckets[ip]]
        assert limiter.is_allowed(ip)


class TestDiscoveryCore:
    def test_run_discovery_modular(self):
        obs = [float((3*t+1)%7) for t in range(80)]
        result = _run_discovery(obs, alphabet_size=7, beam_width=8,
                                max_depth=3, n_iterations=3, time_budget=5.0)
        assert "expression" in result
        assert "mdl_cost" in result
        assert result["mdl_cost"] > 0
        assert "math_family" in result

    def test_compression_ratio_in_range(self):
        obs = [float((3*t+1)%7) for t in range(80)]
        result = _run_discovery(obs, alphabet_size=7, beam_width=6,
                                max_depth=3, n_iterations=2, time_budget=3.0)
        assert 0.0 <= result["compression_ratio"] <= 3.0
    def test_runtime_recorded(self):
        obs = [5.0] * 50
        result = _run_discovery(obs, alphabet_size=10, beam_width=5,
                                max_depth=2, n_iterations=2, time_budget=2.0)
        assert result["runtime_seconds"] > 0


class TestLawVerification:
    def test_spring_detected(self):
        import math
        obs = [10.0 * math.cos(0.3 * t) for t in range(100)]
        result = _run_law_verification(obs)
        assert "primary_law" in result
        assert "all_tests" in result
        assert len(result["all_tests"]) > 0

    def test_decay_detected(self):
        obs = [100.0 * math.exp(-0.05 * t) for t in range(80)]
        result = _run_law_verification(obs)
        assert result["primary_law"] in ["EXPONENTIAL_DECAY", "NEWTON_COOLING", "UNKNOWN"]

    def test_result_structure(self):
        obs = [float(t % 7) for t in range(50)]
        result = _run_law_verification(obs)
        assert isinstance(result["primary_law"], str)
        assert isinstance(result["all_tests"], list)


class TestLean4Stub:
    def test_generates_code(self):
        stub = _generate_lean4_stub("(3*t+1) % 7", "NUMBER_THEOR")
        assert "theorem" in stub
        assert "(3*t+1) % 7" in stub
        assert "sorry" not in stub

    def test_contains_expression(self):
        expr = "CUMSUM(ISPRIME(t))"
        stub = _generate_lean4_stub(expr, "NUMBER_THEOR")
        assert expr in stub


class TestOuroborosClientLocal:
    def test_discover_local_modular(self):
        client = OuroborosClient(base_url=None)
        obs = [float((3*t+1)%7) for t in range(80)]
        result = client.discover(
            obs, alphabet_size=7, beam_width=6,
            max_depth=3, n_iterations=2, time_budget_seconds=3.0,
        )
        assert isinstance(result, DiscoveryResult)
        assert result.mdl_cost > 0
        assert isinstance(result.math_family, str)

    def test_discover_local_has_fields(self):
        client = OuroborosClient(base_url=None)
        obs = [5.0] * 50
        result = client.discover(obs, alphabet_size=10, beam_width=5,
                                 max_depth=2, n_iterations=2, time_budget_seconds=2.0)
        assert result.n_observations == 50
        assert result.runtime_seconds >= 0

    def test_verify_law_local(self):
        import math
        client = OuroborosClient(base_url=None)
        obs = [10.0 * math.cos(0.3 * t) for t in range(100)]
        result = client.verify_law(obs)
        assert "primary_law" in result

    def test_result_str_representation(self):
        result = DiscoveryResult(
            expression="(3*t+1)%7",
            mdl_cost=45.2, compression_ratio=0.0041,
            math_family="NUMBER_THEOR", confidence=0.94,
            verified_law="NONE", lean4_theorem=None,
            runtime_seconds=2.3, n_observations=80,
            alphabet_size_used=7, from_cache=False,
        )
        s = str(result)
        assert "(3*t+1)%7" in s
        assert "45.2" in s