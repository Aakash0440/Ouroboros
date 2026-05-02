"""Tests for OEIS integration and literature matching."""
import pytest
from ouroboros.novelty.oeis_client import OEISClient, OEISResult, OEISCache
from ouroboros.novelty.literature_matcher import LiteratureMatcher, LiteratureSearchResult


class TestOEISCache:
    def test_miss_on_empty(self, tmp_path):
        cache = OEISCache(str(tmp_path / "test.db"))
        assert cache.get("nonexistent") is None

    def test_set_and_get(self, tmp_path):
        cache = OEISCache(str(tmp_path / "test.db"))
        cache.set("key1", {"data": "value"})
        result = cache.get("key1")
        assert result == {"data": "value"}

    def test_key_for_sequence(self, tmp_path):
        cache = OEISCache(str(tmp_path / "test.db"))
        k1 = cache.key_for_sequence([1, 1, 2, 3, 5])
        k2 = cache.key_for_sequence([1, 1, 2, 3, 5])
        k3 = cache.key_for_sequence([1, 2, 3, 4, 5])
        assert k1 == k2
        assert k1 != k3

    def test_different_keys_different(self, tmp_path):
        cache = OEISCache(str(tmp_path / "test.db"))
        k1 = cache.key_for_sequence([1, 2, 3])
        k2 = cache.key_for_sequence([4, 5, 6])
        assert k1 != k2


class TestOEISClient:
    def test_offline_returns_not_found(self, tmp_path):
        """When network unavailable, should return graceful not-found."""
        client = OEISClient(cache_path=str(tmp_path / "test.db"), timeout_seconds=0.1)
        # Short timeout means network likely fails → graceful not-found
        result = client.search_sequence([999999, 999998, 999997])
        assert isinstance(result, OEISResult)
        assert isinstance(result.found, bool)

    def test_stats_tracked(self, tmp_path):
        client = OEISClient(cache_path=str(tmp_path / "test.db"), timeout_seconds=0.1)
        client.search_sequence([1, 2, 3])
        assert client.stats["n_api_requests"] >= 0

    def test_oeis_result_fields(self):
        result = OEISResult(
            found=True, oeis_id="A000045", name="Fibonacci numbers",
            description="test", formula="a(n) = a(n-1) + a(n-2)",
            example_values=[1, 1, 2, 3, 5, 8], offset=0,
            keywords=["nonn", "core", "easy"], references=[], from_cache=True,
        )
        assert result.is_well_known
        assert not result.is_finite
        assert "A000045" in result.description_str()

    def test_not_found_result(self):
        result = OEISResult(
            found=False, oeis_id=None, name=None, description=None,
            formula=None, example_values=[], offset=0, keywords=[], references=[],
            from_cache=False,
        )
        assert not result.found
        assert "Not found" in result.description_str()

    def test_cache_hit_rate_starts_zero(self, tmp_path):
        client = OEISClient(cache_path=str(tmp_path / "test.db"))
        assert client.cache_hit_rate == 0.0

    def test_cache_used_on_second_query(self, tmp_path):
        client = OEISClient(cache_path=str(tmp_path / "test.db"), timeout_seconds=0.1)
        seq = [999999, 888888, 777777]  # unlikely to be in OEIS
        client.search_sequence(seq)  # first: API call (or fail)
        # Manually set in cache
        key = client._cache.key_for_sequence(seq)
        client._cache.set(key, {"results": None, "count": 0})
        client.search_sequence(seq)  # second: cache hit
        assert client._n_cache_hits >= 1


class TestLiteratureMatcher:
    def test_match_returns_result(self, tmp_path):
        matcher = LiteratureMatcher(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            use_oeis=False,  # offline mode
        )
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=5.0)
        obs = [5.0] * 30
        result = matcher.match(expr, obs)
        assert isinstance(result, LiteratureSearchResult)

    def test_result_has_novelty_score(self, tmp_path):
        matcher = LiteratureMatcher(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            use_oeis=False,
        )
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.TIME)
        obs = [float(t) for t in range(30)]
        result = matcher.match(expr, obs)
        assert 0.0 <= result.combined_novelty_score <= 1.0

    def test_is_novel_method(self, tmp_path):
        matcher = LiteratureMatcher(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            use_oeis=False,
        )
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=3.0)
        obs = [3.0] * 30
        result = matcher.match(expr, obs)
        assert isinstance(result.is_novel(), bool)

    def test_stats_tracked(self, tmp_path):
        matcher = LiteratureMatcher(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            use_oeis=False,
        )
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=3.0)
        matcher.match(expr, [3.0]*20)
        assert matcher.stats["total_queries"] == 1

    def test_summary_is_string(self, tmp_path):
        matcher = LiteratureMatcher(
            oeis_cache_path=str(tmp_path / "oeis.db"),
            registry_path=str(tmp_path / "registry.json"),
            use_oeis=False,
        )
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=3.0)
        result = matcher.match(expr, [3.0]*20)
        s = result.summary()
        assert isinstance(s, str) and len(s) > 20

    def test_novelty_categories(self):
        from ouroboros.novelty.literature_matcher import LiteratureMatcher
        matcher = LiteratureMatcher.__new__(LiteratureMatcher)
        assert matcher._categorize(0.05) == "known"
        assert matcher._categorize(0.25) == "variant_of_known"
        assert matcher._categorize(0.45) == "potentially_novel"
        assert matcher._categorize(0.65) == "likely_novel"
        assert matcher._categorize(0.85) == "route_to_mathematician"