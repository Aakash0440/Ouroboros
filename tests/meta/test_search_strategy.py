"""Tests for Layer 3 search strategy system."""
import pytest
import time
from ouroboros.meta.search_strategy import (
    SearchStrategy, SearchConfig, SearchResult,
    BeamSearchStrategy, RandomRestartStrategy,
    AnnealingStrategy, HybridStrategy, MultiScaleStrategy,
)
from ouroboros.meta.strategy_library import (
    StrategyProposal, StrategyEvaluationResult, SearchStrategyLibrary,
    STRATEGY_LIBRARY,
)
from ouroboros.meta.strategy_market import StrategyProofMarket, StrategyMarketConfig
from ouroboros.meta.layer3_agent import Layer3Agent, Layer3AgentConfig


# ─── SearchConfig ─────────────────────────────────────────────────────────────

class TestSearchConfig:
    def test_defaults_valid(self):
        cfg = SearchConfig()
        assert cfg.beam_width > 0
        assert cfg.time_budget_seconds > 0
        assert cfg.max_depth > 0

    def test_custom_values(self):
        cfg = SearchConfig(beam_width=50, time_budget_seconds=10.0)
        assert cfg.beam_width == 50
        assert cfg.time_budget_seconds == 10.0


# ─── SearchResult ─────────────────────────────────────────────────────────────

class TestSearchResult:
    def test_found_something(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=5)
        result = SearchResult(
            best_expr=expr,
            best_mdl_cost=50.0,
            n_evaluations=100,
            wall_time_seconds=1.0,
            strategy_name="test",
        )
        assert result.found_something

    def test_found_nothing(self):
        result = SearchResult(
            best_expr=None,
            best_mdl_cost=float('inf'),
            n_evaluations=0,
            wall_time_seconds=0.1,
            strategy_name="test",
        )
        assert not result.found_something

    def test_evaluations_per_second(self):
        result = SearchResult(None, float('inf'), 1000, 2.0, "test")
        assert result.evaluations_per_second == pytest.approx(500.0)


# ─── Concrete strategies: interface compliance ─────────────────────────────────

class TestStrategyInterface:
    """Every strategy must return a valid SearchResult."""
    
    OBS = list(range(100))   # simple arithmetic sequence — any strategy can handle it
    CFG = SearchConfig(
        beam_width=8, time_budget_seconds=2.0,
        n_restarts=3, node_budget=500,
        mcmc_iterations=20,
    )

    def _check_result(self, result: SearchResult, strategy_name: str):
        assert isinstance(result, SearchResult), f"{strategy_name} must return SearchResult"
        assert isinstance(result.strategy_name, str)
        assert result.n_evaluations >= 0
        assert result.wall_time_seconds >= 0.0
        # best_mdl_cost can be inf if nothing found, but must be finite if found
        if result.found_something:
            assert result.best_mdl_cost < float('inf')

    def test_beam_search_returns_result(self):
        result = BeamSearchStrategy().search(self.OBS, self.CFG)
        self._check_result(result, "BeamSearch")

    def test_random_restart_returns_result(self):
        result = RandomRestartStrategy().search(self.OBS, self.CFG)
        self._check_result(result, "RandomRestart")

    def test_annealing_returns_result(self):
        result = AnnealingStrategy().search(self.OBS, self.CFG)
        self._check_result(result, "SimulatedAnnealing")

    def test_hybrid_returns_result(self):
        result = HybridStrategy().search(self.OBS, self.CFG)
        self._check_result(result, "Hybrid")

    def test_multiscale_returns_result(self):
        result = MultiScaleStrategy(scales=[1, 4]).search(self.OBS, self.CFG)
        self._check_result(result, "MultiScale")


class TestStrategyDescriptionBits:
    def test_beam_cheapest(self):
        b = BeamSearchStrategy()
        h = HybridStrategy()
        assert b.description_bits() <= h.description_bits()

    def test_all_positive(self):
        for s in STRATEGY_LIBRARY.all_strategies():
            assert s.description_bits() > 0


# ─── StrategyLibrary ──────────────────────────────────────────────────────────

class TestStrategyLibrary:
    def test_contains_defaults(self):
        lib = SearchStrategyLibrary()
        assert "BeamSearch" in lib.all_names()
        assert "RandomRestart" in lib.all_names()
        assert "SimulatedAnnealing" in lib.all_names()

    def test_get_existing(self):
        lib = SearchStrategyLibrary()
        s = lib.get("BeamSearch")
        assert s is not None
        assert s.name() == "BeamSearch"

    def test_get_nonexistent_returns_none(self):
        lib = SearchStrategyLibrary()
        assert lib.get("NonExistentStrategy") is None

    def test_alternatives_excludes_current(self):
        lib = SearchStrategyLibrary()
        alts = lib.get_alternatives_to("BeamSearch")
        names = [s.name() for s in alts]
        assert "BeamSearch" not in names
        assert len(names) > 0

    def test_register_custom(self):
        lib = SearchStrategyLibrary()

        class MyStrategy(SearchStrategy):
            def name(self): return "MyCustomStrategy"
            def description(self): return "test"
            def search(self, obs, cfg): return SearchResult(None, float('inf'), 0, 0.0, self.name())

        lib.register(MyStrategy())
        assert "MyCustomStrategy" in lib.all_names()


# ─── StrategyProposal ────────────────────────────────────────────────────────

class TestStrategyProposal:
    def _make_proposal(self, current_cost=100.0, proposed_cost=85.0,
                       current_time=1.0, proposed_time=1.2):
        return StrategyProposal(
            proposing_agent="TEST",
            current_strategy_name="BeamSearch",
            proposed_strategy_name="RandomRestart",
            training_env_name="TestEnv",
            current_best_cost=current_cost,
            proposed_best_cost=proposed_cost,
            current_time_seconds=current_time,
            proposed_time_seconds=proposed_time,
        )

    def test_improvement_computed(self):
        p = self._make_proposal(100.0, 85.0)
        assert p.cost_improvement == pytest.approx(15.0)

    def test_is_improvement_positive(self):
        p = self._make_proposal(100.0, 85.0)
        assert p.is_improvement

    def test_is_improvement_negative(self):
        p = self._make_proposal(100.0, 110.0)
        assert not p.is_improvement

    def test_cost_efficiency(self):
        p = self._make_proposal(100.0, 85.0, 1.0, 3.0)  # 15 bits saved, 2s overhead
        assert p.cost_efficiency == pytest.approx(15.0 / 2.0)


# ─── StrategyProofMarket ─────────────────────────────────────────────────────

class TestStrategyProofMarket:
    def _make_proposal(self, improvement=15.0):
        return StrategyProposal(
            proposing_agent="A",
            current_strategy_name="BeamSearch",
            proposed_strategy_name="RandomRestart",
            training_env_name="T",
            current_best_cost=100.0,
            proposed_best_cost=100.0 - improvement,
            current_time_seconds=1.0,
            proposed_time_seconds=1.5,
        )

    def test_insufficient_improvement_rejected(self):
        market = StrategyProofMarket(
            config=StrategyMarketConfig(min_cost_improvement_bits=20.0)
        )
        p = self._make_proposal(improvement=5.0)
        from ouroboros.environments.modular import ModularArithmeticEnv
        result = market.evaluate_proposal(
            p, BeamSearchStrategy(), RandomRestartStrategy(),
            ModularArithmeticEnv()
        )
        assert not result.approved
        assert "Cost improvement" in result.rejection_reason

    def test_approval_rate_zero_initially(self):
        market = StrategyProofMarket()
        assert market.approval_rate == 0.0


# ─── Layer3Agent ────────────────────────────────────────────────────────────

class TestLayer3Agent:
    def test_initial_strategy_is_beam(self):
        agent = Layer3Agent()
        assert agent.current_strategy.name() == "BeamSearch"

    def test_apply_approved_strategy_changes_current(self):
        agent = Layer3Agent()
        agent.apply_approved_strategy("RandomRestart")
        assert agent.current_strategy.name() == "RandomRestart"

    def test_apply_nonexistent_strategy_noop(self):
        agent = Layer3Agent()
        agent.apply_approved_strategy("DoesNotExist")
        assert agent.current_strategy.name() == "BeamSearch"  # unchanged

    def test_stats_initialized(self):
        agent = Layer3Agent()
        assert agent.stats.total_rounds == 0
        assert agent.stats.strategy_proposals == 0
        assert "BeamSearch" in agent.stats.strategy_history

    def test_search_returns_result(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        agent = Layer3Agent(
            config=Layer3AgentConfig(search_time_budget=1.0, search_beam_width=5)
        )
        env = ModularArithmeticEnv(modulus=7, slope=3, intercept=1)
        obs = env.generate(100)
        result = agent.search(obs)
        assert isinstance(result, SearchResult)
        assert result.wall_time_seconds >= 0