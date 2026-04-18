"""Unit tests for CausalTheory and TheoryAgent."""
import pytest
from ouroboros.emergence.causal_theory import CausalTheory, ScaleAxiom
from ouroboros.compression.program_synthesis import build_linear_modular, C, T, MOD


class TestScaleAxiom:
    def test_survival_rate_zero_challenges(self):
        expr = build_linear_modular(3, 1, 7)
        ax = ScaleAxiom(expr, scale=1, confidence=0.8, compression_ratio=0.1)
        assert ax.challenge_survival_rate() == 1.0

    def test_survival_rate_with_challenges(self):
        expr = build_linear_modular(3, 1, 7)
        ax = ScaleAxiom(expr, scale=1, confidence=0.8, compression_ratio=0.1)
        ax.times_challenged = 4
        ax.times_survived_market = 3
        assert abs(ax.challenge_survival_rate() - 0.75) < 0.001


class TestCausalTheory:
    def setup_method(self):
        self.theory = CausalTheory([1, 4, 16], alphabet_size=7)

    def test_empty_theory_not_complete(self):
        assert not self.theory.is_complete()

    def test_update_scale_adds_axiom(self):
        expr = build_linear_modular(3, 1, 7)
        self.theory.update_scale(1, expr, 0.05, 0.90)
        assert self.theory.axioms[1] is not None
        assert self.theory.axioms[1].confidence == 0.90

    def test_update_scale_no_downgrade(self):
        expr1 = build_linear_modular(3, 1, 7)
        expr2 = build_linear_modular(5, 2, 7)
        self.theory.update_scale(1, expr1, 0.05, 0.90)
        self.theory.update_scale(1, expr2, 0.30, 0.60)  # Worse ratio
        assert self.theory.axioms[1].expression.to_string() == expr1.to_string()

    def test_complete_when_all_scales_filled(self):
        for scale in [1, 4, 16]:
            expr = build_linear_modular(3, 1, 7)
            self.theory.update_scale(scale, expr, 0.05, 0.80)
        assert self.theory.is_complete(min_confidence=0.70)

    def test_mean_compression_ratio(self):
        self.theory.update_scale(1, build_linear_modular(3,1,7), 0.10, 0.80)
        self.theory.update_scale(4, build_linear_modular(3,1,7), 0.20, 0.70)
        mean = self.theory.mean_compression_ratio()
        assert abs(mean - 0.15) < 0.001

    def test_richness_score_increases_with_better_axioms(self):
        self.theory.update_scale(1, build_linear_modular(3,1,7), 0.80, 0.30)
        low_richness = self.theory.richness_score()
        t2 = CausalTheory([1, 4, 16], 7)
        t2.update_scale(1, build_linear_modular(3,1,7), 0.05, 0.95)
        t2.update_scale(4, build_linear_modular(3,1,7), 0.06, 0.90)
        high_richness = t2.richness_score()
        assert high_richness > low_richness

    def test_record_market_result_updates_confidence(self):
        expr = build_linear_modular(3, 1, 7)
        self.theory.update_scale(1, expr, 0.05, 0.70)
        self.theory.record_market_result(1, survived=True)
        assert self.theory.axioms[1].confidence > 0.70
        assert self.theory.axioms[1].times_survived_market == 1

    def test_best_expression_returns_lowest_ratio(self):
        self.theory.update_scale(1, build_linear_modular(3,1,7), 0.20, 0.80)
        self.theory.update_scale(4, build_linear_modular(5,2,7), 0.05, 0.90)
        best = self.theory.best_expression()
        assert best is not None
        assert best.to_string() == build_linear_modular(5,2,7).to_string()

    def test_to_dict_structure(self):
        expr = build_linear_modular(3, 1, 7)
        self.theory.update_scale(1, expr, 0.05, 0.90)
        d = self.theory.to_dict()
        assert 'axioms' in d
        assert '1' in d['axioms']
        assert 'expression' in d['axioms']['1']

    def test_summary_string(self):
        s = self.theory.summary()
        assert 'CausalTheory' in s
        assert 'Scale' in s


class TestTheoryAgent:
    def test_init_creates_theory(self):
        from ouroboros.agents.theory_agent import TheoryAgent
        agent = TheoryAgent(0, 7, scales=[1, 4])
        assert agent.theory is not None
        assert len(agent.theory.scales) == 2

    def test_update_theory_from_search(self):
        from ouroboros.agents.theory_agent import TheoryAgent
        from ouroboros.environment.structured import ModularArithmeticEnv
        env = ModularArithmeticEnv(7, 3, 1)
        env.reset(300)
        agent = TheoryAgent(0, 7, scales=[1, 4],
                            beam_width=8, max_depth=2, mcmc_iterations=20)
        agent.observe(env.peek_all())
        agent.search_and_update()
        agent.update_theory_from_search(step=1)
        # Should have at least tried to populate theory
        assert len(agent.theory_richness_history) >= 1