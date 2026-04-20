"""Tests for Layer 2 self-improvement: MDLObjective, ObjectiveProofMarket, Layer2Agent."""
import pytest
import copy
from ouroboros.agents.mdl_objective import (
    MDLObjective, ObjectiveProposal, DEFAULT_OBJECTIVE,
)
from ouroboros.agents.objective_market import ObjectiveProofMarket, ObjectiveMarketConfig
from ouroboros.agents.layer2_agent import Layer2Agent, Layer2AgentConfig


class TestMDLObjective:
    def test_default_is_valid(self):
        assert DEFAULT_OBJECTIVE.is_valid()

    def test_clamp_fixes_invalid(self):
        bad = MDLObjective(lambda_prog=-5.0, lambda_const=1000.0)
        good = bad.clamp()
        assert good.is_valid()

    def test_program_bits_scales_with_nodes(self):
        obj = MDLObjective(lambda_prog=2.0, lambda_const=8.0)
        bits_1 = obj.compute_program_bits(node_count=1, constant_count=0)
        bits_5 = obj.compute_program_bits(node_count=5, constant_count=0)
        assert bits_5 > bits_1
        assert bits_5 == pytest.approx(5 * 2.0)

    def test_program_bits_scales_with_constants(self):
        obj = MDLObjective(lambda_prog=2.0, lambda_const=8.0)
        bits_0 = obj.compute_program_bits(1, 0)
        bits_3 = obj.compute_program_bits(1, 3)
        assert bits_3 > bits_0
        assert bits_3 - bits_0 == pytest.approx(3 * 8.0)

    def test_description_bits_positive(self):
        assert DEFAULT_OBJECTIVE.description_bits() > 0.0

    def test_serialization_roundtrip(self):
        obj = MDLObjective(lambda_prog=3.5, lambda_const=12.0, axiom_r2_threshold=0.98)
        d = obj.to_dict()
        obj2 = MDLObjective.from_dict(d)
        assert obj2.lambda_prog == pytest.approx(3.5)
        assert obj2.lambda_const == pytest.approx(12.0)
        assert obj2.axiom_r2_threshold == pytest.approx(0.98)

    def test_description_is_string(self):
        s = DEFAULT_OBJECTIVE.description()
        assert "MDLObj" in s and "λ_prog" in s


class TestObjectiveProposal:
    def _make_proposal(self, improvement: float) -> ObjectiveProposal:
        current = DEFAULT_OBJECTIVE
        proposed = current.clamp()
        proposed.lambda_prog -= 0.5
        return ObjectiveProposal(
            proposing_agent="TEST_AGENT",
            current_objective=current,
            proposed_objective=proposed,
            training_env_name="TestEnv",
            current_total_bits=100.0,
            proposed_total_bits=100.0 - improvement,
        )

    def test_is_improvement_positive(self):
        p = self._make_proposal(improvement=10.0)
        assert p.is_improvement

    def test_is_improvement_negative(self):
        p = self._make_proposal(improvement=-5.0)
        assert not p.is_improvement

    def test_improvement_bits_computed(self):
        p = self._make_proposal(improvement=15.0)
        assert p.improvement_bits == pytest.approx(15.0)

    def test_improvement_fraction_computed(self):
        p = self._make_proposal(improvement=10.0)
        assert p.improvement_fraction == pytest.approx(0.1)


class TestObjectiveProofMarket:
    def _make_proposal(self, improvement_bits: float = 20.0) -> ObjectiveProposal:
        proposed = copy.deepcopy(DEFAULT_OBJECTIVE)
        proposed.lambda_prog = max(0.1, proposed.lambda_prog - 0.5)
        return ObjectiveProposal(
            proposing_agent="TEST",
            current_objective=DEFAULT_OBJECTIVE,
            proposed_objective=proposed,
            training_env_name="TestEnv",
            current_total_bits=200.0,
            proposed_total_bits=200.0 - improvement_bits,
        )

    def test_invalid_objective_rejected(self):
        market = ObjectiveProofMarket(config=ObjectiveMarketConfig(n_adversaries=0))
        bad_objective = MDLObjective(lambda_prog=-99.0)
        proposal = ObjectiveProposal(
            proposing_agent="T",
            current_objective=DEFAULT_OBJECTIVE,
            proposed_objective=bad_objective,
            training_env_name="T",
            current_total_bits=100.0,
            proposed_total_bits=50.0,
        )
        from ouroboros.environments.modular import ModularArithmeticEnv
        result = market.evaluate_proposal(proposal, ModularArithmeticEnv())
        assert not result.approved

    def test_insufficient_improvement_rejected(self):
        market = ObjectiveProofMarket(
            config=ObjectiveMarketConfig(min_improvement_bits=20.0, n_adversaries=0)
        )
        proposal = self._make_proposal(improvement_bits=1.0)  # too small
        from ouroboros.environments.modular import ModularArithmeticEnv
        result = market.evaluate_proposal(proposal, ModularArithmeticEnv())
        assert not result.approved
        assert "Insufficient" in result.rejection_reason

    def test_approval_rate_zero_initially(self):
        market = ObjectiveProofMarket()
        assert market.approval_rate == 0.0

    def test_best_objective_default_when_empty(self):
        market = ObjectiveProofMarket()
        obj = market.get_best_objective()
        assert obj.lambda_prog == DEFAULT_OBJECTIVE.lambda_prog


class TestLayer2Agent:
    def test_initial_objective_is_default(self):
        agent = Layer2Agent()
        assert agent.current_objective.lambda_prog == DEFAULT_OBJECTIVE.lambda_prog

    def test_apply_approved_updates_objective(self):
        agent = Layer2Agent()
        new_obj = MDLObjective(lambda_prog=1.0, lambda_const=5.0)
        agent.apply_approved_objective(new_obj)
        assert agent.current_objective.lambda_prog == pytest.approx(1.0)

    def test_stats_track_proposals(self):
        agent = Layer2Agent()
        agent.stats.objective_proposals_made = 3
        agent.stats.objective_proposals_approved = 1
        assert agent.stats.objective_proposals_made == 3
        assert agent.stats.objective_proposals_approved == 1

    def test_generate_candidates_returns_list(self):
        agent = Layer2Agent()
        candidates = agent._generate_objective_candidates()
        assert len(candidates) > 0
        assert all(c.is_valid() for c in candidates)

    def test_generate_candidates_not_all_identical(self):
        agent = Layer2Agent()
        candidates = agent._generate_objective_candidates()
        lambda_progs = [c.lambda_prog for c in candidates]
        assert len(set(lambda_progs)) > 1  # at least some variation