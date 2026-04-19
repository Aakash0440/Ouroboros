"""Tests for HyperparameterAgent and HyperparameterMarket."""
import pytest
import copy
from ouroboros.agents.hyperparameter_agent import (
    HyperparameterSet, HyperparameterProposal, HyperparameterAgent
)
from ouroboros.proof_market.hp_market import (
    HyperparameterMarket, HPMarketRound, HPCounterexample
)


class TestHyperparameterSet:
    def test_default_values(self):
        hp = HyperparameterSet()
        assert hp.beam_width == 25
        assert hp.mcmc_iterations == 200
        assert hp.const_range == 16
        assert hp.max_depth == 3
        assert hp.max_lag == 3

    def test_clamp_keeps_valid_values(self):
        hp = HyperparameterSet(beam_width=30)
        clamped = hp.clamp()
        assert clamped.beam_width == 30

    def test_clamp_enforces_upper_bound(self):
        hp = HyperparameterSet(beam_width=9999)
        clamped = hp.clamp()
        assert clamped.beam_width == 100  # upper bound

    def test_clamp_enforces_lower_bound(self):
        hp = HyperparameterSet(beam_width=0)
        clamped = hp.clamp()
        assert clamped.beam_width == 1  # lower bound

    def test_to_dict_structure(self):
        hp = HyperparameterSet()
        d = hp.to_dict()
        assert 'beam_width' in d
        assert 'mcmc_iterations' in d
        assert 'const_range' in d
        assert 'max_depth' in d
        assert 'max_lag' in d

    def test_compute_cost_increases_with_params(self):
        hp_small = HyperparameterSet(beam_width=5, mcmc_iterations=10, max_depth=1)
        hp_large = HyperparameterSet(beam_width=50, mcmc_iterations=500, max_depth=5)
        assert hp_large.compute_cost() > hp_small.compute_cost()

    def test_description_bits_positive(self):
        hp = HyperparameterSet()
        assert hp.description_bits() > 0

    def test_equality(self):
        hp1 = HyperparameterSet(beam_width=30)
        hp2 = HyperparameterSet(beam_width=30)
        hp3 = HyperparameterSet(beam_width=25)
        assert hp1 == hp2
        assert hp1 != hp3

    def test_repr_contains_key_params(self):
        hp = HyperparameterSet(beam_width=30)
        r = repr(hp)
        assert 'bw=30' in r


class TestHyperparameterProposal:
    def setup_method(self):
        self.proposal = HyperparameterProposal(
            agent_id=0,
            current_hp=HyperparameterSet(beam_width=25),
            proposed_hp=HyperparameterSet(beam_width=30),
            current_best_cost=500.0,
            proposed_best_cost=450.0,
            improvement_bits=50.0,
            validation_data=list(range(100)),
            alphabet_size=7,
            changed_param='beam_width',
            change_direction='increase'
        )

    def test_is_improvement_positive(self):
        assert self.proposal.is_improvement()

    def test_is_improvement_negative(self):
        bad = HyperparameterProposal(
            0, HyperparameterSet(), HyperparameterSet(),
            current_best_cost=400.0,
            proposed_best_cost=500.0,
            improvement_bits=-100.0,
            validation_data=[],
            alphabet_size=7
        )
        assert not bad.is_improvement()

    def test_to_market_proposal_returns_pair(self):
        result = self.proposal.to_market_proposal()
        assert result is not None
        curr_expr, prop_expr = result
        assert curr_expr is not None
        assert prop_expr is not None


class TestHyperparameterAgent:
    def test_init_sets_hp(self):
        hp = HyperparameterSet(beam_width=30)
        agent = HyperparameterAgent(0, 7, initial_hp=hp,
                                     beam_width=30, max_depth=2,
                                     mcmc_iterations=50)
        assert agent.current_hp.beam_width == 30

    def test_candidate_hp_sets_non_empty(self):
        agent = HyperparameterAgent(0, 7, beam_width=20, max_depth=2,
                                     mcmc_iterations=50)
        candidates = agent._candidate_hp_sets()
        assert len(candidates) > 0

    def test_candidate_hp_covers_all_params(self):
        agent = HyperparameterAgent(0, 7, beam_width=20, max_depth=2,
                                     mcmc_iterations=50)
        candidates = agent._candidate_hp_sets()
        params = {c[0] for c in candidates}
        assert 'beam_width' in params
        assert 'mcmc_iterations' in params
        assert 'const_range' in params

    def test_generate_hp_proposal_returns_none_too_early(self):
        agent = HyperparameterAgent(0, 7, hp_mod_frequency=5,
                                     beam_width=10, max_depth=2,
                                     mcmc_iterations=20)
        data = [(3*t+1)%7 for t in range(100)]
        # Only 1 round — below frequency threshold of 5
        proposal = agent.generate_hp_proposal(data)
        assert proposal is None

    def test_apply_hp_modification_updates(self):
        agent = HyperparameterAgent(0, 7, beam_width=20, max_depth=2,
                                     mcmc_iterations=50)
        old_bw = agent.current_hp.beam_width
        proposal = HyperparameterProposal(
            0,
            HyperparameterSet(beam_width=old_bw),
            HyperparameterSet(beam_width=old_bw + 5),
            500.0, 450.0, 50.0,
            list(range(50)), 7,
            'beam_width', 'increase'
        )
        agent.apply_hp_modification(proposal, round_num=5)
        assert agent.current_hp.beam_width == old_bw + 5
        assert agent.hp_approved == 1
        assert len(agent.hp_history) == 1

    def test_hp_improvement_score_zero_initially(self):
        agent = HyperparameterAgent(0, 7, beam_width=10, max_depth=2,
                                     mcmc_iterations=20)
        assert agent.hp_improvement_score() == 0.0

    def test_hp_improvement_score_increases(self):
        agent = HyperparameterAgent(0, 7, beam_width=10, max_depth=2,
                                     mcmc_iterations=20)
        proposal = HyperparameterProposal(
            0, HyperparameterSet(), HyperparameterSet(beam_width=15),
            100.0, 50.0, 50.0, list(range(50)), 7,
            'beam_width', 'increase'
        )
        agent.apply_hp_modification(proposal, 1)
        assert agent.hp_improvement_score() == 50.0


class TestHyperparameterMarket:
    def setup_method(self):
        self.agents = [
            HyperparameterAgent(
                i, 7, initial_hp=HyperparameterSet(beam_width=20+i*2),
                beam_width=20+i*2, max_depth=2, mcmc_iterations=50
            )
            for i in range(4)
        ]
        self.market = HyperparameterMarket(self.agents)

    def test_init_credits(self):
        for aid in range(4):
            assert self.market.credits[aid] == 100.0

    def test_run_hp_round_deducts_bounty(self):
        proposal = HyperparameterProposal(
            0,
            HyperparameterSet(beam_width=20),
            HyperparameterSet(beam_width=25),
            500.0, 450.0, 50.0,
            [(3*t+1)%7 for t in range(100)], 7,
            'beam_width', 'increase'
        )
        approved, stats = self.market.run_hp_round(proposal, bounty=5.0)
        # Proposer should have been debited or credited back
        assert self.market.credits[0] != 100.0

    def test_market_summary_string(self):
        s = self.market.market_summary()
        assert 'HyperparameterMarket' in s