"""Unit tests for the proof market — commit-reveal and market mechanics."""
import pytest
from ouroboros.proof_market.commit_reveal import (
    make_commitment, make_null_commitment, verify_reveal,
    is_null_commitment, RoundState
)
from ouroboros.proof_market.counterexample import (
    CounterexampleResult, CounterexampleSearcher
)
from ouroboros.proof_market.market import ProofMarket, MarketAgent
from ouroboros.compression.program_synthesis import (
    build_linear_modular, C, T, MOD
)


class TestCommitReveal:
    def test_valid_reveal_returns_true(self):
        ce = b"valid counterexample data"
        commitment = make_commitment(0, ce, "round_001")
        commitment.revealed = True
        assert verify_reveal(commitment)

    def test_tampered_reveal_returns_false(self):
        commitment = make_commitment(0, b"original", "round_001")
        commitment.counterexample = b"tampered"
        commitment.revealed = True
        assert not verify_reveal(commitment)

    def test_hash_is_64_chars(self):
        c = make_commitment(0, b"data", "round_001")
        assert len(c.commitment_hash) == 64

    def test_different_agents_different_hashes(self):
        ce = b"same_data"
        c1 = make_commitment(0, ce, "r1")
        c2 = make_commitment(1, ce, "r1")
        assert c1.commitment_hash != c2.commitment_hash

    def test_same_agent_different_rounds_different_hashes(self):
        ce = b"same_data"
        c1 = make_commitment(0, ce, "round_001")
        c2 = make_commitment(0, ce, "round_002")
        # Salt is random so hashes differ even with same inputs
        assert c1.commitment_hash != c2.commitment_hash

    def test_null_commitment_verifies(self):
        c = make_null_commitment(0, "round_001")
        c.revealed = True
        assert verify_reveal(c)

    def test_is_null_commitment(self):
        null = make_null_commitment(0, "r1")
        normal = make_commitment(0, b"real_data", "r1")
        assert is_null_commitment(null)
        assert not is_null_commitment(normal)

    def test_unrevealed_commitment_verify_false(self):
        c = make_commitment(0, b"data", "r1")
        assert not verify_reveal(c)   # not revealed yet

    def test_public_view_hides_counterexample(self):
        c = make_commitment(0, b"secret_CE", "r1")
        view = c.public_view()
        assert 'counterexample' not in view
        assert 'commitment_hash' in view

    def test_round_state_lifecycle(self):
        rs = RoundState("r1", 0, b"proposal", "prop description")
        assert rs.phase == 'OPEN'
        rs.advance_to_commit()
        assert rs.phase == 'COMMIT'
        rs.advance_to_reveal()
        assert rs.phase == 'REVEAL'
        rs.advance_to_verify()
        assert rs.phase == 'VERIFY'
        rs.resolve(True, "approved")
        assert rs.phase == 'RESOLVED'
        assert rs.modification_approved


class TestCounterexampleResult:
    def test_to_bytes_deterministic(self):
        expr = build_linear_modular(3, 1, 7)
        ce = CounterexampleResult(expr, 100.0, 200.0, True, 0)
        b1 = ce.to_bytes()
        b2 = ce.to_bytes()
        assert b1 == b2

    def test_null_result(self):
        ce = CounterexampleResult.null_result(agent_id=2, proposal_cost=500.0)
        assert not ce.is_valid_counterexample
        assert ce.expression is None
        assert ce.agent_id == 2

    def test_valid_ce_beats_proposal(self):
        ce = CounterexampleResult(
            expression=build_linear_modular(3, 1, 7),
            ce_mdl_cost=50.0,
            proposal_mdl_cost=500.0,
            is_valid_counterexample=True,
            agent_id=0
        )
        assert ce.is_valid_counterexample
        assert ce.ce_mdl_cost < ce.proposal_mdl_cost

    def test_to_bytes_is_bytes(self):
        ce = CounterexampleResult.null_result(0, 100.0)
        assert isinstance(ce.to_bytes(), bytes)


class TestProofMarket:
    def setup_method(self):
        self.market = ProofMarket(num_agents=6, starting_credit=100.0)
        self.test_seq = [(3*t+1) % 7 for t in range(100)]
        self.current = MOD(T(), C(7))
        self.proposed = build_linear_modular(3, 1, 7)

    def test_propose_deducts_bounty(self):
        self.market.propose(
            0, self.current, self.proposed, self.test_seq, 7, bounty=10.0
        )
        assert self.market.agents[0].credit == 90.0
        # Clean up
        self.market.current_round = None
        self.market.current_proposal = None
        self.market.agents[0].credit = 100.0

    def test_insufficient_credit_raises(self):
        self.market.agents[0].credit = 5.0
        with pytest.raises(ValueError):
            self.market.propose(0, self.current, self.proposed, self.test_seq, 7, bounty=10.0)

    def test_second_propose_without_resolution_raises(self):
        self.market.propose(0, self.current, self.proposed, self.test_seq, 7)
        with pytest.raises(RuntimeError):
            self.market.propose(1, self.current, self.proposed, self.test_seq, 7)

    def test_full_round_genuine_improvement(self):
        """Genuine improvement: no valid CEs → APPROVED."""
        ce_results = {
            aid: CounterexampleResult.null_result(aid, 200.0)
            for aid in range(1, 6)
        }
        approved = self.market.run_full_round(
            proposer_id=0,
            current_expr=self.current,
            proposed_expr=self.proposed,
            test_sequence=self.test_seq,
            alphabet_size=7,
            adversarial_agents=list(range(1, 6)),
            ce_results=ce_results,
            bounty=10.0
        )
        assert approved
        assert self.market.agents[0].credit > 100.0  # Got bounty back + bonus

    def test_full_round_bad_modification(self):
        """Bad modification: valid CEs found → REJECTED."""
        bad_proposed = C(3)
        ce_results = {
            aid: CounterexampleResult(
                expression=self.proposed,
                ce_mdl_cost=50.0,
                proposal_mdl_cost=500.0,
                is_valid_counterexample=True,
                agent_id=aid
            )
            for aid in range(1, 6)
        }
        approved = self.market.run_full_round(
            proposer_id=0,
            current_expr=self.current,
            proposed_expr=bad_proposed,
            test_sequence=self.test_seq,
            alphabet_size=7,
            adversarial_agents=list(range(1, 6)),
            ce_results=ce_results,
            bounty=10.0
        )
        assert not approved
        # CE finders should have earned bounty
        for aid in range(1, 6):
            assert self.market.agents[aid].credit > 100.0

    def test_invalid_reveal_penalized(self):
        """Agents with hash mismatch get penalized."""
        self.market.propose(0, self.current, self.proposed, self.test_seq, 7)
        null_ce = CounterexampleResult.null_result(1, 200.0)
        self.market.commit(1, null_ce)
        self.market.close_commit_phase()

        # Tamper with the commitment
        self.market.current_round.commitments[1].counterexample = b"tampered"
        self.market.reveal(1)

        assert self.market.agents[1].credit < 100.0
        assert self.market.agents[1].invalid_reveals == 1

    def test_credit_summary(self):
        summary = self.market.credit_summary()
        assert len(summary) == 6
        assert all(v == 100.0 for v in summary.values())

    def test_role_update_after_rounds(self):
        """Agents with many CE finds should become adversaries."""
        agent = self.market.agents[1]
        agent.adversary_score = 1.0
        agent.proposer_score = 0.1
        agent.update_role()
        assert agent.role == 'adversary'

    def test_market_summary_string(self):
        s = self.market.market_summary()
        assert isinstance(s, str)
        assert 'ProofMarket' in s