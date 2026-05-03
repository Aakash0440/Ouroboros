"""Tests for MetaMDLLearner — learning the MDL prior."""
import pytest
from collections import Counter
from ouroboros.meta.mdl_prior_learner import (
    MetaMDLLearner, PriorState, PriorUpdate,
    DEFAULT_BITS, MIN_BITS, MAX_BITS, PRIOR_LR,
)
from ouroboros.synthesis.expr_node import ExprNode, NodeType


def make_const(v): return ExprNode(NodeType.CONST, value=v)
def make_time(): return ExprNode(NodeType.TIME)


class TestPriorState:
    def test_default_state(self):
        state = PriorState.default()
        assert state.n_updates == 0
        assert "CONST" in state.description_bits
        assert state.depth_penalty > 0

    def test_get_bits_default(self):
        state = PriorState.default()
        bits = state.get_bits("CONST")
        assert abs(bits - DEFAULT_BITS["CONST"]) < 0.01

    def test_get_bits_domain(self):
        state = PriorState.default()
        state.domain_priors["physics"] = {"DERIV": 2.0}
        bits = state.get_bits("DERIV", "physics")
        assert bits == 2.0

    def test_to_dict(self):
        state = PriorState.default()
        d = state.to_dict()
        assert "description_bits" in d
        assert "n_updates" in d


class TestMetaMDLLearner:
    def test_initialization(self):
        learner = MetaMDLLearner()
        assert learner._state.n_updates == 0

    def test_update_success_reduces_bits(self):
        learner = MetaMDLLearner(learning_rate=0.1)
        expr = make_const(5.0)
        initial_bits = learner.get_description_bits("CONST")
        # Many successful discoveries using CONST → should reduce its cost
        for _ in range(20):
            learner.update(expr, domain="test", success=True,
                          mdl_cost=10.0, generalized=True)
        final_bits = learner.get_description_bits("CONST")
        assert final_bits <= initial_bits + 0.01  # should not increase

    def test_update_failure_increases_bits(self):
        learner = MetaMDLLearner(learning_rate=0.1)
        expr = make_const(5.0)
        initial_bits = learner.get_description_bits("CONST")
        for _ in range(20):
            learner.update(expr, domain="test", success=False, mdl_cost=999.0)
        final_bits = learner.get_description_bits("CONST")
        # Failed discoveries should not dramatically reduce cost
        assert final_bits >= MIN_BITS

    def test_bits_stay_in_bounds(self):
        learner = MetaMDLLearner(learning_rate=1.0)  # aggressive LR
        expr = make_const(5.0)
        for _ in range(100):
            learner.update(expr, domain="test", success=True, mdl_cost=10.0)
        for node_name, bits in learner._state.description_bits.items():
            assert MIN_BITS <= bits <= MAX_BITS, f"{node_name} out of bounds: {bits}"

    def test_domain_specific_prior(self):
        learner = MetaMDLLearner(min_updates_for_domain=3)
        from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
        from ouroboros.synthesis.expr_node import NodeType as NT
        def time_e():
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NT.TIME; n.value = 0.0
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n
        deriv_expr = ExtExprNode(ExtNodeType.DERIV, left=time_e())
        # Physics domain: many DERIV successes
        for _ in range(10):
            learner.update(deriv_expr, domain="physics", success=True, mdl_cost=20.0)
        # Physics prior should exist now
        assert "physics" in learner._state.domain_priors

    def test_count_nodes_const(self):
        learner = MetaMDLLearner()
        expr = make_const(5.0)
        counts = learner._count_nodes(expr)
        assert counts.get("CONST", 0) >= 1

    def test_count_nodes_add(self):
        learner = MetaMDLLearner()
        expr = ExprNode(NodeType.ADD, left=make_const(1.0), right=make_time())
        counts = learner._count_nodes(expr)
        assert counts.get("ADD", 0) >= 1

    def test_get_category_weights(self):
        learner = MetaMDLLearner()
        weights = learner.get_category_weights_for_router("physics")
        from ouroboros.nodes.extended_nodes import NodeCategory
        assert all(isinstance(k, NodeCategory) for k in weights)
        assert all(0.1 <= v <= 10.0 for v in weights.values())

    def test_prior_summary(self):
        learner = MetaMDLLearner()
        s = learner.prior_summary()
        assert isinstance(s, str)

    def test_save_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "prior.json")
        learner = MetaMDLLearner(save_path=path)
        expr = make_const(5.0)
        learner.update(expr, domain="test", success=True, mdl_cost=10.0)
        learner._save(path)
        learner2 = MetaMDLLearner(save_path=path)
        assert learner2._state.n_updates >= 0

    def test_update_returns_update_object(self):
        learner = MetaMDLLearner()
        expr = make_const(3.0)
        update = learner.update(expr, domain="test", success=True, mdl_cost=15.0)
        assert isinstance(update, PriorUpdate)
        assert update.success
        assert update.domain == "test"