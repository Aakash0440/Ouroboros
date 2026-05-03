"""Tests for CausalMDLScorer and causal MDL integration."""
import pytest
import math
from ouroboros.causal.causal_mdl import CausalMDLScorer, CausalConsistencyScore
from ouroboros.causal.causal_graph import CausalGraph, CausalEdge, CausalVariable
from ouroboros.synthesis.expr_node import ExprNode, NodeType


def make_const(v): return ExprNode(NodeType.CONST, value=v)
def make_var(name): return CausalVariable(name=name, var_type="observed")


class TestCausalMDLScorer:
    def test_score_returns_result(self):
        scorer = CausalMDLScorer(causal_weight=0.1)
        expr = make_const(5.0)
        obs = [5.0] * 50
        result = scorer.score(expr, obs)
        assert isinstance(result, CausalConsistencyScore)

    def test_base_cost_finite(self):
        scorer = CausalMDLScorer()
        expr = make_const(3.0)
        obs = [3.0] * 50
        result = scorer.score(expr, obs)
        assert math.isfinite(result.base_mdl_cost)

    def test_causal_bonus_with_graph(self):
        scorer = CausalMDLScorer(causal_weight=0.1)
        # Build a graph with acceleration edge
        g = CausalGraph()
        vpos = make_var("position")
        vacc = make_var("acceleration")
        g.add_variable(vpos); g.add_variable(vacc)
        g.add_edge(CausalEdge(vpos, vacc, lag=0))

        obs = [math.cos(0.3*t)*10 for t in range(50)]
        from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
        from ouroboros.synthesis.expr_node import NodeType as NT
        def time_e():
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NT.TIME; n.value = 0.0
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n
        # Expression with DERIV2 should get causal bonus
        deriv2_expr = ExtExprNode(ExtNodeType.DERIV2, left=time_e())
        result = scorer.score(deriv2_expr, obs, causal_graph=g)
        assert isinstance(result, CausalConsistencyScore)
        # DERIV2 expression with acceleration graph → should have reasons
        assert len(result.consistency_reasons) > 0

    def test_adjusted_cost_lower_with_bonus(self):
        scorer = CausalMDLScorer(causal_weight=0.5)
        g = CausalGraph()
        g.add_variable(make_var("acc"))
        g.add_variable(make_var("pos"))
        g.add_edge(CausalEdge(make_var("pos"), make_var("acc"), lag=0))

        from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType
        from ouroboros.synthesis.expr_node import NodeType as NT
        def time_e():
            n = ExtExprNode.__new__(ExtExprNode)
            n.node_type = NT.TIME; n.value = 0.0
            n.lag = 1; n.state_key = 0; n.window = 10
            n.left = n.right = n.third = None; n._cache = {}
            return n
        deriv2 = ExtExprNode(ExtNodeType.DERIV2, left=time_e())
        obs = [math.sin(t*0.3)*10 for t in range(50)]
        result = scorer.score(deriv2, obs, causal_graph=g)
        # adjusted should be ≤ base (bonus reduces cost)
        assert result.adjusted_mdl_cost <= result.base_mdl_cost + 0.01

    def test_improvement_pct(self):
        result = CausalConsistencyScore("test", 100.0, 15.0, 85.0, ["bonus reason"])
        assert abs(result.improvement_pct - 15.0) < 0.01

    def test_no_graph_no_bonus(self):
        scorer = CausalMDLScorer(causal_weight=0.1)
        expr = make_const(5.0)
        result = scorer.score(expr, [5.0]*50, causal_graph=None)
        assert result.causal_bonus == 0.0
        assert result.adjusted_mdl_cost == pytest.approx(result.base_mdl_cost)
