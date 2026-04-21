"""Tests for cross-agent program sharing and collaborative proof building."""
import pytest
import copy
from ouroboros.agents.expression_fragment import (
    ExpressionFragment, HoleSpec, HoleType,
    create_fragment_from_expr, fill_hole_in_fragment,
    HOLE_SENTINEL, _find_const_paths, _get_node_at_path,
)
from ouroboros.agents.fragment_completer import FragmentCompleter, CompletionResult
from ouroboros.agents.collaborative_proof import (
    CollaborationMessage, CollaborativeAgent,
    CollaborativeSessionResult, CollaborativeProofSession,
)
from ouroboros.synthesis.expr_node import ExprNode, NodeType, build_linear_modular


class TestExpressionFragment:
    def _make_expr(self) -> ExprNode:
        """Build (3 * t + 1) % 7 as an ExprNode."""
        return build_linear_modular(slope=3, intercept=1, modulus=7)

    def test_create_fragment_one_hole(self):
        expr = self._make_expr()
        fragment = create_fragment_from_expr(expr, n_holes=1)
        assert fragment.n_holes == 1
        assert not fragment.is_complete

    def test_create_fragment_zero_holes(self):
        expr = self._make_expr()
        fragment = create_fragment_from_expr(expr, n_holes=0)
        assert fragment.n_holes == 0
        assert fragment.is_complete

    def test_hole_sentinel_in_tree(self):
        expr = self._make_expr()
        fragment = create_fragment_from_expr(expr, n_holes=1)
        # Find const nodes in the fragment — at least one should be sentinel
        const_values = []
        def collect_consts(node):
            if node.node_type == NodeType.CONST:
                const_values.append(node.value)
            if node.left: collect_consts(node.left)
            if node.right: collect_consts(node.right)
        collect_consts(fragment.root)
        assert HOLE_SENTINEL in const_values

    def test_fill_hole_restores_value(self):
        expr = self._make_expr()
        fragment = create_fragment_from_expr(expr, n_holes=1)
        hole = fragment.holes[0]
        filled = fill_hole_in_fragment(fragment, hole_index=0, fill_value=42)
        # The filled node should have value 42
        node = _get_node_at_path(filled, hole.position_path)
        assert node is not None
        assert node.value == 42

    def test_fragment_description_is_string(self):
        expr = self._make_expr()
        fragment = create_fragment_from_expr(expr, n_holes=1)
        fragment.creator_agent = "AGENT_00"
        fragment.environment_name = "ModArith"
        s = fragment.description()
        assert isinstance(s, str) and len(s) > 0

    def test_find_const_paths_finds_consts(self):
        expr = build_linear_modular(slope=3, intercept=1, modulus=7)
        paths = []
        _find_const_paths(expr, [], paths)
        assert len(paths) >= 2  # At least slope and modulus are CONST

    def test_get_node_at_empty_path_is_root(self):
        expr = ExprNode(NodeType.CONST, value=5)
        result = _get_node_at_path(expr, [])
        assert result is expr

    def test_get_node_at_left_path(self):
        left = ExprNode(NodeType.CONST, value=3)
        right = ExprNode(NodeType.CONST, value=7)
        root = ExprNode(NodeType.ADD, left=left, right=right)
        assert _get_node_at_path(root, [0]) is left
        assert _get_node_at_path(root, [1]) is right


class TestFragmentCompleter:
    def _make_simple_fragment(self) -> tuple:
        """Create a simple fragment: CONST(?) and observations matching CONST(5)."""
        expr = ExprNode(NodeType.CONST, value=5)
        fragment = create_fragment_from_expr(expr, n_holes=1)
        observations = [5] * 100   # perfect prediction if CONST(5)
        return fragment, observations

    def test_complete_single_hole_finds_correct_constant(self):
        fragment, obs = self._make_simple_fragment()
        completer = FragmentCompleter(completing_agent="B", const_range=15)
        result = completer.complete(fragment, obs)
        # Should find CONST(5) as the best fill
        assert result.is_valid
        assert 5 in result.filled_values

    def test_complete_returns_completion_result(self):
        fragment, obs = self._make_simple_fragment()
        completer = FragmentCompleter(completing_agent="B", const_range=10)
        result = completer.complete(fragment, obs)
        assert isinstance(result, CompletionResult)

    def test_already_complete_fragment(self):
        expr = ExprNode(NodeType.CONST, value=5)
        fragment = ExpressionFragment(
            root=expr, holes=[],
            creator_agent="A", environment_name="T",
            partial_mdl_cost=50.0,
        )
        completer = FragmentCompleter("B")
        result = completer.complete(fragment, [5]*50)
        assert result.is_valid
        assert result.filled_values == []

    def test_completion_mdl_cost_positive(self):
        fragment, obs = self._make_simple_fragment()
        completer = FragmentCompleter("B", const_range=10)
        result = completer.complete(fragment, obs)
        if result.is_valid:
            assert result.mdl_cost > 0

    def test_completing_agent_recorded(self):
        fragment, obs = self._make_simple_fragment()
        completer = FragmentCompleter("AGENT_CHARLIE", const_range=10)
        result = completer.complete(fragment, obs)
        assert result.completing_agent == "AGENT_CHARLIE"


class TestCollaborationMessage:
    def test_fragment_message_fields(self):
        msg = CollaborationMessage(
            sender_id="A", receiver_id="B",
            message_type="FRAGMENT",
            fragment=None,
            round_sent=3,
        )
        assert msg.message_type == "FRAGMENT"
        assert msg.round_sent == 3
        assert msg.sender_id == "A"

    def test_adopt_message_has_expr(self):
        expr = ExprNode(NodeType.CONST, value=7)
        msg = CollaborationMessage(
            sender_id="SESSION", receiver_id="ALL",
            message_type="ADOPT",
            completed_expr=expr,
            final_mdl_cost=42.0,
        )
        assert msg.completed_expr is expr
        assert msg.final_mdl_cost == 42.0


class TestCollaborativeAgent:
    def _make_agent(self, agent_id: str, seed: int = 42) -> CollaborativeAgent:
        return CollaborativeAgent(
            agent_id=agent_id,
            const_range=20,
            beam_width=10,
            max_depth=3,
            random_seed=seed,
        )

    def test_initial_state(self):
        agent = self._make_agent("A")
        assert agent.best_expr is None
        assert agent.best_cost == float('inf')

    def test_receive_message_queues(self):
        agent = self._make_agent("B")
        msg = CollaborationMessage("A", "B", "FRAGMENT")
        agent.receive_message(msg)
        assert len(agent._messages_received) == 1

    def test_adopt_message_updates_best(self):
        agent = self._make_agent("B")
        expr = ExprNode(NodeType.CONST, value=5)
        msg = CollaborationMessage(
            sender_id="A", receiver_id="B",
            message_type="ADOPT",
            completed_expr=expr,
            final_mdl_cost=30.0,
        )
        agent.receive_message(msg)
        agent.process_messages([5] * 50)
        # Agent should have adopted the expression
        assert agent.best_cost == 30.0

    def test_adopt_only_if_better(self):
        agent = self._make_agent("B")
        agent.best_cost = 20.0   # already has a good expression
        expr = ExprNode(NodeType.CONST, value=5)
        msg = CollaborationMessage(
            sender_id="A", receiver_id="B",
            message_type="ADOPT",
            completed_expr=expr,
            final_mdl_cost=50.0,   # worse than current
        )
        agent.receive_message(msg)
        agent.process_messages([5] * 50)
        assert agent.best_cost == 20.0   # unchanged


class TestCollaborativeProofSession:
    def test_session_result_type(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        agents = [
            CollaborativeAgent(f"A{i}", const_range=15, beam_width=8, random_seed=42+i)
            for i in range(3)
        ]
        session = CollaborativeProofSession(agents, n_rounds=2, stream_length=100)
        env = ModularArithmeticEnv(modulus=5, slope=2, intercept=1)
        result = session.run(env, verbose=False)
        assert isinstance(result, CollaborativeSessionResult)
        assert result.n_rounds == 2

    def test_session_has_agent_count(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        agents = [
            CollaborativeAgent(f"B{i}", const_range=10, beam_width=5, random_seed=i)
            for i in range(4)
        ]
        session = CollaborativeProofSession(agents, n_rounds=1, stream_length=50)
        env = ModularArithmeticEnv(modulus=3, slope=1, intercept=0)
        result = session.run(env, verbose=False)
        assert len(result.participating_agents) == 4

    def test_solo_best_recorded(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        agents = [CollaborativeAgent("X", const_range=10, beam_width=5, random_seed=1)]
        session = CollaborativeProofSession(agents, n_rounds=1, stream_length=50)
        env = ModularArithmeticEnv(modulus=3, slope=1, intercept=0)
        result = session.run(env, verbose=False)
        assert result.solo_best_cost >= 0  # could be inf if nothing found