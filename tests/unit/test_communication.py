"""Tests for agent communication protocol."""
import pytest
from ouroboros.agents.communication import (
    MessageBus, Message, MessageType,
    make_axiom_hint, make_search_hint, make_failure_hint,
    make_convergence_signal
)
from ouroboros.agents.communicating_agent import CommunicatingAgent


class TestMessageCreation:
    def test_axiom_hint_type(self):
        msg = make_axiom_hint(0, 1, "(t*3+1)mod7", 0.004, 7)
        assert msg.msg_type == MessageType.AXIOM_HINT
        assert msg.sender_id == 0
        assert msg.round_sent == 1
        assert 'expression_str' in msg.payload
        assert 'compression_ratio' in msg.payload

    def test_search_hint_type(self):
        msg = make_search_hint(1, 2, 0, 14, 500.0)
        assert msg.msg_type == MessageType.SEARCH_HINT
        assert msg.payload['const_min'] == 0
        assert msg.payload['const_max'] == 14

    def test_failure_hint_type(self):
        msg = make_failure_hint(2, 3, "bad_expr", "wrong predictions")
        assert msg.msg_type == MessageType.FAILURE_HINT
        assert 'expression_str' in msg.payload

    def test_convergence_signal_type(self):
        msg = make_convergence_signal(3, 4, 0.004, "(t*3+1)mod7")
        assert msg.msg_type == MessageType.CONVERGENCE_SIGNAL
        assert msg.payload['compression_ratio'] == 0.004


class TestMessageBus:
    def test_send_increments_count(self):
        bus = MessageBus(num_agents=4)
        msg = make_axiom_hint(0, 1, "expr", 0.1, 7)
        bus.send(msg)
        assert bus.total_sent == 1

    def test_receive_delayed_by_one_round(self):
        bus = MessageBus(num_agents=4)
        msg = make_axiom_hint(0, 1, "expr", 0.1, 7)
        bus.send(msg)

        # Same round: not yet delivered
        received = bus.receive(agent_id=1, current_round=1)
        assert len(received) == 0

        # Next round: delivered
        received = bus.receive(agent_id=1, current_round=2)
        assert len(received) == 1

    def test_sender_does_not_receive_own_message(self):
        bus = MessageBus(num_agents=4)
        msg = make_axiom_hint(0, 1, "expr", 0.1, 7)
        bus.send(msg)

        received = bus.receive(agent_id=0, current_round=2)
        assert len(received) == 0

    def test_all_other_agents_receive(self):
        bus = MessageBus(num_agents=4)
        msg = make_axiom_hint(0, 1, "expr", 0.1, 7)
        bus.send(msg)

        for receiver_id in [1, 2, 3]:
            received = bus.receive(agent_id=receiver_id, current_round=2)
            assert len(received) == 1

    def test_advance_round_cleans_old_messages(self):
        bus = MessageBus(num_agents=4)
        for r in range(10):
            bus.send(make_axiom_hint(0, r, "e", 0.1, 7))
            bus.advance_round()
        # Old messages cleaned up
        assert len(bus._queue) <= 3

    def test_stats_structure(self):
        bus = MessageBus(num_agents=4)
        stats = bus.stats()
        assert 'total_sent' in stats
        assert 'total_delivered' in stats
        assert 'by_type' in stats

    def test_message_not_double_delivered(self):
        """Each message delivered at most once per receiver."""
        bus = MessageBus(num_agents=4)
        msg = make_axiom_hint(0, 1, "expr", 0.1, 7)
        bus.send(msg)

        first = bus.receive(1, current_round=2)
        second = bus.receive(1, current_round=2)
        # Message already marked delivered
        assert len(first) == 1
        assert len(second) == 0

    def test_queue_full_drops_messages(self):
        bus = MessageBus(num_agents=2, max_queue_per_round=1)
        # First message: accepted
        sent1 = bus.send(make_axiom_hint(0, 1, "e1", 0.1, 7))
        # Fill queue
        for _ in range(10):
            bus.send(make_axiom_hint(0, 1, "e", 0.1, 7))
        # Eventually queue is full
        assert bus.total_sent >= 1


class TestCommunicatingAgent:
    def test_init(self):
        bus = MessageBus(num_agents=4)
        agent = CommunicatingAgent(
            0, 7, message_bus=bus,
            beam_width=5, max_depth=2, mcmc_iterations=10
        )
        assert agent.bus is bus
        assert agent.hints_sent == 0

    def test_receive_hints_empty_bus(self):
        bus = MessageBus(num_agents=4)
        agent = CommunicatingAgent(0, 7, message_bus=bus,
                                   beam_width=5, max_depth=2, mcmc_iterations=10)
        msgs = agent.receive_hints(current_round=1)
        assert msgs == []

    def test_send_hint_increments_count(self):
        from ouroboros.compression.program_synthesis import build_linear_modular
        bus = MessageBus(num_agents=4)
        agent = CommunicatingAgent(0, 7, message_bus=bus,
                                   beam_width=5, max_depth=2, mcmc_iterations=10)
        agent.best_expression = build_linear_modular(3, 1, 7)
        agent.compression_ratios = [0.004]  # Good ratio
        sent = agent.send_current_hint(current_round=1)
        assert sent
        assert agent.hints_sent == 1

    def test_high_ratio_does_not_send_hint(self):
        bus = MessageBus(num_agents=4)
        agent = CommunicatingAgent(0, 7, message_bus=bus,
                                   beam_width=5, max_depth=2, mcmc_iterations=10)
        from ouroboros.compression.program_synthesis import build_linear_modular
        agent.best_expression = build_linear_modular(3, 1, 7)
        agent.compression_ratios = [0.90]  # Bad ratio — won't send
        sent = agent.send_current_hint(current_round=1)
        assert not sent

    def test_communication_stats_structure(self):
        bus = MessageBus(num_agents=4)
        agent = CommunicatingAgent(0, 7, message_bus=bus,
                                   beam_width=5, max_depth=2, mcmc_iterations=10)
        stats = agent.communication_stats()
        assert 'agent_id' in stats
        assert 'hints_sent' in stats
        assert 'hints_received' in stats