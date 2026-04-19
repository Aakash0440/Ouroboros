"""
Agent Communication Protocol for OUROBOROS.

Design principles:
    1. DELAYED delivery: messages sent in round N are received in round N+1.
       This prevents coordination during the commit phase.

    2. TYPED messages: agents send structured hints, not free-form text.
       Types: AxiomHint, SearchHint, FailureHint, ConvergenceSignal

    3. OPTIONAL reception: agents can IGNORE messages.
       The system tests whether communication helps; agents aren't forced to use hints.

    4. CRYPTOGRAPHIC safety: message content is separate from commit-reveal.
       Sending a message does NOT affect your commitment hash.
       Lying in a message is detectable by comparing with your eventual commit.

Message types:
    AxiomHint: "I found an expression with good compression — try this region"
        Carries: expression_string, compression_ratio, scale
        Use: receiver can use this as a beam search prior

    SearchHint: "I've explored const_range 0..10 and found nothing good"
        Carries: searched_range, best_cost_found
        Use: receiver can focus on un-searched regions

    FailureHint: "I tried this expression and it failed — don't bother"
        Carries: expression_string, failure_reason
        Use: receiver can skip this expression in beam search

    ConvergenceSignal: "I think we've converged — my ratio is X"
        Carries: compression_ratio, expression_string
        Use: measure whether society is converging
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum, auto
import time
from collections import defaultdict


class MessageType(Enum):
    AXIOM_HINT        = auto()  # Found a good expression
    SEARCH_HINT       = auto()  # Explored a region, nothing good
    FAILURE_HINT      = auto()  # This expression failed
    CONVERGENCE_SIGNAL = auto() # Agent thinks society has converged


@dataclass
class Message:
    """
    A typed message from one agent to all others.

    All messages are broadcast (no private messaging —
    private messaging could undermine the commit-reveal protocol).

    Fields:
        sender_id: Agent who sent this
        msg_type: What kind of hint
        round_sent: Which round this was sent in
        payload: Type-specific data (dict)
        delivered: Whether receivers have seen this yet
    """
    sender_id: int
    msg_type: MessageType
    round_sent: int
    payload: Dict[str, Any] = field(default_factory=dict)
    delivered: bool = False
    delivered_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (f"Message(from={self.sender_id}, "
                f"type={self.msg_type.name}, "
                f"round={self.round_sent}, "
                f"payload={list(self.payload.keys())})")


def make_axiom_hint(
    sender_id: int,
    round_sent: int,
    expression_str: str,
    compression_ratio: float,
    alphabet_size: int,
    scale: int = 1
) -> Message:
    """Broadcast: 'I found this expression with this compression ratio.'"""
    return Message(
        sender_id=sender_id,
        msg_type=MessageType.AXIOM_HINT,
        round_sent=round_sent,
        payload={
            'expression_str': expression_str,
            'compression_ratio': compression_ratio,
            'alphabet_size': alphabet_size,
            'scale': scale,
        }
    )


def make_search_hint(
    sender_id: int,
    round_sent: int,
    searched_const_min: int,
    searched_const_max: int,
    best_cost: float
) -> Message:
    """Broadcast: 'I searched constants [min..max] and best cost was X.'"""
    return Message(
        sender_id=sender_id,
        msg_type=MessageType.SEARCH_HINT,
        round_sent=round_sent,
        payload={
            'const_min': searched_const_min,
            'const_max': searched_const_max,
            'best_cost': best_cost,
        }
    )


def make_failure_hint(
    sender_id: int,
    round_sent: int,
    expression_str: str,
    failure_reason: str
) -> Message:
    """Broadcast: 'This expression failed — don't bother trying it.'"""
    return Message(
        sender_id=sender_id,
        msg_type=MessageType.FAILURE_HINT,
        round_sent=round_sent,
        payload={
            'expression_str': expression_str,
            'failure_reason': failure_reason,
        }
    )


def make_convergence_signal(
    sender_id: int,
    round_sent: int,
    compression_ratio: float,
    expression_str: str
) -> Message:
    """Broadcast: 'My current ratio is X — I think we're converging.'"""
    return Message(
        sender_id=sender_id,
        msg_type=MessageType.CONVERGENCE_SIGNAL,
        round_sent=round_sent,
        payload={
            'compression_ratio': compression_ratio,
            'expression_str': expression_str,
        }
    )


class MessageBus:
    """
    Delayed broadcast message bus for agent society.

    Messages sent in round N are delivered in round N+1.
    This prevents commit-phase coordination.

    Usage:
        bus = MessageBus(num_agents=8)

        # Round N: agents send messages
        bus.send(make_axiom_hint(agent_id=2, round_sent=5, ...))

        # End of Round N: advance bus
        bus.advance_round()

        # Round N+1: agents receive messages sent in round N
        messages = bus.receive(agent_id=3, current_round=6)
    """

    def __init__(self, num_agents: int, max_queue_per_round: int = 5):
        self.num_agents = num_agents
        self.max_queue_per_round = max_queue_per_round
        self.current_round = 0

        # Queue: round_sent → List[Message]
        self._queue: Dict[int, List[Message]] = defaultdict(list)

        # Stats
        self.total_sent: int = 0
        self.total_delivered: int = 0
        self.messages_by_type: Dict[MessageType, int] = defaultdict(int)

    def send(self, message: Message) -> bool:
        """
        Send a message. Delivered to all other agents in the NEXT round.

        Returns False if queue is full (dropped), True if accepted.
        """
        round_queue = self._queue[message.round_sent]
        if len(round_queue) >= self.max_queue_per_round * self.num_agents:
            return False  # Queue full — message dropped

        self._queue[message.round_sent].append(message)
        self.total_sent += 1
        self.messages_by_type[message.msg_type] += 1
        return True

    def receive(
        self,
        agent_id: int,
        current_round: int
    ) -> List[Message]:
        """
        Get all messages available to this agent in the current round.

        Returns messages sent in previous rounds (round < current_round)
        that are not from this agent.

        This enforces the delayed delivery rule.
        """
        messages = []
        for sent_round, round_msgs in self._queue.items():
            if sent_round >= current_round:
                continue  # Not yet delivered
            for msg in round_msgs:
                if msg.sender_id == agent_id:
                    continue  # Don't receive own messages
                if not msg.delivered:
                    messages.append(msg)
                    self.total_delivered += 1

        # Mark delivered
        for msg in messages:
            msg.delivered = True

        return messages

    def advance_round(self) -> None:
        """Advance the bus to the next round."""
        self.current_round += 1
        # Clean up old messages (keep only last 3 rounds)
        old_rounds = [r for r in self._queue if r < self.current_round - 3]
        for r in old_rounds:
            del self._queue[r]

    def stats(self) -> Dict:
        return {
            'current_round': self.current_round,
            'total_sent': self.total_sent,
            'total_delivered': self.total_delivered,
            'by_type': {t.name: c for t, c in self.messages_by_type.items()},
        }

    def summary(self) -> str:
        s = self.stats()
        lines = ['MessageBus stats:']
        for k, v in s.items():
            if k != 'by_type':
                lines.append(f'  {k}: {v}')
        lines.append(f"  by_type: {s['by_type']}")
        return '\n'.join(lines)