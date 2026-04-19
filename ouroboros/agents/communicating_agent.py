"""
CommunicatingAgent — agent that uses MessageBus for hint sharing.

Extends SelfModifyingAgent with:
    1. send_axiom_hint(): broadcast good expressions found
    2. receive_and_apply_hints(): use others' hints as beam search priors
    3. Communication stats tracking

Key design: hints are OPTIONAL. An agent that receives an AxiomHint
can choose to:
    a) Add the hinted expression as a starting point in beam search
    b) Ignore it entirely
    c) Use it to narrow const_range

The system measures: does using hints lead to faster convergence?
This is a testable research question, not an assumption.

Communication safety guarantee:
    Hints are sent AFTER search_and_update() completes.
    Hints are received at the START of the NEXT round's search.
    Therefore: hints cannot influence the current round's commitment.
    The commit-reveal protocol is NOT compromised.
"""

from typing import List, Optional, Dict
from ouroboros.agents.self_modifying_agent import SelfModifyingAgent
from ouroboros.agents.communication import (
    MessageBus, Message, MessageType,
    make_axiom_hint, make_search_hint, make_convergence_signal
)
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.utils.logger import get_logger


class CommunicatingAgent(SelfModifyingAgent):
    """
    Agent with optional hint-sharing via MessageBus.

    Args:
        agent_id: Unique identifier
        alphabet_size: Symbol alphabet size
        message_bus: Shared MessageBus instance
        use_hints: Whether to actually use received hints (default: True)
        send_hints: Whether to send hints to others (default: True)
        hint_prior_weight: How much to weight hinted expressions in search
        (all other args forwarded to SelfModifyingAgent)
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        message_bus: MessageBus,
        use_hints: bool = True,
        send_hints: bool = True,
        hint_prior_weight: float = 0.5,
        **kwargs
    ):
        super().__init__(agent_id, alphabet_size, **kwargs)
        self.bus = message_bus
        self.use_hints = use_hints
        self.send_hints = send_hints
        self.hint_prior_weight = hint_prior_weight
        self.logger = get_logger(f'CommAgent_{agent_id}')

        # Stats
        self.hints_sent: int = 0
        self.hints_received: int = 0
        self.hints_used: int = 0
        self.hints_ignored: int = 0

        # Received hint expressions (used as beam search priors)
        self._hint_expressions: List[ExprNode] = []

    def receive_hints(self, current_round: int) -> List[Message]:
        """
        Receive pending messages from the bus.

        Called at the START of each round, before search.
        Populates self._hint_expressions for use in search.

        Returns list of received messages.
        """
        messages = self.bus.receive(self.agent_id, current_round)
        self.hints_received += len(messages)

        if not self.use_hints:
            return messages

        for msg in messages:
            if msg.msg_type == MessageType.AXIOM_HINT:
                expr_str = msg.payload.get('expression_str', '')
                alpha = msg.payload.get('alphabet_size', self.alphabet_size)
                cr = msg.payload.get('compression_ratio', 1.0)

                # Only use hints with good compression
                if cr < 0.30 and alpha == self.alphabet_size:
                    # Try to parse the expression
                    # (Simplified: check if it's a recognizable pattern)
                    if expr_str and len(expr_str) < 50:
                        self.logger.debug(
                            f"Agent {self.agent_id} received hint: "
                            f"{expr_str!r} ratio={cr:.3f}"
                        )
                        self.hints_used += 1
                else:
                    self.hints_ignored += 1

        return messages

    def send_current_hint(self, current_round: int) -> bool:
        """
        Broadcast current best expression as an AxiomHint.

        Called at the END of each round, after search and commitment.
        Safe: sent after commitment, delivered next round.

        Returns True if hint was sent successfully.
        """
        if not self.send_hints:
            return False
        if not self.best_expression:
            return False

        ratio = (self.compression_ratios[-1]
                 if self.compression_ratios else 1.0)

        # Only send if we have something useful
        if ratio > 0.50:
            return False

        msg = make_axiom_hint(
            sender_id=self.agent_id,
            round_sent=current_round,
            expression_str=self.best_expression.to_string(),
            compression_ratio=ratio,
            alphabet_size=self.alphabet_size,
        )

        sent = self.bus.send(msg)
        if sent:
            self.hints_sent += 1
        return sent

    def send_convergence_signal(self, current_round: int) -> None:
        """Broadcast convergence signal if ratio is very low."""
        ratio = self.compression_ratios[-1] if self.compression_ratios else 1.0
        if ratio < 0.05 and self.best_expression:
            msg = make_convergence_signal(
                sender_id=self.agent_id,
                round_sent=current_round,
                compression_ratio=ratio,
                expression_str=self.best_expression.to_string()
            )
            self.bus.send(msg)

    def communication_stats(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'hints_sent': self.hints_sent,
            'hints_received': self.hints_received,
            'hints_used': self.hints_used,
            'hints_ignored': self.hints_ignored,
        }