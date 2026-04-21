"""
CollaborativeProofSession — Multi-agent collaborative expression synthesis.

Orchestrates a group of agents where:
  - Agents with partial solutions broadcast their fragments
  - Agents receiving fragments attempt completion
  - Completed expressions are shared back to the originator
  - The best completed expression is submitted to ProtoAxiomPool

This implements true cross-agent program sharing:
  Agent A builds the SKELETON: MUL(CONST(?), MOD(TIME, CONST(?)))
  Agent B fills CONST hole 1: CONST(3) → MUL(CONST(3), MOD(TIME, CONST(?)))
  Agent C fills CONST hole 2: CONST(7) → MUL(CONST(3), MOD(TIME, CONST(7)))
  → Final expression: (3 * t) % 7

Without collaboration: each agent must find (3*t)%7 independently.
With collaboration: one agent finds the structure, others fill the constants.

This is qualitatively different from the MessageBus (Day 19) which only
shared high-level hints ("I think the modulus is around 7").
Here, agents share the actual expression tree and delegate specific
sub-searches to their peers.
"""

from __future__ import annotations
import copy
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from ouroboros.agents.expression_fragment import (
    ExpressionFragment, HoleSpec, HoleType,
    create_fragment_from_expr,
)
from ouroboros.agents.fragment_completer import FragmentCompleter, CompletionResult
from ouroboros.synthesis.expr_node import ExprNode, NodeType
from ouroboros.compression.mdl_engine import MDLEngine
from ouroboros.environments.base import Environment


@dataclass
class CollaborationMessage:
    """A message sent between agents during a collaborative proof session."""
    sender_id: str
    receiver_id: str   # or "ALL" for broadcast
    message_type: str  # "FRAGMENT", "COMPLETION", "ADOPT"
    
    # FRAGMENT message: sender is broadcasting a partial expression
    fragment: Optional[ExpressionFragment] = None
    
    # COMPLETION message: sender has completed a received fragment
    completion: Optional[CompletionResult] = None
    
    # ADOPT message: all agents should consider this as the new best
    completed_expr: Optional[ExprNode] = None
    final_mdl_cost: float = float('inf')
    
    round_sent: int = 0


@dataclass
class CollaborativeSessionResult:
    """Result of a full collaborative proof session."""
    participating_agents: List[str]
    environment_name: str
    n_rounds: int
    
    # Was a complete expression found?
    best_expr: Optional[ExprNode]
    best_mdl_cost: float
    
    # Statistics
    fragments_broadcast: int = 0
    completions_attempted: int = 0
    completions_successful: int = 0
    solo_best_cost: float = float('inf')  # cost if agents searched independently

    @property
    def collaboration_benefit_bits(self) -> float:
        """How many bits did collaboration save vs solo search?"""
        return self.solo_best_cost - self.best_mdl_cost

    def description(self) -> str:
        return (
            f"Collaborative Session ({len(self.participating_agents)} agents, "
            f"{self.n_rounds} rounds)\n"
            f"  Environment: {self.environment_name}\n"
            f"  Fragments broadcast: {self.fragments_broadcast}\n"
            f"  Completions: {self.completions_successful}/{self.completions_attempted}\n"
            f"  Best MDL: {self.best_mdl_cost:.2f} bits\n"
            f"  Solo best: {self.solo_best_cost:.2f} bits\n"
            f"  Collaboration benefit: {self.collaboration_benefit_bits:.2f} bits"
        )


class CollaborativeAgent:
    """
    An agent that participates in collaborative proof sessions.
    
    Extends SynthesisAgent with the ability to:
    1. Create fragments from its current best expression
    2. Complete received fragments from other agents
    3. Adopt completed expressions that are better than its own
    """

    def __init__(
        self,
        agent_id: str,
        const_range: int = 30,
        beam_width: int = 20,
        max_depth: int = 4,
        max_lag: int = 3,
        random_seed: int = 42,
    ):
        self.agent_id = agent_id
        self.const_range = const_range
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.max_lag = max_lag
        self.random_seed = random_seed

        self._mdl = MDLEngine()
        self._completer = FragmentCompleter(
            completing_agent=agent_id,
            const_range=const_range,
            max_lag=max_lag,
        )

        self.best_expr: Optional[ExprNode] = None
        self.best_cost: float = float('inf')
        self._messages_received: List[CollaborationMessage] = []

    def solo_search(
        self,
        observations: List[int],
    ) -> Optional[ExprNode]:
        """Run solo beam search (no collaboration)."""
        from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
        cfg = BeamConfig(
            beam_width=self.beam_width,
            const_range=self.const_range,
            max_depth=self.max_depth,
            max_lag=self.max_lag,
            mcmc_iterations=100,
            random_seed=self.random_seed,
        )
        synthesizer = BeamSearchSynthesizer(cfg)
        expr = synthesizer.search(observations)
        if expr is not None:
            preds = [expr.evaluate(t, observations[:t]) for t in range(len(observations))]
            r = self._mdl.compute(preds, observations, expr.node_count(), expr.constant_count())
            if r.total_mdl_cost < self.best_cost:
                self.best_cost = r.total_mdl_cost
                self.best_expr = expr
        return self.best_expr

    def create_fragment(
        self,
        n_holes: int = 1,
    ) -> Optional[ExpressionFragment]:
        """
        Create a fragment from current best expression by
        replacing n_holes CONST nodes with holes.
        
        Returns None if no best expression yet.
        """
        if self.best_expr is None:
            return None
        fragment = create_fragment_from_expr(
            self.best_expr,
            n_holes=n_holes,
            hole_type=HoleType.HOLE_CONST,
        )
        fragment.creator_agent = self.agent_id
        fragment.partial_mdl_cost = self.best_cost
        return fragment

    def complete_fragment(
        self,
        fragment: ExpressionFragment,
        observations: List[int],
    ) -> CompletionResult:
        """Complete a fragment received from another agent."""
        return self._completer.complete(fragment, observations)

    def receive_message(self, message: CollaborationMessage) -> None:
        """Store an incoming message for processing."""
        self._messages_received.append(message)

    def process_messages(
        self,
        observations: List[int],
    ) -> List[CollaborationMessage]:
        """
        Process all pending messages.
        
        For each FRAGMENT message: attempt completion, send back COMPLETION.
        For each COMPLETION/ADOPT message: adopt if better than current.
        
        Returns response messages to send.
        """
        responses = []
        for msg in self._messages_received:
            if msg.message_type == "FRAGMENT" and msg.fragment is not None:
                result = self.complete_fragment(msg.fragment, observations)
                if result.is_valid:
                    response = CollaborationMessage(
                        sender_id=self.agent_id,
                        receiver_id=msg.sender_id,
                        message_type="COMPLETION",
                        completion=result,
                    )
                    responses.append(response)
                    # Also check if this is better than our own
                    if result.mdl_cost < self.best_cost:
                        self.best_cost = result.mdl_cost
                        self.best_expr = copy.deepcopy(result.completed_expr)

            elif msg.message_type in ("COMPLETION", "ADOPT"):
                cost = msg.final_mdl_cost
                expr = msg.completed_expr
                if msg.message_type == "COMPLETION" and msg.completion:
                    cost = msg.completion.mdl_cost
                    expr = msg.completion.completed_expr
                if expr is not None and cost < self.best_cost:
                    self.best_cost = cost
                    self.best_expr = copy.deepcopy(expr)

        self._messages_received.clear()
        return responses


class CollaborativeProofSession:
    """
    Orchestrates a multi-round collaborative proof session.
    
    Round structure:
      1. All agents run solo search (establish baselines)
      2. The agent with the best expression broadcasts a fragment
      3. All other agents attempt to complete the fragment
      4. The best completion is adopted by all agents
      5. Repeat for N rounds
    """

    def __init__(
        self,
        agents: List[CollaborativeAgent],
        n_rounds: int = 5,
        stream_length: int = 400,
        n_holes_to_broadcast: int = 1,
    ):
        self.agents = agents
        self.n_rounds = n_rounds
        self.stream_length = stream_length
        self.n_holes = n_holes_to_broadcast

    def run(
        self,
        env: Environment,
        verbose: bool = False,
    ) -> CollaborativeSessionResult:
        """Run the full collaborative session."""
        observations = env.generate(self.stream_length)

        result = CollaborativeSessionResult(
            participating_agents=[a.agent_id for a in self.agents],
            environment_name=env.name,
            n_rounds=self.n_rounds,
            best_expr=None,
            best_mdl_cost=float('inf'),
        )

        # Round 0: solo search to establish baselines
        if verbose:
            print(f"\nRound 0: Solo search (establishing baselines)")
        for agent in self.agents:
            agent.solo_search(observations)
            if verbose:
                print(f"  {agent.agent_id}: cost={agent.best_cost:.2f}")

        result.solo_best_cost = min(a.best_cost for a in self.agents)

        # Collaborative rounds
        for round_num in range(1, self.n_rounds + 1):
            if verbose:
                print(f"\nRound {round_num}: Collaborative synthesis")

            # Find the agent with the best current expression
            leader = min(self.agents, key=lambda a: a.best_cost)
            if leader.best_expr is None:
                continue

            # Leader broadcasts a fragment
            fragment = leader.create_fragment(n_holes=self.n_holes)
            if fragment is None:
                continue

            result.fragments_broadcast += 1
            if verbose:
                print(f"  {leader.agent_id} broadcasting fragment: "
                      f"{fragment.root.to_string()} ({fragment.n_holes} holes)")

            # Broadcast to all other agents
            for agent in self.agents:
                if agent.agent_id == leader.agent_id:
                    continue
                msg = CollaborationMessage(
                    sender_id=leader.agent_id,
                    receiver_id=agent.agent_id,
                    message_type="FRAGMENT",
                    fragment=fragment,
                    round_sent=round_num,
                )
                agent.receive_message(msg)

            # Agents attempt completion and send responses
            best_completion: Optional[CompletionResult] = None
            for agent in self.agents:
                if agent.agent_id == leader.agent_id:
                    continue
                result.completions_attempted += 1
                responses = agent.process_messages(observations)
                for resp in responses:
                    if resp.completion and resp.completion.is_valid:
                        result.completions_successful += 1
                        if best_completion is None or \
                                resp.completion.mdl_cost < best_completion.mdl_cost:
                            best_completion = resp.completion
                        if verbose:
                            print(f"  {agent.agent_id} completed: "
                                  f"cost={resp.completion.mdl_cost:.2f}, "
                                  f"filled={resp.completion.filled_values}")

            # Broadcast the best completion to all agents
            if best_completion and best_completion.mdl_cost < result.best_mdl_cost:
                result.best_mdl_cost = best_completion.mdl_cost
                result.best_expr = best_completion.completed_expr

                adopt_msg = CollaborationMessage(
                    sender_id="SESSION",
                    receiver_id="ALL",
                    message_type="ADOPT",
                    completed_expr=best_completion.completed_expr,
                    final_mdl_cost=best_completion.mdl_cost,
                    round_sent=round_num,
                )
                for agent in self.agents:
                    agent.receive_message(adopt_msg)
                    agent.process_messages(observations)

                if verbose:
                    print(f"  ✅ ADOPTED: cost={result.best_mdl_cost:.2f} bits")

        # Final check: is any agent's solo best better than collaboration?
        for agent in self.agents:
            if agent.best_cost < result.best_mdl_cost:
                result.best_mdl_cost = agent.best_cost
                result.best_expr = agent.best_expr

        if verbose:
            print(f"\n{result.description()}")

        return result