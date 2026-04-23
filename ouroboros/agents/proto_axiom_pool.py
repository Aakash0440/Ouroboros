from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class AxiomSubmission:
    agent_id: str
    expr: Any
    mdl_cost: float
    round_num: int

class ProtoAxiomPool:
    def __init__(self, consensus_threshold: float = 0.5, n_agents: int = 1):
        self.consensus_threshold = consensus_threshold
        self.n_agents = n_agents
        self._submissions: Dict[int, List[AxiomSubmission]] = {}
        self._promoted: bool = False
        self._promoted_round: Optional[int] = None
        self._promoted_expr: Optional[Any] = None

    def submit(self, agent_id: str, expr: Any, mdl_cost: float, round_num: int) -> None:
        if round_num not in self._submissions:
            self._submissions[round_num] = []
        self._submissions[round_num] = [s for s in self._submissions[round_num] if s.agent_id != agent_id]
        self._submissions[round_num].append(AxiomSubmission(agent_id=agent_id, expr=expr, mdl_cost=mdl_cost, round_num=round_num))
        self._check_consensus(round_num)

    def _check_consensus(self, round_num: int) -> None:
        if self._promoted:
            return
        subs = self._submissions.get(round_num, [])
        fraction = len(set(s.agent_id for s in subs)) / self.n_agents if self.n_agents > 0 else 0.0
        if fraction >= self.consensus_threshold:
            best = min(subs, key=lambda s: s.mdl_cost)
            self._promoted = True
            self._promoted_round = round_num
            self._promoted_expr = best.expr

    def has_promoted_axiom(self) -> bool:
        return self._promoted

    @property
    def promoted_round(self) -> Optional[int]:
        return self._promoted_round

    @property
    def promoted_expr(self) -> Optional[Any]:
        return self._promoted_expr

    def reset(self) -> None:
        self._submissions.clear()
        self._promoted = False
        self._promoted_round = None
        self._promoted_expr = None
