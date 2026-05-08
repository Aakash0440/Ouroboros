"""
ouroboros.grammar.dynamic_adapter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DynamicGrammarAdapter: plugs PrimitiveProposer + PrimitiveVerifier +
DynamicGrammar into the search router's stuck-detection hook.

Call adapter.maybe_expand(sequence, best_cost) after each search round.
If a new primitive is registered it returns the DynamicNodeType, else None.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from ouroboros.synthesis.primitive_proposer import PrimitiveProposer
from ouroboros.synthesis.primitive_verifier import PrimitiveVerifier
from ouroboros.grammar.dynamic_grammar import DynamicGrammar, DynamicNodeType, DYNAMIC_GRAMMAR


class DynamicGrammarAdapter:
    """
    Ties together the propose → verify → register pipeline.

    Parameters
    ----------
    stuck_threshold : float
        MDL cost above which the search is considered stuck.
    min_gain : float
        Minimum estimated bit gain to bother proposing a primitive.
    grammar : DynamicGrammar
        Shared registry (defaults to module-level DYNAMIC_GRAMMAR).
    """

    def __init__(
        self,
        stuck_threshold: float = 100.0,
        min_gain: float = 10.0,
        grammar: Optional[DynamicGrammar] = None,
    ):
        self.proposer  = PrimitiveProposer(stuck_threshold, min_compression_gain=min_gain)
        self.verifier  = PrimitiveVerifier()
        self.grammar   = grammar or DYNAMIC_GRAMMAR
        self._seen: set[str] = set()   # avoid re-proposing same sequence

    def maybe_expand(
        self,
        sequence: Sequence[int],
        best_expr: Any,
        best_cost: float,
    ) -> Optional[DynamicNodeType]:
        """
        Run propose → verify → register.
        Returns the new DynamicNodeType if one was registered, else None.
        """
        # Fingerprint to avoid redundant work
        fp = str(list(sequence)[:10])
        if fp in self._seen:
            return None
        self._seen.add(fp)

        proposal = self.proposer.maybe_propose(sequence, best_expr, best_cost)
        if proposal is None:
            return None

        result = self.verifier.verify(proposal.specification)
        if not result.is_valid:
            return None

        node = self.grammar.register(result, name=proposal.proposed_name)
        if node is not None:
            print(f"[DynamicGrammar] Registered '{node.name}' "
                  f"(props: {node.description})")
        return node