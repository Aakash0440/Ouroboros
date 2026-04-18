"""
TheoryAgent — Phase 3 agent that maintains a full CausalTheory.

Extends HierarchicalAgent with:
1. CausalTheory: a collection of scale-tagged axioms
2. Theory-level proposal generation: propose upgrades to the whole theory
3. Theory richness tracking: measures recursive ascent at the theory level
4. Cross-environment theory combination (for CRT experiment)

The key upgrade from Phase 2 to Phase 3:
    Phase 2: Agent proposes ONE expression replacement at a time.
    Phase 3: Agent proposes THEORY UPGRADES — replacing multiple
             scale-axioms simultaneously when cross-scale consistency improves.

Args:
    agent_id: Unique identifier
    alphabet_size: Symbol alphabet size
    scales: Temporal scales for the theory
    (all other args forwarded to HierarchicalAgent)
"""

from typing import List, Dict, Optional, Tuple
from ouroboros.agents.hierarchical_agent import HierarchicalAgent
from ouroboros.emergence.causal_theory import CausalTheory, ScaleAxiom
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import MDLCost, naive_bits
from ouroboros.agents.self_modifying_agent import ModificationProposal
from ouroboros.utils.logger import get_logger


class TheoryProposal:
    """
    A proposed upgrade to an agent's CausalTheory.

    Unlike ModificationProposal (single expression), a TheoryProposal
    may upgrade multiple scales at once.

    Fields:
        agent_id: Proposing agent
        old_theory_richness: Richness before upgrade
        new_scale_axioms: Dict of scale → new ExprNode
        test_sequence: Test data for market evaluation
        alphabet_size: Symbol alphabet
        primary_scale: The scale with most improvement (for market focus)
    """

    def __init__(
        self,
        agent_id: int,
        old_theory_richness: float,
        new_scale_axioms: Dict[int, ExprNode],
        new_scale_ratios: Dict[int, float],
        test_sequence: List[int],
        alphabet_size: int
    ):
        self.agent_id = agent_id
        self.old_theory_richness = old_theory_richness
        self.new_scale_axioms = new_scale_axioms
        self.new_scale_ratios = new_scale_ratios
        self.test_sequence = test_sequence
        self.alphabet_size = alphabet_size

        # Primary scale: biggest improvement
        if new_scale_ratios:
            self.primary_scale = min(new_scale_ratios, key=new_scale_ratios.get)
            self.primary_expr = new_scale_axioms.get(self.primary_scale)
        else:
            self.primary_scale = 1
            self.primary_expr = None

    def to_modification_proposal(
        self,
        current_theory: CausalTheory
    ) -> Optional[ModificationProposal]:
        """
        Convert the primary-scale improvement to a ModificationProposal
        for compatibility with the existing ProofMarket.
        """
        if self.primary_expr is None:
            return None

        current_ax = current_theory.axioms.get(self.primary_scale)
        if current_ax is None:
            from ouroboros.compression.program_synthesis import C
            current_expr = C(0)
            current_cost = naive_bits(self.test_sequence, self.alphabet_size)
        else:
            current_expr = current_ax.expression
            mdl = MDLCost()
            n = len(self.test_sequence)
            preds = current_expr.predict_sequence(n, self.alphabet_size)
            current_cost = mdl.total_cost(
                current_expr.to_bytes(), preds, self.test_sequence, self.alphabet_size
            )

        mdl = MDLCost()
        n = len(self.test_sequence)
        new_preds = self.primary_expr.predict_sequence(n, self.alphabet_size)
        new_cost = mdl.total_cost(
            self.primary_expr.to_bytes(), new_preds, self.test_sequence, self.alphabet_size
        )

        return ModificationProposal(
            agent_id=self.agent_id,
            current_expr=current_expr,
            proposed_expr=self.primary_expr,
            current_cost=current_cost,
            proposed_cost=new_cost,
            test_sequence=self.test_sequence,
            alphabet_size=self.alphabet_size
        )


class TheoryAgent(HierarchicalAgent):
    """
    Phase 3 agent with full CausalTheory management.

    Args:
        agent_id: Unique identifier
        alphabet_size: Symbol alphabet size
        scales: Theory scales
        (other args to HierarchicalAgent)
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        scales: List[int] = None,
        **kwargs
    ):
        super().__init__(agent_id, alphabet_size, scales=scales, **kwargs)
        self.theory = CausalTheory(
            scales=self.scales,
            alphabet_size=alphabet_size
        )
        self.logger = get_logger(f'TheoryAgent_{agent_id}')
        self.theory_richness_history: List[float] = []

    def update_theory_from_search(self, step: int = 0) -> None:
        """
        After search_and_update(), sync theory with per-scale programs.
        """
        for scale in self.scales:
            expr = self.scale_programs.get(scale)
            if expr is None:
                continue
            ratio = self.scale_costs.get(scale, 1.0)
            # Confidence from compression: lower ratio = higher confidence
            confidence = max(0.1, 1.0 - ratio)
            self.theory.update_scale(
                scale=scale,
                expression=expr,
                compression_ratio=ratio,
                confidence=confidence,
                step=step
            )

        self.theory.compute_cross_scale_consistency()
        self.theory_richness_history.append(self.theory.richness_score())

    def generate_theory_proposal(
        self,
        new_data: List[int],
        min_richness_improvement: float = 0.02
    ) -> Optional[TheoryProposal]:
        """
        Search for an improvement to the full theory.

        For each scale, check if a better expression exists.
        If the new theory would be richer, generate a TheoryProposal.

        Args:
            new_data: New observations
            min_richness_improvement: Min richness delta to bother proposing

        Returns:
            TheoryProposal if improvement found, else None
        """
        old_richness = self.theory.richness_score()
        scale_results = self.scale_synth.search_all_scales(
            raw_sequence=new_data[:min(500, len(new_data))],
            min_length=15
        )

        new_scale_axioms: Dict[int, ExprNode] = {}
        new_scale_ratios: Dict[int, float] = {}

        for scale, (expr, norm_cost) in scale_results.items():
            if expr is None:
                continue
            current_ax = self.theory.axioms.get(scale)
            current_ratio = current_ax.compression_ratio if current_ax else 1.0
            if norm_cost < current_ratio - 0.02:
                new_scale_axioms[scale] = expr
                new_scale_ratios[scale] = norm_cost

        if not new_scale_axioms:
            return None

        # Estimate new richness
        test_theory = CausalTheory(self.scales, self.alphabet_size)
        for s in self.scales:
            ax = self.theory.axioms.get(s)
            if ax:
                test_theory.update_scale(s, ax.expression, ax.compression_ratio,
                                         ax.confidence)
        for s, expr in new_scale_axioms.items():
            test_theory.update_scale(s, expr, new_scale_ratios[s],
                                     max(0.1, 1.0 - new_scale_ratios[s]))
        test_theory.compute_cross_scale_consistency()
        new_richness = test_theory.richness_score()

        if new_richness - old_richness < min_richness_improvement:
            return None

        return TheoryProposal(
            agent_id=self.agent_id,
            old_theory_richness=old_richness,
            new_scale_axioms=new_scale_axioms,
            new_scale_ratios=new_scale_ratios,
            test_sequence=new_data[:200],
            alphabet_size=self.alphabet_size
        )

    def apply_theory_upgrade(
        self,
        proposal: TheoryProposal,
        step: int
    ) -> None:
        """Apply an approved theory upgrade."""
        for scale, expr in proposal.new_scale_axioms.items():
            ratio = proposal.new_scale_ratios.get(scale, 1.0)
            confidence = max(0.1, 1.0 - ratio)
            self.theory.update_scale(scale, expr, ratio, confidence, step=step)
            self.scale_programs[scale] = expr
            self.scale_costs[scale] = ratio

        self.theory.compute_cross_scale_consistency()
        self.theory_richness_history.append(self.theory.richness_score())

        # Update primary expression for compatibility
        best_expr = self.theory.best_expression()
        if best_expr:
            self.best_expression = best_expr
            self._using_symbolic = True

        self.logger.info(
            f"Agent {self.agent_id}: theory upgraded at step {step}. "
            f"New richness={self.theory.richness_score():.4f}"
        )

    def theory_summary(self) -> str:
        return self.theory.summary()