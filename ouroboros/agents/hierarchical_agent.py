"""
HierarchicalAgent — agent that searches for programs at multiple scales.

Extends SynthesisAgent with:
1. HierarchicalMDL: compression profile at all scales
2. ScaleAwareSynthesizer: beam search at each scale
3. Scale-specific program store: one best expression per scale
4. Cross-scale consistency check: do programs at different scales agree?

When run on MultiScaleEnv(slow=100, fast=7):
    Scale 1:  finds "(t * ? + ?) mod 7"   — fast pattern
    Scale 16: finds "(t * ? + ?) mod 100" — slow pattern (after aggregation)
    These are TWO DIFFERENT axioms, each describing different structure.

When run on ModularArithmeticEnv:
    Scale 1:  finds "(t * 3 + 1) mod 7"   — the rule
    Scale 4:  finds similar (same rule, different aggregation)
    Scale 16: finds similar
    → All scales agree → high cross-scale consistency → high axiom confidence

The cross-scale consistency is what makes an axiom robust.
An axiom that looks right at scale 1 but breaks at scale 16
is a scale-specific artifact, not a fundamental rule.
"""

from typing import List, Dict, Optional, Tuple
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.compression.hierarchical_mdl import HierarchicalMDL
from ouroboros.compression.scale_aware_synthesis import ScaleAwareSynthesizer
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import naive_bits, MDLCost


class HierarchicalAgent(SynthesisAgent):
    """
    Agent that discovers programs at multiple temporal scales.

    Search hierarchy per checkpoint:
    1. Scale-aware beam search (all scales)
    2. MCMC refinement on best-scale candidate
    3. N-gram fallback (from BaseAgent)
    4. Keep whichever has lowest MDL cost overall
    5. Track per-scale programs for axiom pool submission

    Args:
        agent_id: Unique agent ID
        alphabet_size: Symbol alphabet size
        scales: Temporal scales to probe (default [1, 4, 16, 64])
        beam_width: Beam width per scale
        max_depth: Max expression depth
        const_range: Constant search range
        mcmc_iterations: MCMC refinement steps
        seed: Random seed
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        scales: List[int] = None,
        beam_width: int = 20,
        max_depth: int = 3,
        const_range: int = 16,
        mcmc_iterations: int = 150,
        lambda_weight: float = 1.0,
        seed: int = 42
    ):
        super().__init__(
            agent_id=agent_id,
            alphabet_size=alphabet_size,
            beam_width=beam_width,
            max_depth=max_depth,
            const_range=const_range,
            mcmc_iters=mcmc_iterations,
            lambda_weight=lambda_weight,
            seed=seed
        )
        self.scales = scales or [1, 4, 16, 64]
        self.hier = HierarchicalMDL(self.scales, alphabet_size)
        self.scale_synth = ScaleAwareSynthesizer(
            scales=self.scales,
            alphabet_size=alphabet_size,
            beam_width=beam_width,
            max_depth=max_depth,
            const_range=const_range,
            mdl_lambda=lambda_weight
        )

        # Best program per scale
        self.scale_programs: Dict[int, Optional[ExprNode]] = {s: None for s in self.scales}
        self.scale_costs: Dict[int, float] = {s: float('inf') for s in self.scales}
        self.scale_compression_ratios: Dict[int, List[float]] = {s: [] for s in self.scales}

        # Cross-scale consistency score [0, 1]
        self.cross_scale_consistency: float = 0.0

        # Which scale had the best program overall
        self.dominant_scale: int = 1

    def search_and_update(self, search_budget: int = 100) -> float:
        """
        Search at all scales + standard hybrid search.

        Returns best MDL cost found.
        """
        history = self.observation_history
        if len(history) < max(self.scales) * 4:
            # Not enough data to aggregate at all scales
            return super().search_and_update()

        # Scale-aware search
        search_data = history[:min(800, len(history))]
        scale_results = self.scale_synth.search_all_scales(
            raw_sequence=search_data,
            min_length=15,
            verbose=False
        )

        # Store per-scale programs
        for scale, (expr, norm_cost) in scale_results.items():
            if expr is not None:
                self.scale_programs[scale] = expr
                self.scale_costs[scale] = norm_cost

        # Find dominant scale
        valid_scales = [s for s in self.scales
                        if self.scale_programs[s] is not None]
        if valid_scales:
            self.dominant_scale = min(valid_scales, key=lambda s: self.scale_costs[s])

        # Compute cross-scale consistency
        self.cross_scale_consistency = self._compute_consistency(search_data)

        # Standard hybrid search (parent class handles MCMC + n-gram)
        base_cost = super().search_and_update()

        # Override with best scale program if it's better
        dominant_expr = self.scale_programs.get(self.dominant_scale)
        if dominant_expr is not None:
            dominant_cost = self.scale_costs[self.dominant_scale]
            if dominant_cost < (base_cost / max(naive_bits(search_data, self.alphabet_size), 1)):
                self.best_expression = dominant_expr
                self._using_symbolic = True

        return base_cost

    def measure_compression_ratio(self) -> float:
        """
        Measure compression at the dominant scale.
        Also records per-scale ratios.
        """
        history = self.observation_history
        if not history:
            return 1.0

        # Record per-scale ratios
        for scale in self.scales:
            profile = self.hier.compression_profile(history)
            ratio = profile.get(scale, 1.0)
            self.scale_compression_ratios[scale].append(ratio)

        return super().measure_compression_ratio()

    def _compute_consistency(self, sequence: List[int]) -> float:
        """
        Cross-scale consistency: do programs at different scales agree?

        Method:
        1. Get predictions from scale-1 program on raw sequence
        2. Get predictions from scale-k program on aggregated sequence
        3. Check: are the aggregated scale-1 predictions similar to scale-k predictions?

        Perfect consistency (1.0): all scales predict the same aggregated behavior.
        Zero consistency (0.0): scales completely disagree.
        """
        if not any(self.scale_programs[s] for s in self.scales):
            return 0.0

        test_len = min(100, len(sequence))
        if test_len < 10:
            return 0.0

        scale1_expr = self.scale_programs.get(1)
        if scale1_expr is None:
            return 0.0

        # Get scale-1 predictions
        scale1_preds = scale1_expr.predict_sequence(test_len, self.alphabet_size)

        # Compare aggregated scale-1 with each other scale's predictions
        consistency_scores = []
        for scale in self.scales:
            if scale == 1:
                continue
            scale_expr = self.scale_programs.get(scale)
            if scale_expr is None:
                continue

            from ouroboros.compression.hierarchical_mdl import aggregate_sequence
            agg_scale1 = aggregate_sequence(scale1_preds, scale, self.alphabet_size)
            agg_len = len(agg_scale1)
            if agg_len == 0:
                continue

            scale_preds = scale_expr.predict_sequence(agg_len, self.alphabet_size)
            matches = sum(
                p == a for p, a in zip(scale_preds[:agg_len], agg_scale1)
            )
            consistency_scores.append(matches / agg_len)

        return float(sum(consistency_scores) / max(len(consistency_scores), 1))

    def scale_programs_summary(self) -> str:
        """Human-readable summary of programs at each scale."""
        lines = [f"HierarchicalAgent {self.agent_id} scale programs:"]
        for scale in self.scales:
            expr = self.scale_programs.get(scale)
            cost = self.scale_costs.get(scale, float('inf'))
            expr_str = expr.to_string() if expr else "none"
            dominant_mark = " ← DOMINANT" if scale == self.dominant_scale else ""
            lines.append(f"  Scale {scale:3d}: {expr_str[:30]!r}  cost={cost:.4f}{dominant_mark}")
        lines.append(f"  Cross-scale consistency: {self.cross_scale_consistency:.4f}")
        return '\n'.join(lines)

    def get_all_scale_expressions(self) -> Dict[int, Optional[ExprNode]]:
        """Return dict of all scale → expression (for axiom pool submission)."""
        return {s: expr for s, expr in self.scale_programs.items()
                if expr is not None}