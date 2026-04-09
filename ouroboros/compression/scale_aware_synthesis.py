"""
Scale-aware program synthesis.

Standard BeamSearchSynthesizer finds expressions for the raw sequence.
ScaleAwareSynthesizer finds expressions for AGGREGATED sequences at each scale.

Key insight: the expression for a scale-16 aggregated sequence
is NOT the same as the expression for the raw sequence.

Example:
    Raw (scale 1): obs[t] = (3t+1) mod 7   ← fast pattern
    Aggregated at scale 16:
        agg[k] = sum(obs[16k..16k+15]) mod 7
               = sum((3(16k+j)+1) for j in 0..15) mod 7
               = (some function of k) mod 7  ← a different expression

An agent that searches at scale 1 finds the fast rule.
An agent that searches at scale 16 finds the slow rule (after aggregation).
Together they describe the full multi-scale structure.

This module provides:
    ScaleAwareSynthesizer: runs BeamSearch at each specified scale
    Results keyed by scale: {1: (expr, cost), 4: (expr, cost), ...}
"""

from typing import List, Dict, Tuple, Optional
from ouroboros.compression.program_synthesis import ExprNode, C
from ouroboros.compression.beam_search import BeamSearchSynthesizer
from ouroboros.compression.hierarchical_mdl import HierarchicalMDL, aggregate_sequence
from ouroboros.compression.mdl import naive_bits


class ScaleAwareSynthesizer:
    """
    Runs BeamSearchSynthesizer at each temporal scale.

    For each scale k in scales:
    1. Aggregate raw sequence into windows of size k
    2. Run BeamSearch on the aggregated sequence
    3. Store (expression, mdl_cost) keyed by scale

    An agent using this synthesizer maintains a separate best expression
    per scale, and can report: "at scale 1, I found X; at scale 16, I found Y."

    Args:
        scales: List of window sizes to probe
        alphabet_size: Symbol alphabet size
        beam_width: BeamSearch beam width per scale
        max_depth: Max expression tree depth
        const_range: Search constants 0..const_range
        mdl_lambda: MDL lambda weight
    """

    def __init__(
        self,
        scales: List[int] = None,
        alphabet_size: int = 4,
        beam_width: int = 20,
        max_depth: int = 3,
        const_range: int = 16,
        mdl_lambda: float = 1.0
    ):
        self.scales = scales or [1, 4, 16, 64]
        self.alphabet_size = alphabet_size
        self.hier = HierarchicalMDL(scales, alphabet_size)

        # One synthesizer per scale (same params, separate instances)
        self.synthesizers: Dict[int, BeamSearchSynthesizer] = {
            scale: BeamSearchSynthesizer(
                beam_width=beam_width,
                max_depth=max_depth,
                const_range=const_range,
                alphabet_size=alphabet_size,
                lambda_weight=mdl_lambda
            )
            for scale in self.scales
        }

    def search_all_scales(
        self,
        raw_sequence: List[int],
        min_length: int = 20,
        verbose: bool = False
    ) -> Dict[int, Tuple[Optional[ExprNode], float]]:
        """
        Run beam search at each scale on the aggregated sequences.

        Args:
            raw_sequence: Full raw observation sequence
            min_length: Skip scales where aggregated sequence is too short
            verbose: Print search status per scale

        Returns:
            Dict mapping scale → (best_expression, mdl_cost)
            mdl_cost is normalized to the aggregated sequence length
        """
        results: Dict[int, Tuple[Optional[ExprNode], float]] = {}

        for scale in self.scales:
            # Aggregate sequence at this scale
            aggregated = aggregate_sequence(
                raw_sequence, scale, self.alphabet_size
            )

            if len(aggregated) < min_length:
                results[scale] = (None, float('inf'))
                continue

            if verbose:
                print(f"  Scale {scale}: aggregated length={len(aggregated)}")

            # Search for best expression
            synth = self.synthesizers[scale]
            expr, cost = synth.search(aggregated, verbose=verbose)

            # Normalize cost by naive bits of aggregated sequence
            nb = naive_bits(aggregated, self.alphabet_size)
            normalized_cost = cost / nb if nb > 0 else 1.0

            results[scale] = (expr, normalized_cost)

            if verbose:
                print(f"  Scale {scale}: best={expr.to_string()!r}  "
                      f"normalized_cost={normalized_cost:.4f}")

        return results

    def best_scale_result(
        self,
        results: Dict[int, Tuple[Optional[ExprNode], float]]
    ) -> Tuple[int, Optional[ExprNode], float]:
        """
        Return the scale with the best (lowest) normalized MDL cost.

        Returns: (best_scale, best_expression, best_normalized_cost)
        """
        best_scale = min(
            (s for s in results if results[s][0] is not None),
            key=lambda s: results[s][1],
            default=self.scales[0]
        )
        expr, cost = results[best_scale]
        return best_scale, expr, cost

    def compression_improvement_across_scales(
        self,
        results: Dict[int, Tuple[Optional[ExprNode], float]]
    ) -> float:
        """
        How much does the best scale improve over scale-1?

        Returns: ratio_scale1 / ratio_best_scale
        If > 1.5: meaningful multi-scale structure detected.
        """
        if 1 not in results or results[1][0] is None:
            return 1.0
        cost_scale1 = results[1][1]
        best_cost = min(
            results[s][1] for s in results if results[s][0] is not None
        )
        if best_cost == 0:
            return float('inf')
        return cost_scale1 / best_cost