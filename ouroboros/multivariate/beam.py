"""
MultivariateBeam: discovers symbolic expressions that reference multiple
input channels. Core use case: given (position, velocity, acceleration),
find that acceleration = -0.09 * position.

Architecture:
  - channels: Dict[str, List[int]]  e.g. {"pos": [...], "vel": [...], "acc": [...]}
  - target: str                     which channel to explain
  - predictor channels are injected as CHANNEL_PREV seeds
  - Uses existing GrammarConstrainedBeam with extended seed set
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ouroboros.search.grammar_beam import GrammarConstrainedBeam, GrammarBeamConfig, GrammarBeamCandidate
from ouroboros.nodes.extended_nodes import ExtExprNode, ExtNodeType, NodeCategory
from ouroboros.compression.mdl_engine import MDLEngine
from ouroboros.synthesis.expr_node import NodeType


@dataclass
class MultivariateResult:
    target_channel: str
    expr: Optional[ExtExprNode]
    mdl_cost: float
    ratio: float
    time_seconds: float
    channel_names: List[str]

    def description(self) -> str:
        expr_str = self.expr.to_string() if self.expr else "None"
        return (f"{self.target_channel} = {expr_str}  "
                f"(ratio={self.ratio:.3f}, t={self.time_seconds:.1f}s)")


class MultivariateBeam:
    """
    Searches for symbolic expressions over multiple channels.

    Usage:
        mb = MultivariateBeam()
        channels = {"pos": [...], "acc": [...]}
        result = mb.search(channels, target="acc")
        print(result.description())
        # acc = MUL(-0.090, ch[0][t-1])  (ratio=0.041)
    """

    def __init__(self, beam_width: int = 25, n_iterations: int = 15):
        self._beam_width   = beam_width
        self._n_iterations = n_iterations
        self._mdl = MDLEngine()

    def search(
        self,
        channels: Dict[str, List[int]],
        target: str,
        verbose: bool = False,
    ) -> MultivariateResult:

        if target not in channels:
            raise ValueError(f"Target channel '{target}' not in channels")

        t0 = time.time()
        target_seq   = channels[target]
        channel_list = [name for name in channels if name != target]
        channel_data = [channels[name] for name in channel_list]

        # Pack channel data into state dict for evaluate()
        # state[-1] = {channel_idx: List[int]}
        ch_map = {i: data for i, data in enumerate(channel_data)}

        # Build beam config with cross-channel seeds injected
        cfg = GrammarBeamConfig(
            beam_width=self._beam_width,
            n_iterations=self._n_iterations,
            const_range=max(target_seq) + 2 if target_seq else 50,
        )
        beam = GrammarConstrainedBeam(cfg)

        # Inject cross-channel seeds before search
        extra_seeds = self._cross_channel_seeds(
        target_seq, channel_data, channel_list, ch_map
        )

        # Run search with extra seeds pre-loaded
        expr = beam.search(
            target_seq,
            extra_seeds=extra_seeds,
            channel_state=ch_map,
            verbose=verbose,
        )

        # Score
        cost  = float('inf')
        ratio = float('inf')
        if expr is not None:
            try:
                state_with_channels = {-1: ch_map}
                preds = [
                    int(round(expr.evaluate(t, target_seq[:t], state_with_channels)))
                    for t in range(len(target_seq))
                ]
                alpha = max(target_seq) + 2
                r = self._mdl.compute(preds, target_seq,
                                      expr.node_count(), expr.constant_count(),
                                      alphabet_size=alpha)
                b = self._mdl.compute(
                    [int(sum(target_seq)/len(target_seq))] * len(target_seq),
                    target_seq, 1, 1, alphabet_size=alpha
                )
                cost  = r.total_mdl_cost
                b_cost = b.total_mdl_cost

                # MDL convention: more negative = better compression.
                # ratio = b / r so that ratio < 1 means expression beats baseline.
                # (both negative: -100 / -1163 = 0.086 → expression is ~12x better)
                if b_cost != 0 and r.total_mdl_cost != 0:
                    ratio = b_cost / r.total_mdl_cost
                else:
                    ratio = 999
            except Exception:
                pass

        return MultivariateResult(
            target_channel=target,
            expr=expr,
            mdl_cost=cost,
            ratio=ratio,
            time_seconds=time.time() - t0,
            channel_names=channel_list,
        )

    def _cross_channel_seeds(
        self,
        target: List[int],
        channel_data: List[List[int]],
        channel_names: List[str],
        ch_map: dict,
    ) -> List[GrammarBeamCandidate]:
        """Generate seeds that reference other channels, including lag-0."""
        seeds = []
        cfg_tmp = GrammarBeamConfig(beam_width=self._beam_width)
        beam_tmp = GrammarConstrainedBeam(cfg_tmp)
        beam_tmp._channel_state = ch_map

        # ── Single-channel seeds: CHANNEL_PREV and MUL(CONST, CHANNEL_PREV) ──────
        for ch_idx, (ch_data, ch_name) in enumerate(zip(channel_data, channel_names)):
            for lag in [0, 1, 2, 3]:          # lag=0 is the key fix
                if lag > 0 and len(ch_data) <= lag:
                    continue
                try:
                    ch_node = ExtExprNode(
                        ExtNodeType.CHANNEL_PREV,
                        channel_idx=ch_idx,
                        lag=lag,
                    )
                    score = beam_tmp._score(ch_node, target)
                    seeds.append(GrammarBeamCandidate(ch_node, score))

                    coeff = self._estimate_coeff(target, ch_data, lag)
                    if coeff != 0:
                        c_node = ExtExprNode(NodeType.CONST, value=coeff)
                        mul = ExtExprNode(NodeType.MUL, left=c_node, right=ch_node)
                        score2 = beam_tmp._score(mul, target)
                        seeds.append(GrammarBeamCandidate(mul, score2))

                except Exception:
                    pass

        # ── Additive combination seeds: c1*ch[i][lag_i] + c2*ch[j][lag_j] ────────
        for ci in range(len(channel_data)):
            for cj in range(ci + 1, len(channel_data)):
                for lag_i in [0, 1]:
                    for lag_j in [0, 1]:
                        try:
                            a = self._estimate_coeff(target, channel_data[ci], lag_i)
                            b = self._estimate_coeff(target, channel_data[cj], lag_j)
                            if a == 0 or b == 0:
                                continue
                            chi = ExtExprNode(ExtNodeType.CHANNEL_PREV,
                                            channel_idx=ci, lag=lag_i)
                            chj = ExtExprNode(ExtNodeType.CHANNEL_PREV,
                                            channel_idx=cj, lag=lag_j)
                            mul_i = ExtExprNode(NodeType.MUL,
                                                left=ExtExprNode(NodeType.CONST, value=a),
                                                right=chi)
                            mul_j = ExtExprNode(NodeType.MUL,
                                                left=ExtExprNode(NodeType.CONST, value=b),
                                                right=chj)
                            add_node = ExtExprNode(NodeType.ADD, left=mul_i, right=mul_j)
                            score3 = beam_tmp._score(add_node, target)
                            seeds.append(GrammarBeamCandidate(add_node, score3))
                        except Exception:
                            pass

        seeds.sort()
        return seeds[:self._beam_width]

    @staticmethod
    def _estimate_coeff(target: List[int], source: List[int], lag: int) -> float:
        """Least-squares coefficient for target[t] ≈ coeff * source[t-lag]."""
        pairs = [
            (source[t - lag], target[t])
            for t in range(lag, min(len(target), len(source)))
        ]
        if not pairs:
            return 0.0
        sx  = sum(x for x, _ in pairs)
        sy  = sum(y for _, y in pairs)
        sxx = sum(x*x for x, _ in pairs)
        sxy = sum(x*y for x, y in pairs)
        n   = len(pairs)
        denom = n * sxx - sx * sx
        return (n * sxy - sx * sy) / denom if abs(denom) > 1e-10 else 0.0