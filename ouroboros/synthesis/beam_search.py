"""
Shim: wraps ouroboros.compression.program_synthesis.BeamSearchSynthesizer
behind a BeamConfig dataclass so search_strategy.py can do:

    from ouroboros.synthesis.beam_search import BeamSearchSynthesizer, BeamConfig
    cfg = BeamConfig(beam_width=25, ...)
    synth = BeamSearchSynthesizer(cfg)
    expr = synth.search(observations)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ouroboros.compression.program_synthesis import (
    BeamSearchSynthesizer as _CoreSynthesizer,
    ExprNode,
)


@dataclass
class BeamConfig:
    beam_width: int = 25
    const_range: int = 16          # reduced from 30 — kills the O(n²) rescue loop
    max_depth: int = 4
    max_lag: int = 3
    mcmc_iterations: int = 150     # accepted but not used by core synthesizer
    random_seed: int = 42
    alphabet_size: int = 256       # MUST be overridden by caller with env.alphabet_size
    enable_prev: bool = True
    enable_if: bool = True
    enable_pow: bool = True


class BeamSearchSynthesizer:
    """
    Thin wrapper that accepts a BeamConfig and delegates to the
    core BeamSearchSynthesizer in compression.program_synthesis.

    search() returns an ExprNode (not a tuple) so callers don't
    have to unpack — search_strategy.py does:
        best_expr = synthesizer.search(observations)

    IMPORTANT: Always set cfg.alphabet_size = env.alphabet_size before
    constructing this. The default 256 will produce wrong predictions
    for small-modulus environments.
    """

    def __init__(self, config: BeamConfig = None):
        self.cfg = config or BeamConfig()
        self._core = _CoreSynthesizer(
            beam_width=self.cfg.beam_width,
            max_depth=self.cfg.max_depth,
            const_range=self.cfg.const_range,
            max_lag=self.cfg.max_lag,
            alphabet_size=self.cfg.alphabet_size,
            enable_prev=self.cfg.enable_prev,
            enable_if=self.cfg.enable_if,
            enable_pow=self.cfg.enable_pow,
        )

    def search(self, observations: List[int]) -> Optional[ExprNode]:
        """Returns best ExprNode, or None if observations is empty."""
        if not observations:
            return None
        best_expr, _cost = self._core.search(observations)
        return best_expr