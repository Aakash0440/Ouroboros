# ouroboros/agents/synthesis_agent.py

"""
SynthesisAgent — agent with symbolic program synthesis.

Extends BaseAgent by replacing n-gram search with:
1. BeamSearchSynthesizer  → finds the best symbolic expression
2. MCMCRefiner            → refines the constants
3. Comparison with n-gram → keeps whichever has lower MDL cost

When a SynthesisAgent running on ModularArithmeticEnv(7,3,1)
returns expression = "(t * 3 + 1) mod 7" with ratio < 0.005,
it has DISCOVERED MODULAR ARITHMETIC from compression pressure alone.

The n-gram fallback matters because:
- Some environments have no algebraic expression (PrimeSequenceEnv)
- In those cases, n-gram is the best available program
- The hybrid agent handles both cases gracefully
"""

from typing import List, Optional, Tuple
import numpy as np

from ouroboros.agents.base_agent import BaseAgent
from ouroboros.compression.beam_search import BeamSearchSynthesizer
from ouroboros.compression.mcmc_refiner import MCMCRefiner
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import MDLCost, naive_bits


class SynthesisAgent(BaseAgent):
    """
    Hybrid agent: symbolic expression search + n-gram fallback.

    Search strategy per call to search_and_update():
    1. Run BeamSearchSynthesizer on the first min(500, len) symbols
    2. Optionally run MCMCRefiner to polish constants (if enabled)
    3. Run n-gram search (parent class)
    4. Compare symbolic vs n-gram MDL costs
    5. Keep whichever is lower
    6. Track symbolic_wins / ngram_wins counts

    The symbolic search runs on a prefix (500 symbols) for speed.
    Final MDL evaluation always uses the FULL history.

    Args:
        agent_id: Unique ID
        alphabet_size: Symbol count
        beam_width: Beam search width (default 25)
        max_depth: Expression tree depth (default 3)
        const_range: Constant search range 0..const_range (default 16)
        use_mcmc: Run MCMC refinement after beam search (default True)
        mcmc_iters: MCMC iterations (default 150)
        seed: Random seed
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        beam_width: int = 25,
        max_depth: int = 3,
        const_range: int = 16,
        use_mcmc: bool = True,
        mcmc_iters: int = 150,
        mcmc_iterations: Optional[int] = None,
        max_context_length: int = 6,
        lambda_weight: float = 1.0,
        seed: int = 42,
    ):
    
        super().__init__(
            agent_id=agent_id,
            alphabet_size=alphabet_size,
            max_context_length=max_context_length,
            lambda_weight=lambda_weight,
            seed=seed,
            
        )
        if mcmc_iterations is not None:           
            mcmc_iters = mcmc_iterations 

        self.synthesizer = BeamSearchSynthesizer(
            beam_width=beam_width,
            max_depth=max_depth,
            const_range=const_range,
            alphabet_size=alphabet_size,
            lambda_weight=lambda_weight,
        )

        self.refiner: Optional[MCMCRefiner] = None
        if use_mcmc:
            self.refiner = MCMCRefiner(
                num_iterations=mcmc_iters,
                alphabet_size=alphabet_size,
                seed=seed + agent_id,
            )

        # Best symbolic expression found so far
        self.best_expression: Optional[ExprNode] = None
        self.best_expression_cost: float = float('inf')

        # Counts for diagnostics
        self.symbolic_wins: int = 0
        self.ngram_wins: int = 0
        self._using_symbolic: bool = False

    def search_and_update(self) -> float:
        """
        Hybrid search: symbolic beam search vs. n-gram.
        Keeps whichever program has lower MDL cost.

        Returns:
            Best MDL cost found (lower is better)
        """
        history = self.observation_history
        if len(history) < 8:
            return float('inf')

        mdl = MDLCost(lambda_weight=self.synthesizer.mdl.lambda_weight)

        # ── Path 1: Symbolic beam search ──────────────────────────────────────
        # Use prefix of up to 500 symbols for speed
        search_data = history[:500] if len(history) > 500 else history

        sym_expr, sym_cost_on_prefix = self.synthesizer.search(
            actuals=search_data,
            verbose=False,
        )

        # Optionally refine with MCMC — re-score both with same MDL to compare
        if self.refiner is not None:
            refined_expr, _ = self.refiner.refine(
                sym_expr, search_data, verbose=False
            )
            n_prefix = len(search_data)
            beam_preds = sym_expr.predict_sequence(n_prefix, self.alphabet_size)
            refined_preds = refined_expr.predict_sequence(n_prefix, self.alphabet_size)
            beam_cost_check = mdl.total_cost(
                sym_expr.to_bytes(), beam_preds, search_data, self.alphabet_size
            )
            refined_cost_check = mdl.total_cost(
                refined_expr.to_bytes(), refined_preds, search_data, self.alphabet_size
            )
            if refined_cost_check < beam_cost_check:
                sym_expr = refined_expr

        # Use predict_sequence with seeds for PREV nodes
        n = len(history)
        if sym_expr.has_prev():
            max_lag = getattr(self.synthesizer, "max_lag", 3)
            seeds = history[:max_lag]  # use observed initial conditions
            sym_preds_raw = sym_expr.predict_sequence(n, self.alphabet_size, initial_history=seeds)
            # Lower lambda for PREV: recurrence expressions penalized unfairly for length
            from ouroboros.compression.mdl import MDLCost as _MDL
            prev_mdl = _MDL(lambda_weight=mdl.lambda_weight * 0.15)
            sym_cost_full = prev_mdl.total_cost(
                sym_expr.to_bytes(), sym_preds_raw, history, self.alphabet_size
            )
        else:
            sym_preds_raw = sym_expr.predict_sequence(n, self.alphabet_size)
            sym_cost_full = mdl.total_cost(
                sym_expr.to_bytes(), sym_preds_raw, history, self.alphabet_size
            )

        # ── Path 2: N-gram search (parent class) ──────────────────────────────
        ngram_cost = super().search_and_update()

        # ── Choose best ───────────────────────────────────────────────────────
        
        if sym_cost_full < ngram_cost:
            self.best_expression = sym_expr
            self.best_expression_cost = sym_cost_full
            self._using_symbolic = True
            self.symbolic_wins += 1
            return sym_cost_full
        else:
            self._using_symbolic = False
            self.ngram_wins += 1
            return ngram_cost

    def predict(self) -> int:
        """Predict next symbol (symbolic or n-gram depending on last search)."""
        if self._using_symbolic and self.best_expression is not None:
            t = len(self.observation_history)
            # predict_sequence handles PREV correctly
            preds = self.best_expression.predict_sequence(t + 1, self.alphabet_size)
            return preds[t] if preds else 0
        return super().predict()

    def measure_compression_ratio(self) -> float:
        """Measure compression ratio using best program (symbolic or n-gram)."""
        if not (self._using_symbolic and self.best_expression is not None):
            return super().measure_compression_ratio()

        history = self.observation_history
        if not history:
            return 1.0

        n = len(history)
        preds = self.best_expression.predict_sequence(n, self.alphabet_size)
        prog_bytes = self.best_expression.to_bytes()

        cost = self.mdl.total_cost(prog_bytes, preds, history, self.alphabet_size)
        nb = naive_bits(history, self.alphabet_size)
        ratio = cost / nb if nb > 0 else 1.0

        self.compression_ratios.append(ratio)
        return ratio

    def expression_string(self) -> str:
        """Return current best expression as string (or 'n-gram')."""
        if self._using_symbolic and self.best_expression:
            return self.best_expression.to_string()
        return f"n-gram(k={self.program.context_length})"

    def status_dict(self) -> dict:
        base = super().status_dict()
        base.update({
            'expression': self.expression_string(),
            'symbolic_wins': self.symbolic_wins,
            'ngram_wins': self.ngram_wins,
            'using_symbolic': self._using_symbolic,
        })
        return base

    def __repr__(self) -> str:
        ratio = self.latest_ratio()
        return (
            f"SynthAgent(id={self.agent_id}, "
            f"expr={self.expression_string()!r}, "
            f"ratio={ratio:.4f})"
        )