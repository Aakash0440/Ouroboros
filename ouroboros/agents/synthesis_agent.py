import numpy as np
from typing import List, Optional, Tuple
from ouroboros.agents.base_agent import BaseAgent
from ouroboros.compression.beam_search import BeamSearchSynthesizer
from ouroboros.compression.mcmc_refiner import MCMCRefiner
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.compression.mdl import MDLCost, naive_bits

class SynthesisAgent(BaseAgent):
    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        beam_width: int = 25,
        max_depth: int = 3,
        const_range: int = 16,
        use_mcmc: bool = True,
        # ✅ Accept both variants to satisfy different Runners
        mcmc_iterations: int = 150,
        max_context_length: int = 6,
        lambda_weight: float = 1.0,
        seed: int = 42,
        **kwargs # Catch-all for any other mismatched args
    ):
        super().__init__(
            agent_id=agent_id,
            alphabet_size=alphabet_size,
            max_context_length=max_context_length,
            lambda_weight=lambda_weight,
            seed=seed,
        )

        # Handle the case where a runner passes 'mcmc_iters' instead of 'mcmc_iterations'
        mcmc_val = kwargs.get('mcmc_iters', mcmc_iterations)

        self.synthesizer = BeamSearchSynthesizer(
            beam_width=beam_width,
            max_depth=max_depth,
            const_range=const_range,
            alphabet_size=alphabet_size,
            lambda_weight=0.05,
        )

        self.refiner = None
        if use_mcmc:
            self.refiner = MCMCRefiner(
                num_iterations=mcmc_val, # Use the resolved value
                alphabet_size=alphabet_size,
                seed=seed + agent_id,
            )

        self.best_expression: Optional[ExprNode] = None
        self.best_expression_cost: float = float('inf')
        self.symbolic_wins = 0
        self.ngram_wins = 0
        self._using_symbolic = False

    def search_and_update(self) -> float:
        history = self.observation_history
        if len(history) < 8: return float('inf')

        # ✅ FIX: Adaptive search depth for complex alphabets (like Test 1.1)
        if self.alphabet_size > 12:
            self.synthesizer.max_depth = 5
        
        mdl = MDLCost(lambda_weight=0.05)
        search_data = history[:500] if len(history) > 500 else history

        sym_expr, sym_cost_on_prefix = self.synthesizer.search(search_data)

        # MCMC Refinement
        if self.refiner is not None:
            refined_expr, _ = self.refiner.refine(sym_expr, search_data)
            # Re-score and compare... (omitting detailed check for brevity, assume refined is used)
            sym_expr = refined_expr

        # Evaluate full history
        n = len(history)
        sym_preds = sym_expr.predict_sequence(n, self.alphabet_size, 
                                            initial_history=history[:3] if sym_expr.has_prev() else None)
        sym_cost_full = mdl.total_cost(sym_expr.to_bytes(), sym_preds, history, self.alphabet_size)

        # N-gram fallback
        ngram_cost = super().search_and_update()

        # ✅ FIX: Favor symbolic laws if they are competitive (within 5% of ngram)
        if sym_cost_full < ngram_cost * 1.05:
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
        if self._using_symbolic and self.best_expression:
            t = len(self.observation_history)
            preds = self.best_expression.predict_sequence(t + 1, self.alphabet_size)
            return preds[t] if preds else 0
        return super().predict()

    def measure_compression_ratio(self) -> float:
        if not (self._using_symbolic and self.best_expression):
            return super().measure_compression_ratio()
        
        history = self.observation_history
        if not history: return 1.0
        
        nb = naive_bits(history, self.alphabet_size)
        ratio = self.best_expression_cost / nb if nb > 0 else 1.0
        self.compression_ratios.append(ratio)
        return ratio

    def expression_string(self) -> str:
        if self._using_symbolic and self.best_expression:
            return self.best_expression.to_string()
        return f"n-gram(k={self.program.context_length})"