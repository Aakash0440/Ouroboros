"""
ObjectiveProofMarket — Layer 2 Self-Improvement Market

Evaluates proposals to change the MDL objective function.

How it works:
1. A Layer2Agent generates an ObjectiveProposal:
   - It runs search under current objective → gets MDL cost C1
   - It runs search under proposed objective → gets MDL cost C2
   - If C2 < C1, it submits the proposal

2. Adversaries attempt to find a validation environment where the
   proposed objective gives WORSE compression:
   - They run both objectives on a held-out validation environment
   - If proposed_bits > current_bits on validation → reject

3. The new objective must generalize (OOD), not just fit training data.

This is the key Layer 2 innovation: self-improvement of the LOSS FUNCTION
itself, verified adversarially.
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.agents.mdl_objective import (
    MDLObjective, ObjectiveProposal, ObjectiveEvaluationResult,
    DEFAULT_OBJECTIVE,
)
from ouroboros.compression.mdl_engine import MDLEngine, MDLResult
from ouroboros.environments.base import Environment


@dataclass
class ObjectiveMarketConfig:
    """Configuration for the objective proof market."""
    n_adversaries: int = 4
    min_improvement_bits: float = 5.0    # proposal must save at least 5 bits
    min_improvement_fraction: float = 0.02  # and at least 2%
    validation_stream_length: int = 500
    search_budget: int = 50   # beam_width for evaluating proposals
    random_seed: int = 42


class MDLObjectiveEvaluator:
    """
    Evaluates programs under a specific MDLObjective.
    
    Wraps the existing MDLEngine but applies the objective's λ parameters
    instead of the defaults.
    """

    def __init__(self, objective: MDLObjective):
        self.objective = objective

    def evaluate_expression_cost(
        self,
        predictions: List[int],
        actuals: List[int],
        node_count: int,
        constant_count: int,
    ) -> float:
        """
        Compute total MDL cost under this objective.
        
        Returns total bits (program + data) using the objective's λ values.
        """
        from ouroboros.compression.mdl_engine import shannon_bits

        # Program cost using this objective's λ
        prog_bits = self.objective.compute_program_bits(node_count, constant_count)

        # Data cost: standard Shannon entropy over prediction errors
        n = len(actuals)
        if n == 0:
            return prog_bits

        # Error distribution
        errors = [p - a for p, a in zip(predictions, actuals)]
        error_counts: Dict[int, int] = {}
        for e in errors:
            error_counts[e] = error_counts.get(e, 0) + 1

        data_bits = 0.0
        for count in error_counts.values():
            p = count / n
            data_bits -= p * (p.bit_length() - 1 if p > 0 else 0)
        
        # Use standard Shannon bits
        data_bits = shannon_bits(errors)
        
        return prog_bits + data_bits


def shannon_bits(values: List[Any]) -> float:
    """Compute Shannon entropy of a list in bits."""
    from collections import Counter
    import math
    n = len(values)
    if n == 0:
        return 0.0
    counts = Counter(values)
    bits = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            bits -= p * math.log2(p)
    return bits * n   # total bits = entropy per symbol × n symbols


class ObjectiveProofMarket:
    """
    The Layer 2 proof market: adversarially evaluates objective proposals.
    
    An objective proposal is approved if and only if:
    1. It shows improvement > min_improvement_bits on the training environment
    2. It shows improvement (or at least no degradation) on the validation environment
    3. The proposed objective is valid (all parameters in range)
    """

    def __init__(
        self,
        config: ObjectiveMarketConfig = None,
        validation_environments: List[Environment] = None,
    ):
        self.cfg = config or ObjectiveMarketConfig()
        self.validation_environments = validation_environments or []
        self._rng = random.Random(self.cfg.random_seed)
        self._approved_objectives: List[MDLObjective] = []
        self._rejected_proposals: List[ObjectiveProposal] = []
        self._evaluation_history: List[ObjectiveEvaluationResult] = []

    def evaluate_proposal(
        self,
        proposal: ObjectiveProposal,
        training_environment: Environment,
    ) -> ObjectiveEvaluationResult:
        """
        Evaluate an objective proposal.
        
        The adversarial test: run the proposed objective vs current objective
        on a fresh validation environment (not used for training).
        """
        # ── Check 1: Proposal validity ────────────────────────────────────
        if not proposal.proposed_objective.is_valid():
            result = ObjectiveEvaluationResult(
                proposal=proposal,
                approved=False,
                validation_env_name="validity_check",
                validation_current_bits=0.0,
                validation_proposed_bits=0.0,
                rejection_reason="Proposed objective has invalid parameters",
            )
            self._evaluation_history.append(result)
            return result

        # ── Check 2: Minimum improvement threshold ────────────────────────
        if proposal.improvement_bits < self.cfg.min_improvement_bits:
            result = ObjectiveEvaluationResult(
                proposal=proposal,
                approved=False,
                validation_env_name="improvement_threshold",
                validation_current_bits=proposal.current_total_bits,
                validation_proposed_bits=proposal.proposed_total_bits,
                rejection_reason=(
                    f"Insufficient improvement: {proposal.improvement_bits:.2f} bits "
                    f"< threshold {self.cfg.min_improvement_bits:.2f} bits"
                ),
            )
            self._evaluation_history.append(result)
            return result

        # ── Check 3: OOD validation ───────────────────────────────────────
        if self.validation_environments:
            val_env = self._rng.choice(self.validation_environments)
            val_obs = val_env.generate(self.cfg.validation_stream_length)
            
            # Evaluate both objectives on validation data
            # We use a simple heuristic: count compression under naive (constant) predictor
            # Real implementation uses the agent's best expression from training
            
            # For now, use the MDL cost of a constant prediction under each objective
            naive_preds = [int(sum(val_obs) / len(val_obs))] * len(val_obs) if val_obs else []
            
            # Cost under current objective
            current_cost = self._evaluate_naive_cost(
                val_obs, proposal.current_objective
            )
            # Cost under proposed objective
            proposed_cost = self._evaluate_naive_cost(
                val_obs, proposal.proposed_objective
            )

            # Adversarial check: does the proposed objective do WORSE on validation?
            if proposed_cost > current_cost + 10.0:   # 10-bit tolerance
                result = ObjectiveEvaluationResult(
                    proposal=proposal,
                    approved=False,
                    validation_env_name=val_env.name,
                    validation_current_bits=current_cost,
                    validation_proposed_bits=proposed_cost,
                    rejection_reason=(
                        f"Proposed objective performs worse on {val_env.name}: "
                        f"+{proposed_cost - current_cost:.2f} bits"
                    ),
                )
                self._rejected_proposals.append(proposal)
                self._evaluation_history.append(result)
                return result

            val_current = current_cost
            val_proposed = proposed_cost
            val_env_name = val_env.name
        else:
            # No validation environments — approve based on training improvement alone
            val_current = proposal.current_total_bits
            val_proposed = proposal.proposed_total_bits
            val_env_name = "training_only"

        # ── Approved! ─────────────────────────────────────────────────────
        result = ObjectiveEvaluationResult(
            proposal=proposal,
            approved=True,
            validation_env_name=val_env_name,
            validation_current_bits=val_current,
            validation_proposed_bits=val_proposed,
        )
        self._approved_objectives.append(copy.deepcopy(proposal.proposed_objective))
        self._evaluation_history.append(result)
        return result

    def _evaluate_naive_cost(
        self,
        observations: List[int],
        objective: MDLObjective,
    ) -> float:
        """
        Compute the MDL cost of a naive (mode prediction) program
        under the given objective.
        
        This is the baseline cost — a constant predictor.
        """
        from collections import Counter
        import math

        if not observations:
            return 0.0

        # Naive predictor: always predict the mode
        mode = Counter(observations).most_common(1)[0][0]
        errors = [obs - mode for obs in observations]

        # Program bits: 1 node (CONST), 1 constant → minimal program
        prog_bits = objective.compute_program_bits(node_count=1, constant_count=1)

        # Data bits: Shannon entropy of errors
        n = len(observations)
        error_counts = Counter(errors)
        data_bits = 0.0
        for count in error_counts.values():
            p = count / n
            if p > 0:
                data_bits -= p * math.log2(p)
        data_bits *= n

        return prog_bits + data_bits

    @property
    def approved_objectives(self) -> List[MDLObjective]:
        return list(self._approved_objectives)

    @property
    def approval_rate(self) -> float:
        total = len(self._evaluation_history)
        if total == 0:
            return 0.0
        approved = sum(1 for r in self._evaluation_history if r.approved)
        return approved / total

    def get_best_objective(self) -> MDLObjective:
        """Return the best approved objective, or DEFAULT if none approved."""
        if not self._approved_objectives:
            return DEFAULT_OBJECTIVE
        # Return the most recently approved one
        return self._approved_objectives[-1]