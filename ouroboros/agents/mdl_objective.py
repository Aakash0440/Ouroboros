"""
MDLObjective — OUROBOROS Layer 2 Self-Improvement

In Layer 1 (HyperparameterAgent, Day 20), agents modify their SEARCH parameters:
  beam_width, mcmc_iterations, const_range, max_depth

In Layer 2 (today), agents modify their OBJECTIVE FUNCTION:
  lambda_prog    — how many bits to charge per AST node
  lambda_const   — how many bits to charge per constant
  good_fit_threshold — R² threshold for axiom candidacy (continuous)
  axiom_consensus    — fraction of agents needed for axiom promotion

The insight: the MDL objective is not fixed truth — it's a prior about
how complex programs should be penalized. A better λ leads to:
  1. Finding simpler programs that generalize better (OOD performance)
  2. Promoting axioms with higher empirical validity
  3. Faster convergence on the proof market

The proof market for Layer 2: 
  - Agent proposes a new (lambda_prog, lambda_const) pair
  - Adversaries search for a held-out validation environment
    where the new objective gives WORSE compression than the current one
  - If no adversary can find such an environment → objective approved
"""

from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MDLObjective:
    """
    A complete MDL objective function, including all λ parameters.
    
    This is the object agents propose to modify in Layer 2.
    """
    # Discrete MDL parameters
    lambda_prog: float = 2.0       # bits per AST node (program complexity)
    lambda_const: float = 8.0      # bits per float constant
    
    # Continuous MDL parameters
    gaussian_min_sigma: float = 1e-6
    
    # Axiom promotion thresholds
    axiom_consensus_fraction: float = 0.5   # fraction of agents needed
    axiom_r2_threshold: float = 0.95        # for continuous axioms
    
    # Regularization parameters
    complexity_penalty_exp: float = 1.0     # exponent on node_count
    
    def description_bits(self) -> float:
        """
        How many bits does it take to describe this objective?
        The objective is itself subject to MDL reasoning.
        Each parameter encoded at float precision ≈ 32 bits.
        """
        return 32.0 * 5   # 5 float parameters

    def compute_program_bits(self, node_count: int, constant_count: int) -> float:
        """Compute program description bits under this objective."""
        return (
            self.lambda_prog * (node_count ** self.complexity_penalty_exp)
            + self.lambda_const * constant_count
        )

    def to_dict(self) -> dict:
        return {
            "lambda_prog": self.lambda_prog,
            "lambda_const": self.lambda_const,
            "gaussian_min_sigma": self.gaussian_min_sigma,
            "axiom_consensus_fraction": self.axiom_consensus_fraction,
            "axiom_r2_threshold": self.axiom_r2_threshold,
            "complexity_penalty_exp": self.complexity_penalty_exp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'MDLObjective':
        return cls(**d)

    def is_valid(self) -> bool:
        """Check that all parameters are in reasonable ranges."""
        return (
            0.1 <= self.lambda_prog <= 20.0
            and 1.0 <= self.lambda_const <= 50.0
            and 1e-9 <= self.gaussian_min_sigma <= 0.1
            and 0.1 <= self.axiom_consensus_fraction <= 1.0
            and 0.5 <= self.axiom_r2_threshold <= 1.0
            and 0.5 <= self.complexity_penalty_exp <= 3.0
        )

    def clamp(self) -> 'MDLObjective':
        """Return a copy with all parameters clamped to valid ranges."""
        return MDLObjective(
            lambda_prog=max(0.1, min(20.0, self.lambda_prog)),
            lambda_const=max(1.0, min(50.0, self.lambda_const)),
            gaussian_min_sigma=max(1e-9, min(0.1, self.gaussian_min_sigma)),
            axiom_consensus_fraction=max(0.1, min(1.0, self.axiom_consensus_fraction)),
            axiom_r2_threshold=max(0.5, min(1.0, self.axiom_r2_threshold)),
            complexity_penalty_exp=max(0.5, min(3.0, self.complexity_penalty_exp)),
        )

    def description(self) -> str:
        return (
            f"MDLObj(λ_prog={self.lambda_prog:.3f}, λ_const={self.lambda_const:.3f}, "
            f"consensus={self.axiom_consensus_fraction:.2f}, "
            f"r²_thresh={self.axiom_r2_threshold:.2f})"
        )


# Default OUROBOROS objective (as used in Days 1–20)
DEFAULT_OBJECTIVE = MDLObjective(
    lambda_prog=2.0,
    lambda_const=8.0,
    gaussian_min_sigma=1e-6,
    axiom_consensus_fraction=0.5,
    axiom_r2_threshold=0.95,
    complexity_penalty_exp=1.0,
)


@dataclass
class ObjectiveProposal:
    """
    An agent's proposal to change its MDL objective.
    
    Contains:
    - The proposed new objective
    - Evidence: MDL cost improvement on training data under the new objective
    - The agent that made the proposal
    """
    proposing_agent: str
    current_objective: MDLObjective
    proposed_objective: MDLObjective
    training_env_name: str
    
    # Evidence: cost under current vs proposed objective (lower = better)
    current_total_bits: float
    proposed_total_bits: float
    
    # Derived
    improvement_bits: float = field(init=False)
    improvement_fraction: float = field(init=False)

    def __post_init__(self):
        self.improvement_bits = self.current_total_bits - self.proposed_total_bits
        if self.current_total_bits > 0:
            self.improvement_fraction = self.improvement_bits / self.current_total_bits
        else:
            self.improvement_fraction = 0.0

    @property
    def is_improvement(self) -> bool:
        return self.improvement_bits > 0.0

    def description(self) -> str:
        return (
            f"ObjectiveProposal by {self.proposing_agent}\n"
            f"  Current:  {self.current_objective.description()}\n"
            f"  Proposed: {self.proposed_objective.description()}\n"
            f"  Improvement: {self.improvement_bits:.2f} bits "
            f"({self.improvement_fraction*100:.1f}%)"
        )


@dataclass
class ObjectiveEvaluationResult:
    """Result of the proof market evaluating an objective proposal."""
    proposal: ObjectiveProposal
    approved: bool
    
    # The validation environment used for adversarial testing
    validation_env_name: str
    
    # MDL costs on validation environment under current vs proposed objective
    validation_current_bits: float
    validation_proposed_bits: float
    
    # Did proposed objective perform better on OOD validation?
    validation_improvement: float = field(init=False)
    
    rejection_reason: Optional[str] = None

    def __post_init__(self):
        self.validation_improvement = (
            self.validation_current_bits - self.validation_proposed_bits
        )

    def description(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        return (
            f"{status}: Objective proposal by {self.proposal.proposing_agent}\n"
            f"  Training improvement: {self.proposal.improvement_bits:.2f} bits\n"
            f"  Validation improvement: {self.validation_improvement:.2f} bits\n"
            f"  Verdict: {self.rejection_reason or 'Genuine improvement'}"
        )