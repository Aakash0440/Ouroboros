"""
Counterexample representation and serialization.

A counterexample in OUROBOROS is:
    An expression E such that E achieves strictly better MDL cost
    than the proposed modification P on the test sequence.

If such an expression exists, the modification is REJECTED.
If no agent can find one after the full reveal window, it is APPROVED.

Serialization is critical: agents commit to a hash of the counterexample
bytes. The bytes must be deterministic — same expression always serializes
to the same bytes, regardless of agent or platform.

We use a canonical string form: the expression's to_string() output.
This is deterministic and human-readable.

CounterexampleResult carries everything needed for adjudication:
    - The expression itself
    - Its MDL cost on the test sequence
    - The proposer's MDL cost (for comparison)
    - Whether it actually beats the proposal
"""

import json
from dataclasses import dataclass
from typing import Optional, List
from ouroboros.compression.program_synthesis import ExprNode, BeamSearchSynthesizer
from ouroboros.compression.mcmc_refiner import MCMCRefiner
from ouroboros.compression.mdl import MDLCost, naive_bits


@dataclass
class CounterexampleResult:
    """
    Result of a counterexample search attempt.

    Fields:
        expression: The candidate counterexample expression (or None)
        ce_mdl_cost: MDL cost of the counterexample on test sequence
        proposal_mdl_cost: MDL cost of the proposal on test sequence
        is_valid_counterexample: ce_mdl_cost < proposal_mdl_cost * threshold
        agent_id: Which agent found this
        search_iterations: How many candidates were tried
    """
    expression: Optional[ExprNode]
    ce_mdl_cost: float
    proposal_mdl_cost: float
    is_valid_counterexample: bool
    agent_id: int
    search_iterations: int = 0

    def to_bytes(self) -> bytes:
        """
        Deterministic serialization for commitment hashing.

        Format: JSON with canonical keys, sorted.
        None expression → "__NULL__"
        """
        data = {
            'expression': self.expression.to_string() if self.expression else '__NULL__',
            'ce_cost': round(self.ce_mdl_cost, 4),
            'proposal_cost': round(self.proposal_mdl_cost, 4),
            'is_valid': self.is_valid_counterexample,
            'agent_id': self.agent_id,
        }
        return json.dumps(data, sort_keys=True).encode('utf-8')

    @classmethod
    def null_result(cls, agent_id: int, proposal_cost: float) -> 'CounterexampleResult':
        """No counterexample found."""
        return cls(
            expression=None,
            ce_mdl_cost=float('inf'),
            proposal_mdl_cost=proposal_cost,
            is_valid_counterexample=False,
            agent_id=agent_id
        )

    def __repr__(self) -> str:
        expr_str = self.expression.to_string() if self.expression else 'None'
        return (f"CE(agent={self.agent_id}, "
                f"valid={self.is_valid_counterexample}, "
                f"expr={expr_str!r}, "
                f"ce_cost={self.ce_mdl_cost:.1f}, "
                f"prop_cost={self.proposal_mdl_cost:.1f})")


class CounterexampleSearcher:
    """
    Searches for counterexamples to a proposed modification.

    An adversarial agent uses this to attack a proposal.
    Strategy:
        1. Try BeamSearch on test sequence (fast, structural search)
        2. Try MCMC refinement on best candidate (fine-tune constants)
        3. Check if best found expression beats proposal MDL cost

    The threshold for "valid counterexample" is configurable.
    Default: counterexample must achieve < 90% of proposal's MDL cost.
    Being stricter (e.g., 70%) means harder to attack good proposals.

    Args:
        alphabet_size: Symbol alphabet size
        beam_width: Beam search width
        max_depth: Max expression depth
        const_range: Constant search range
        mcmc_iterations: MCMC refinement steps
        validity_threshold: CE must beat proposal by this factor
        seed: Random seed
    """

    def __init__(
        self,
        alphabet_size: int = 7,
        beam_width: int = 20,
        max_depth: int = 3,
        const_range: int = 20,
        mcmc_iterations: int = 150,
        validity_threshold: float = 0.90,
        seed: int = 42
    ):
        self.alphabet_size = alphabet_size
        self.validity_threshold = validity_threshold
        self.mdl = MDLCost()
        self.beam = BeamSearchSynthesizer(
            beam_width=beam_width,
            max_depth=max_depth,
            const_range=const_range,
            alphabet_size=alphabet_size
        )
        self.mcmc = MCMCRefiner(
            num_iterations=mcmc_iterations,
            alphabet_size=alphabet_size,
            seed=seed
        )

    def compute_mdl_cost(
        self,
        expr: ExprNode,
        test_sequence: List[int]
    ) -> float:
        """Compute MDL cost of an expression on a test sequence."""
        n = len(test_sequence)
        preds = expr.predict_sequence(n, self.alphabet_size)
        return self.mdl.total_cost(
            expr.to_bytes(), preds, test_sequence, self.alphabet_size
        )

    def search(
        self,
        agent_id: int,
        proposal_expr: ExprNode,
        test_sequence: List[int],
        verbose: bool = False
    ) -> CounterexampleResult:
        """
        Search for a counterexample to proposal_expr on test_sequence.

        A counterexample beats proposal_expr under MDL:
            CE_cost < proposal_cost * validity_threshold

        Strategy:
            1. Compute proposal cost (baseline to beat)
            2. Run BeamSearch for a better program
            3. MCMC refine the best candidate
            4. Return result (valid CE or null)

        Args:
            agent_id: Searcher's ID
            proposal_expr: The expression being challenged
            test_sequence: Data to evaluate on
            verbose: Print search progress

        Returns:
            CounterexampleResult
        """
        if not test_sequence:
            return CounterexampleResult.null_result(agent_id, float('inf'))

        # Baseline: how good is the proposal?
        proposal_cost = self.compute_mdl_cost(proposal_expr, test_sequence)

        if verbose:
            print(f"  Agent {agent_id} searching for CE against {proposal_expr.to_string()!r}")
            print(f"  Proposal cost: {proposal_cost:.1f} bits")

        # Beam search for a better expression
        best_expr, beam_cost = self.beam.search(test_sequence, verbose=False)
        iterations = 1

        # MCMC refinement
        refined_expr, refined_cost = self.mcmc.refine(best_expr, test_sequence)
        iterations += 1

        # Use whichever is better
        if refined_cost < beam_cost:
            best_expr = refined_expr
            best_cost = refined_cost
        else:
            best_cost = beam_cost

        # Is this a valid counterexample?
        threshold_cost = proposal_cost * self.validity_threshold
        is_valid = best_cost < threshold_cost

        if verbose:
            print(f"  Best found: {best_expr.to_string()!r}  cost={best_cost:.1f}")
            print(f"  Threshold:  {threshold_cost:.1f}")
            print(f"  Valid CE:   {is_valid}")

        return CounterexampleResult(
            expression=best_expr if is_valid else None,
            ce_mdl_cost=best_cost,
            proposal_mdl_cost=proposal_cost,
            is_valid_counterexample=is_valid,
            agent_id=agent_id,
            search_iterations=iterations
        )