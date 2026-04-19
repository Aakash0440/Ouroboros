"""
HyperparameterAgent — Layer 1 recursive self-improvement.

An agent that can propose modifications to its own search hyperparameters
through the proof market, subject to the same adversarial verification
and OOD pressure as expression modifications.

What hyperparameters can be modified:
    beam_width:       Number of beam search candidates (1..100)
    mcmc_iterations:  MCMC refinement steps (10..1000)
    const_range:      Constant search range (4..200)
    max_depth:        Expression tree depth (1..5)
    max_lag:          PREV node maximum lag (1..10)

How modification works:
    1. Agent proposes new hyperparameter values
    2. The proposed values are tested: run search with new params on fresh data
    3. Compare MDL cost: do new params find a better expression?
    4. If yes (improvement > threshold) → submit to proof market
    5. Market evaluates: do other agents agree the new params are better?
    6. OOD test: do the new params generalize to novel environments?
    7. If all pass → agent permanently updates its hyperparameters

Why this is genuine recursive self-improvement:
    The agent is modifying the PROCEDURE it uses to find programs,
    not just the programs themselves. Better search procedure →
    better programs → better compression → more structure discovered.
    This is the first step toward the full Gödel Machine vision.

Why Layer 1 and not full Gödel Machine:
    Modifying the search algorithm STRUCTURE (beam vs MCMC vs random)
    requires the search algorithm itself to be represented as a program
    that can be searched over — a meta-level language.
    That's Layer 3 and requires months more work.
    Layer 1 (hyperparameter tuning) is tractable now and already novel.

Args:
    agent_id: Unique identifier
    alphabet_size: Symbol alphabet size
    initial_hyperparams: Starting hyperparameter values (defaults used if None)
    hyperparam_mod_threshold: Min MDL improvement (bits) to propose a change
    (all other args forwarded to SynthesisAgent)
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
import copy
import numpy as np
from ouroboros.agents.synthesis_agent import SynthesisAgent
from ouroboros.compression.program_synthesis import (
    BeamSearchSynthesizer, ExprNode
)
from ouroboros.compression.mcmc_refiner import MCMCRefiner
from ouroboros.compression.mdl import MDLCost, naive_bits
from ouroboros.utils.logger import get_logger


@dataclass
class HyperparameterSet:
    """
    A complete set of search hyperparameters.

    These define HOW the agent searches — not WHAT it finds.
    Modifying these changes the agent's search capability.

    Constraints:
        beam_width:      integer in [1, 100]
        mcmc_iterations: integer in [10, 1000]
        const_range:     integer in [4, 200]
        max_depth:       integer in [1, 5]
        max_lag:         integer in [1, 10]
    """
    beam_width: int = 25
    mcmc_iterations: int = 200
    const_range: int = 16
    max_depth: int = 3
    max_lag: int = 3

    # Bounds for each parameter
    BOUNDS: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'beam_width':      (1, 100),
        'mcmc_iterations': (10, 1000),
        'const_range':     (4, 200),
        'max_depth':       (1, 5),
        'max_lag':         (1, 10),
    })

    def clamp(self) -> 'HyperparameterSet':
        """Return a new HyperparameterSet with all values clamped to bounds."""
        bounds = self.BOUNDS
        return HyperparameterSet(
            beam_width=max(bounds['beam_width'][0],
                          min(self.beam_width, bounds['beam_width'][1])),
            mcmc_iterations=max(bounds['mcmc_iterations'][0],
                               min(self.mcmc_iterations,
                                   bounds['mcmc_iterations'][1])),
            const_range=max(bounds['const_range'][0],
                           min(self.const_range, bounds['const_range'][1])),
            max_depth=max(bounds['max_depth'][0],
                         min(self.max_depth, bounds['max_depth'][1])),
            max_lag=max(bounds['max_lag'][0],
                       min(self.max_lag, bounds['max_lag'][1])),
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            'beam_width': self.beam_width,
            'mcmc_iterations': self.mcmc_iterations,
            'const_range': self.const_range,
            'max_depth': self.max_depth,
            'max_lag': self.max_lag,
        }

    def compute_cost(self) -> int:
        """
        Compute runtime cost of these hyperparameters.
        Higher = more expensive search.
        Used to penalize excessively expensive configurations.
        """
        return (self.beam_width *
                self.mcmc_iterations *
                self.max_depth)

    def description_bits(self) -> float:
        """
        Bits to describe this hyperparameter set.
        Used in MDL-style comparison of hyperparameter sets.
        """
        import math
        bits = 0
        for param, (lo, hi) in self.BOUNDS.items():
            val = getattr(self, param)
            bits += math.log2(hi - lo + 1)
        return bits

    def __repr__(self) -> str:
        return (f"HP(bw={self.beam_width}, "
                f"mcmc={self.mcmc_iterations}, "
                f"cr={self.const_range}, "
                f"d={self.max_depth}, "
                f"lag={self.max_lag})")

    def __eq__(self, other) -> bool:
        if not isinstance(other, HyperparameterSet):
            return False
        return self.to_dict() == other.to_dict()


@dataclass
class HyperparameterProposal:
    """
    A proposed change to hyperparameters.

    Fields:
        agent_id: Proposing agent
        current_hp: Current hyperparameters
        proposed_hp: Proposed new hyperparameters
        current_best_cost: MDL cost with current HP on validation data
        proposed_best_cost: MDL cost with proposed HP on validation data
        improvement_bits: current_cost - proposed_cost (positive = better)
        validation_data: Data used for comparison
        alphabet_size: Symbol alphabet
        changed_param: Which parameter changed (for logging)
        change_direction: 'increase' or 'decrease'
    """
    agent_id: int
    current_hp: HyperparameterSet
    proposed_hp: HyperparameterSet
    current_best_cost: float
    proposed_best_cost: float
    improvement_bits: float
    validation_data: List[int]
    alphabet_size: int
    changed_param: str = ''
    change_direction: str = ''

    def is_improvement(self) -> bool:
        return self.improvement_bits > 0

    def to_market_proposal(self) -> Optional[Tuple[ExprNode, ExprNode]]:
        """
        Convert to (current_expr, proposed_expr) for ProofMarket.

        We encode hyperparameter sets as CONST expressions for market
        compatibility. The market evaluates the MDL improvement on
        validation data.

        This is a proxy: the market evaluates whether the new HP setting
        finds a better expression, which it does if proposed_best_cost
        < current_best_cost.
        """
        from ouroboros.compression.program_synthesis import C
        # Encode HP as sum of params (used as a unique identifier)
        current_val = sum(self.current_hp.to_dict().values())
        proposed_val = sum(self.proposed_hp.to_dict().values())
        return C(current_val % 256), C(proposed_val % 256)

    def __repr__(self) -> str:
        return (f"HPProposal(agent={self.agent_id}, "
                f"{self.changed_param} "
                f"{self.change_direction}: "
                f"Δ={self.improvement_bits:.1f}bits)")


class HyperparameterAgent(SynthesisAgent):
    """
    Agent with Layer 1 recursive self-improvement.

    Modifies its own search hyperparameters through the proof market.

    Search procedure:
    1. Run standard synthesis (parent class)
    2. Periodically: propose hyperparameter changes
       For each HP parameter: try increase AND decrease
       Run search with each candidate on validation data
       If improvement > threshold: generate HyperparameterProposal
    3. Submit proposal to proof market (same mechanism as expression proposals)
    4. If approved + OOD: update own hyperparameters permanently

    The recursive loop:
        Better HP → better search → better expressions → better compression
        → more confident axioms → higher market approval of future proposals
        → agent updates HP again → loop

    Args:
        agent_id: Unique identifier
        alphabet_size: Symbol alphabet size
        initial_hp: Starting HyperparameterSet (defaults if None)
        hp_mod_threshold: Min MDL improvement (bits) to propose HP change
        hp_eval_stream_length: Length of validation stream for HP comparison
        hp_mod_frequency: How often to check for HP improvements (rounds)
        (all other args forwarded to SynthesisAgent)
    """

    def __init__(
        self,
        agent_id: int,
        alphabet_size: int,
        initial_hp: Optional[HyperparameterSet] = None,
        hp_mod_threshold: float = 5.0,
        hp_eval_stream_length: int = 200,
        hp_mod_frequency: int = 5,
        **kwargs
    ):
        # Initialize with hyperparameters
        hp = initial_hp or HyperparameterSet()

        # Override parent kwargs with HP values
        kwargs['beam_width'] = hp.beam_width
        kwargs['mcmc_iterations'] = hp.mcmc_iterations
        kwargs['const_range'] = hp.const_range
        kwargs['max_depth'] = hp.max_depth

        super().__init__(agent_id, alphabet_size, **kwargs)

        self.current_hp = hp
        self.hp_mod_threshold = hp_mod_threshold
        self.hp_eval_stream_length = hp_eval_stream_length
        self.hp_mod_frequency = hp_mod_frequency
        self.logger = get_logger(f'HPAgent_{agent_id}')

        # History of hyperparameter modifications
        self.hp_history: List[Tuple[int, HyperparameterSet, float]] = []
        # (round, new_hp, improvement_bits)

        self.hp_approved: int = 0
        self.hp_rejected: int = 0
        self.hp_proposed: int = 0
        self._rounds_since_hp_check: int = 0

    def _run_search_with_hp(
        self,
        hp: HyperparameterSet,
        data: List[int]
    ) -> Tuple[Optional[ExprNode], float]:
        """
        Run beam+MCMC search with a specific hyperparameter set.
        Returns (best_expression, best_mdl_cost).
        """
        synth = BeamSearchSynthesizer(
            beam_width=hp.beam_width,
            max_depth=hp.max_depth,
            const_range=hp.const_range,
            alphabet_size=self.alphabet_size,
            max_lag=hp.max_lag,
        )
        refiner = MCMCRefiner(
            num_iterations=hp.mcmc_iterations,
            alphabet_size=self.alphabet_size,
            const_range=hp.const_range,
        )

        search_data = data[:min(500, len(data))]
        expr, cost = synth.search(search_data)
        refined, refined_cost = refiner.refine(expr, search_data)

        if refined_cost < cost:
            return refined, refined_cost
        return expr, cost

    def _candidate_hp_sets(self) -> List[Tuple[str, str, HyperparameterSet]]:
        """
        Generate candidate hyperparameter modifications.

        For each parameter, try:
            - Small increase (+delta)
            - Small decrease (-delta)

        Returns list of (param_name, direction, candidate_hp).
        """
        candidates = []
        bounds = self.current_hp.BOUNDS

        deltas = {
            'beam_width':      5,
            'mcmc_iterations': 50,
            'const_range':     4,
            'max_depth':       1,
            'max_lag':         1,
        }

        for param, delta in deltas.items():
            current_val = getattr(self.current_hp, param)
            lo, hi = bounds[param]

            # Try increase
            new_val_up = min(current_val + delta, hi)
            if new_val_up != current_val:
                hp_up = copy.copy(self.current_hp)
                setattr(hp_up, param, new_val_up)
                candidates.append((param, 'increase', hp_up.clamp()))

            # Try decrease
            new_val_down = max(current_val - delta, lo)
            if new_val_down != current_val:
                hp_down = copy.copy(self.current_hp)
                setattr(hp_down, param, new_val_down)
                candidates.append((param, 'decrease', hp_down.clamp()))

        return candidates

    def generate_hp_proposal(
        self,
        validation_data: List[int]
    ) -> Optional[HyperparameterProposal]:
        """
        Search for a hyperparameter improvement on validation_data.

        Tries all candidate HP sets.
        Returns the best improvement as a HyperparameterProposal,
        or None if no improvement exceeds the threshold.

        Args:
            validation_data: Fresh data for HP comparison

        Returns:
            HyperparameterProposal or None
        """
        self._rounds_since_hp_check += 1
        if self._rounds_since_hp_check < self.hp_mod_frequency:
            return None
        self._rounds_since_hp_check = 0

        if len(validation_data) < 50:
            return None

        eval_data = validation_data[:self.hp_eval_stream_length]

        # Score current HP
        _, current_cost = self._run_search_with_hp(self.current_hp, eval_data)

        best_improvement = 0.0
        best_candidate = None
        best_param = ''
        best_direction = ''
        best_cost = current_cost

        for param, direction, candidate_hp in self._candidate_hp_sets():
            # Skip if candidate is identical to current
            if candidate_hp == self.current_hp:
                continue

            _, candidate_cost = self._run_search_with_hp(
                candidate_hp, eval_data
            )
            improvement = current_cost - candidate_cost

            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = candidate_hp
                best_param = param
                best_direction = direction
                best_cost = candidate_cost

        if best_improvement < self.hp_mod_threshold or best_candidate is None:
            return None

        self.hp_proposed += 1
        proposal = HyperparameterProposal(
            agent_id=self.agent_id,
            current_hp=copy.copy(self.current_hp),
            proposed_hp=best_candidate,
            current_best_cost=current_cost,
            proposed_best_cost=best_cost,
            improvement_bits=best_improvement,
            validation_data=eval_data[:100],
            alphabet_size=self.alphabet_size,
            changed_param=best_param,
            change_direction=best_direction,
        )

        self.logger.info(
            f"Agent {self.agent_id}: HP proposal "
            f"{best_param} {best_direction} "
            f"(Δ={best_improvement:.1f}bits)"
        )

        return proposal

    def apply_hp_modification(
        self,
        proposal: HyperparameterProposal,
        round_num: int
    ) -> None:
        """
        Permanently update hyperparameters after market approval.

        This updates the synthesizer and refiner objects too,
        so all future searches use the new hyperparameters.
        """
        old_hp = copy.copy(self.current_hp)
        self.current_hp = proposal.proposed_hp

        # Update synthesizer and refiner with new HP
        self.synthesizer = BeamSearchSynthesizer(
            beam_width=self.current_hp.beam_width,
            max_depth=self.current_hp.max_depth,
            const_range=self.current_hp.const_range,
            alphabet_size=self.alphabet_size,
            max_lag=self.current_hp.max_lag,
        )
        self.refiner = MCMCRefiner(
            num_iterations=self.current_hp.mcmc_iterations,
            alphabet_size=self.alphabet_size,
            const_range=self.current_hp.const_range,
        )

        self.hp_approved += 1
        self.hp_history.append((
            round_num,
            copy.copy(self.current_hp),
            proposal.improvement_bits
        ))

        self.logger.info(
            f"Agent {self.agent_id}: HP UPDATED at round {round_num}. "
            f"{old_hp} → {self.current_hp}"
        )

    def record_hp_rejection(
        self,
        proposal: HyperparameterProposal
    ) -> None:
        """Log a rejected HP modification."""
        self.hp_rejected += 1
        self.logger.debug(
            f"Agent {self.agent_id}: HP rejected "
            f"({proposal.changed_param} {proposal.change_direction})"
        )

    def hp_improvement_score(self) -> float:
        """
        Measure total recursive self-improvement from HP changes.

        Score = total improvement bits from all approved HP modifications.
        Higher = more recursive improvement achieved through HP tuning.
        """
        if not self.hp_history:
            return 0.0
        return sum(imp for _, _, imp in self.hp_history)

    def hp_summary(self) -> str:
        lines = [
            f"HyperparameterAgent {self.agent_id}:",
            f"  Current HP: {self.current_hp}",
            f"  Proposed: {self.hp_proposed}  "
            f"Approved: {self.hp_approved}  "
            f"Rejected: {self.hp_rejected}",
            f"  HP improvement score: {self.hp_improvement_score():.2f} bits",
        ]
        if self.hp_history:
            lines.append("  History:")
            for round_num, hp, imp in self.hp_history:
                lines.append(f"    Round {round_num}: {hp}  Δ={imp:.1f}bits")
        return '\n'.join(lines)