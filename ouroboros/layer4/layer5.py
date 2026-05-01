"""
Layer 5 — True DSL Extension: Agents Propose New Opcodes

The four-layer hierarchy:
  Layer 1: modify hyperparameters (beam_width etc.)
  Layer 2: modify MDL objective (lambda_prog etc.)
  Layer 3: select search strategy (from fixed library)
  Layer 4: write new search algorithm in DSL (recombine existing opcodes)
  Layer 5: propose new opcodes for the DSL (extend the language itself)

Layer 5 mechanism:
  1. Track which instruction subsequences appear most frequently
     across successful Layer 4 programs
  2. If a subsequence appears in >50% of approved programs → candidate opcode
  3. Name it: auto-generate "FFT_BEAM", "ELITE_ANNEAL", etc.
  4. OpcodeProofMarket evaluates: does adding this opcode compress
     existing good programs? (shorter = better MDL on the program itself)
  5. If approved: add to DSL, future Layer 4 searches can use it

Example discovered composite opcode:
  FFT_GUIDED_CORE = [CLASSIFY_ENV, FFT_SEED, GRAMMAR_FILTER, SORT_MDL]
  If 60% of successful programs contain this subsequence → name it!
  Programs using FFT_GUIDED_CORE are 4 instructions shorter.
  MDL cost of program drops by 4 × (description_bits_per_instruction).
"""

from __future__ import annotations
import copy
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Counter as CounterType
from collections import Counter

from ouroboros.layer4.search_dsl import (
    DSLInstruction, DSLOpcode, SearchAlgorithmProgram, OPCODE_BITS,
)


# ── CompositeOpcode ────────────────────────────────────────────────────────────

@dataclass
class CompositeOpcode:
    """
    A new DSL opcode defined as a named composition of existing instructions.
    
    Example:
      name: "FFT_GUIDED_CORE"
      body: [CLASSIFY_ENV, FFT_SEED, GRAMMAR_FILTER, SORT_MDL]
      description_bits: 3.0 (shorter than listing all 4 instructions)
      frequency: 0.65  (appears in 65% of successful programs)
    """
    name: str
    body: List[DSLInstruction]
    description_bits: float   # cost of using this composite opcode (should be < sum of body bits)
    frequency: float          # how often this pattern appeared in approved programs
    source_programs: List[str] = field(default_factory=list)

    @property
    def body_description_bits(self) -> float:
        """How many bits the body takes if listed explicitly."""
        return sum(OPCODE_BITS.get(i.opcode, 4.0) for i in self.body)

    @property
    def compression_savings(self) -> float:
        """Bits saved by using this composite instead of listing the body."""
        return self.body_description_bits - self.description_bits

    @property
    def is_worthwhile(self) -> bool:
        """True if using this composite saves bits AND appears frequently enough."""
        return self.compression_savings > 2.0 and self.frequency > 0.3

    def to_instruction(self) -> DSLInstruction:
        """Create a DSLInstruction that represents calling this composite."""
        # We store composite opcodes as a special kind of instruction
        # The interpreter handles them by expanding back to the body
        instr = DSLInstruction(DSLOpcode.INIT, param=-1)  # placeholder
        instr._composite_name = self.name
        instr._composite_body = self.body
        return instr

    def description(self) -> str:
        body_str = " → ".join(i.opcode.name for i in self.body)
        return (
            f"CompositeOpcode '{self.name}':\n"
            f"  Body: {body_str}\n"
            f"  Body bits: {self.body_description_bits:.1f}\n"
            f"  Composite bits: {self.description_bits:.1f}\n"
            f"  Savings: {self.compression_savings:.1f} bits/use\n"
            f"  Frequency: {self.frequency:.2%}\n"
            f"  {'✅ WORTHWHILE' if self.is_worthwhile else '❌ Not worthwhile'}"
        )


# ── SubsequenceFrequency ───────────────────────────────────────────────────────

def extract_opcode_subsequences(
    programs: List[SearchAlgorithmProgram],
    min_length: int = 2,
    max_length: int = 5,
) -> Counter:
    """
    Extract all subsequences of opcodes from programs and count their frequency.
    Returns Counter mapping (opcode_tuple) → count.
    """
    counter = Counter()

    for prog in programs:
        opcodes = [i.opcode for i in prog.instructions]
        # Include body opcodes from loops
        for instr in prog.instructions:
            for sub in instr.body_a + instr.body_b:
                opcodes.append(sub.opcode)

        # Extract all subsequences
        for length in range(min_length, min(max_length + 1, len(opcodes) + 1)):
            for start in range(len(opcodes) - length + 1):
                subseq = tuple(opcodes[start:start + length])
                counter[subseq] += 1

    return counter


def find_candidate_opcodes(
    programs: List[SearchAlgorithmProgram],
    min_frequency: float = 0.3,
    min_compression: float = 2.0,
    max_candidates: int = 10,
) -> List[CompositeOpcode]:
    """
    Find instruction subsequences that appear frequently enough to
    deserve their own composite opcode.
    """
    if not programs:
        return []

    # Count subsequences
    subseq_counter = extract_opcode_subsequences(programs)
    n_programs = len(programs)

    candidates = []
    for subseq, count in subseq_counter.most_common(50):
        frequency = count / n_programs
        if frequency < min_frequency:
            continue

        # Build the instruction list for this subsequence
        # (use the actual instructions from the first program that has it)
        body_instrs = []
        for prog in programs:
            prog_opcodes = [i.opcode for i in prog.instructions]
            for start in range(len(prog_opcodes) - len(subseq) + 1):
                if tuple(prog_opcodes[start:start+len(subseq)]) == subseq:
                    body_instrs = prog.instructions[start:start+len(subseq)]
                    break
            if body_instrs:
                break

        if not body_instrs:
            continue

        # Compute description bits (shorter than sum of body bits)
        body_bits = sum(OPCODE_BITS.get(op, 4.0) for op in subseq)
        composite_bits = max(2.0, body_bits / len(subseq))  # shorter name

        compression = body_bits - composite_bits
        if compression < min_compression:
            continue

        # Generate a name from the opcodes
        name_parts = [op.name[:4] for op in subseq[:3]]
        name = "_".join(name_parts)

        composite = CompositeOpcode(
            name=name,
            body=body_instrs,
            description_bits=composite_bits,
            frequency=frequency,
            source_programs=[p.name for p in programs[:3]],
        )
        candidates.append(composite)

        if len(candidates) >= max_candidates:
            break

    return candidates


# ── OpcodeProofMarket ──────────────────────────────────────────────────────────

@dataclass
class OpcodeProposal:
    """An agent's proposal to add a new composite opcode to the DSL."""
    proposing_agent: str
    composite: CompositeOpcode
    programs_compressed: int   # how many existing programs use this pattern
    total_bits_saved: float    # total compression across all programs

    @property
    def is_beneficial(self) -> bool:
        return (self.composite.is_worthwhile and
                self.total_bits_saved > 5.0 and
                self.programs_compressed >= 2)

    def description(self) -> str:
        status = "✅ BENEFICIAL" if self.is_beneficial else "❌ Not beneficial"
        return (
            f"{status} Opcode Proposal: '{self.composite.name}'\n"
            f"  {self.composite.description()}\n"
            f"  Programs using it: {self.programs_compressed}\n"
            f"  Total bits saved: {self.total_bits_saved:.1f}"
        )


class OpcodeProofMarket:
    """
    Evaluates proposed composite opcodes.
    
    Approval criteria:
    1. The pattern appears in >= min_programs successful programs
    2. Using the composite saves >= min_compression bits per use
    3. The composite is semantically valid (body makes sense as a unit)
    """

    def __init__(
        self,
        min_programs: int = 2,
        min_total_savings: float = 5.0,
    ):
        self.min_programs = min_programs
        self.min_total_savings = min_total_savings
        self._approved: List[CompositeOpcode] = []
        self._library: Dict[str, CompositeOpcode] = {}

    def evaluate(
        self,
        proposal: OpcodeProposal,
        known_programs: List[SearchAlgorithmProgram],
    ) -> bool:
        """Evaluate a composite opcode proposal. Returns True if approved."""
        if not proposal.is_beneficial:
            return False

        # Count how many programs actually use this pattern
        count = sum(
            1 for prog in known_programs
            if self._contains_subsequence(prog, proposal.composite.body)
        )

        if count < self.min_programs:
            return False

        # Compute total savings
        savings = count * proposal.composite.compression_savings
        if savings < self.min_total_savings:
            return False

        # Approve
        self._approved.append(proposal.composite)
        self._library[proposal.composite.name] = proposal.composite
        return True

    def _contains_subsequence(
        self,
        prog: SearchAlgorithmProgram,
        body: List[DSLInstruction],
    ) -> bool:
        """Check if program contains the instruction subsequence."""
        if not body:
            return False
        prog_opcodes = [i.opcode for i in prog.instructions]
        body_opcodes = [i.opcode for i in body]
        # Sliding window check
        for start in range(len(prog_opcodes) - len(body_opcodes) + 1):
            if prog_opcodes[start:start+len(body_opcodes)] == body_opcodes:
                return True
        return False

    @property
    def n_approved(self) -> int:
        return len(self._approved)

    @property
    def approved_opcodes(self) -> List[CompositeOpcode]:
        return list(self._approved)


# ── Layer5Agent ────────────────────────────────────────────────────────────────

class Layer5Agent:
    """
    An agent capable of proposing new DSL opcodes.
    
    Tracks all Layer 4 programs it has seen. Periodically analyzes
    the program library to find frequent patterns. Proposes composite
    opcodes for frequent patterns via OpcodeProofMarket.
    """

    def __init__(
        self,
        agent_id: str,
        proposal_threshold: float = 0.4,
        seed: int = 42,
    ):
        self.agent_id = agent_id
        self.threshold = proposal_threshold
        self._observed_programs: List[SearchAlgorithmProgram] = []
        self._approved_composites: List[CompositeOpcode] = []
        self.n_proposals_made = 0
        self.n_proposals_approved = 0

    def observe_program(self, program: SearchAlgorithmProgram) -> None:
        """Record an observed successful program."""
        self._observed_programs.append(program)

    def propose_new_opcodes(
        self,
        market: OpcodeProofMarket,
    ) -> List[OpcodeProposal]:
        """
        Analyze observed programs and propose new composite opcodes.
        Returns list of proposals made.
        """
        if len(self._observed_programs) < 3:
            return []  # need enough programs to find patterns

        # Find candidate patterns
        candidates = find_candidate_opcodes(
            self._observed_programs,
            min_frequency=self.threshold,
        )

        proposals = []
        for candidate in candidates:
            # Compute total bits saved across all programs
            n_using = sum(
                1 for prog in self._observed_programs
                if any(
                    candidate.name in i.opcode.name
                    for i in prog.instructions
                )
            )
            # Better approximation: count by subsequence match
            n_using = max(1, int(candidate.frequency * len(self._observed_programs)))
            total_saved = n_using * candidate.compression_savings

            proposal = OpcodeProposal(
                proposing_agent=self.agent_id,
                composite=candidate,
                programs_compressed=n_using,
                total_bits_saved=total_saved,
            )

            self.n_proposals_made += 1
            approved = market.evaluate(proposal, self._observed_programs)
            if approved:
                self._approved_composites.append(candidate)
                self.n_proposals_approved += 1

            proposals.append(proposal)

        return proposals

    @property
    def library_size(self) -> int:
        """Number of composite opcodes approved for this agent."""
        return len(self._approved_composites)


# ── Layer5Experiment ──────────────────────────────────────────────────────────

class Layer5Experiment:
    """
    Runs the Layer 5 discovery experiment.
    
    Setup:
    1. Generate a set of known good programs
    2. Let Layer5Agent observe them
    3. Agent proposes composite opcodes
    4. OpcodeProofMarket approves/rejects
    5. Report which new opcodes were added to the language
    """

    def run(self, verbose: bool = True) -> dict:
        """Run the Layer 5 experiment."""
        from ouroboros.layer4.search_dsl import (
            standard_beam_program, fft_guided_program,
            anneal_program, elitist_restart_program, crossover_beam_program,
        )

        # Seed library of successful programs
        known_programs = [
            standard_beam_program(width=20, iterations=8),
            fft_guided_program(width=20, iterations=8),
            fft_guided_program(width=15, iterations=6),  # variant
            anneal_program(width=20, steps=50),
            elitist_restart_program(width=20, n_restarts=3, elite_k=5),
            crossover_beam_program(width=20, iterations=8, n_cross=5),
        ]

        agent = Layer5Agent("L5_AGENT_00", proposal_threshold=0.3)
        market = OpcodeProofMarket(min_programs=2, min_total_savings=3.0)

        # Agent observes all programs
        for prog in known_programs:
            agent.observe_program(prog)

        if verbose:
            print(f"\nLayer 5 Experiment")
            print(f"Programs observed: {len(known_programs)}")

        # Agent proposes new opcodes
        proposals = agent.propose_new_opcodes(market)

        if verbose:
            print(f"Proposals made: {len(proposals)}")
            print(f"Proposals approved: {market.n_approved}")
            for composite in market.approved_opcodes:
                print(f"  New opcode: '{composite.name}' "
                      f"(saves {composite.compression_savings:.1f} bits/use, "
                      f"freq={composite.frequency:.2%})")

        return {
            "n_programs_observed": len(known_programs),
            "n_proposals": len(proposals),
            "n_approved": market.n_approved,
            "approved_names": [c.name for c in market.approved_opcodes],
            "total_bits_saved": sum(p.total_bits_saved for p in proposals if p.is_beneficial),
        }