"""
SearchAlgorithmDSL — A small language for describing search algorithms.

Agents write programs in this DSL to propose new search strategies.
The DSL is intentionally minimal — 14 instructions — so that:
  1. Programs are short (< 20 instructions on average)
  2. Mutations rarely produce nonsense
  3. The interpreter is fast (< 5s per run for typical programs)

Instruction set:
  INIT(n)              — initialize beam with n random grammar-valid expressions
  BEAM(width)          — set beam width
  MUTATE(n)            — apply n mutations to each beam member
  FFT_SEED             — detect period in observations, add sin/mod seeds
  MCMC(iters)          — apply MCMC refinement to top-k beam members
  GRAMMAR_FILTER       — remove grammar-invalid expressions from beam
  SORT_MDL             — sort beam by MDL cost (ascending)
  TAKE(k)              — keep only top-k beam members
  LOOP(n, body)        — repeat body n times
  CLASSIFY_ENV         — run EnvironmentClassifier, adjust category weights
  IF_PERIODIC(a, b)    — if periodic: execute a, else execute b
  SAVE_BEST            — save current best to memory
  LOAD_BEST            — load from memory, add to beam
  PARALLEL(k, body)    — run body k times with different seeds, merge beams

Description bits per instruction:
  Simple (SORT_MDL, GRAMMAR_FILTER, SAVE_BEST, LOAD_BEST): 2 bits
  Parameterized (INIT, BEAM, MUTATE, MCMC, TAKE, LOOP): 4 bits + param bits
  Compound (IF_PERIODIC, PARALLEL): 6 bits + body bits
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Union


class DSLOpcode(Enum):
    INIT          = auto()
    BEAM          = auto()
    MUTATE        = auto()
    FFT_SEED      = auto()
    MCMC          = auto()
    GRAMMAR_FILTER = auto()
    SORT_MDL      = auto()
    TAKE          = auto()
    LOOP          = auto()
    CLASSIFY_ENV  = auto()
    IF_PERIODIC   = auto()
    SAVE_BEST     = auto()
    LOAD_BEST     = auto()
    PARALLEL      = auto()


# Description bits per opcode (for MDL cost of the program itself)
OPCODE_BITS: Dict[DSLOpcode, float] = {
    DSLOpcode.INIT:           4.0,
    DSLOpcode.BEAM:           3.0,
    DSLOpcode.MUTATE:         3.0,
    DSLOpcode.FFT_SEED:       2.0,
    DSLOpcode.MCMC:           4.0,
    DSLOpcode.GRAMMAR_FILTER: 2.0,
    DSLOpcode.SORT_MDL:       2.0,
    DSLOpcode.TAKE:           3.0,
    DSLOpcode.LOOP:           5.0,
    DSLOpcode.CLASSIFY_ENV:   3.0,
    DSLOpcode.IF_PERIODIC:    6.0,
    DSLOpcode.SAVE_BEST:      2.0,
    DSLOpcode.LOAD_BEST:      2.0,
    DSLOpcode.PARALLEL:       6.0,
}


@dataclass
class DSLInstruction:
    """One instruction in the Search Algorithm DSL."""
    opcode: DSLOpcode
    param: int = 0              # for INIT/BEAM/MUTATE/MCMC/TAKE/LOOP/PARALLEL
    body_a: List['DSLInstruction'] = field(default_factory=list)  # for LOOP, IF_PERIODIC (true branch), PARALLEL
    body_b: List['DSLInstruction'] = field(default_factory=list)  # for IF_PERIODIC (false branch)

    def description_bits(self) -> float:
        """MDL cost to describe this instruction."""
        bits = OPCODE_BITS.get(self.opcode, 4.0)
        if self.param > 0:
            bits += max(1.0, float(int(self.param).bit_length()))
        for instr in self.body_a:
            bits += instr.description_bits()
        for instr in self.body_b:
            bits += instr.description_bits()
        return bits

    def to_string(self) -> str:
        """Human-readable representation."""
        op = self.opcode.name
        if self.opcode in (DSLOpcode.INIT, DSLOpcode.BEAM, DSLOpcode.MUTATE,
                           DSLOpcode.MCMC, DSLOpcode.TAKE):
            return f"{op}({self.param})"
        if self.opcode == DSLOpcode.LOOP:
            body_str = " ".join(i.to_string() for i in self.body_a)
            return f"LOOP({self.param}, [{body_str}])"
        if self.opcode == DSLOpcode.IF_PERIODIC:
            a_str = " ".join(i.to_string() for i in self.body_a)
            b_str = " ".join(i.to_string() for i in self.body_b)
            return f"IF_PERIODIC([{a_str}], [{b_str}])"
        if self.opcode == DSLOpcode.PARALLEL:
            body_str = " ".join(i.to_string() for i in self.body_a)
            return f"PARALLEL({self.param}, [{body_str}])"
        return op


@dataclass
class SearchAlgorithmProgram:
    """A complete search algorithm described in the DSL."""
    instructions: List[DSLInstruction]
    name: str = "unnamed"
    author_agent: str = "unknown"

    def description_bits(self) -> float:
        """Total MDL cost of describing this program."""
        return sum(i.description_bits() for i in self.instructions)

    def to_string(self) -> str:
        return " | ".join(i.to_string() for i in self.instructions)

    def program_length(self) -> int:
        """Total number of instructions (including nested)."""
        total = len(self.instructions)
        for instr in self.instructions:
            total += len(instr.body_a) + len(instr.body_b)
        return total


# ── Known programs (the original 5 strategies, encoded in DSL) ─────────────

def standard_beam_program(width: int = 25, iterations: int = 10) -> SearchAlgorithmProgram:
    """BeamSearch encoded in the DSL."""
    return SearchAlgorithmProgram(
        name="BeamSearch_DSL",
        instructions=[
            DSLInstruction(DSLOpcode.INIT, param=width * 3),
            DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.TAKE, param=width),
            DSLInstruction(DSLOpcode.LOOP, param=iterations, body_a=[
                DSLInstruction(DSLOpcode.MUTATE, param=3),
                DSLInstruction(DSLOpcode.SORT_MDL),
                DSLInstruction(DSLOpcode.TAKE, param=width),
            ]),
        ]
    )


def fft_guided_program(width: int = 20, iterations: int = 8) -> SearchAlgorithmProgram:
    """The novel FFT-guided strategy agents should discover."""
    return SearchAlgorithmProgram(
        name="FFTGuided_DSL",
        instructions=[
            DSLInstruction(DSLOpcode.CLASSIFY_ENV),
            DSLInstruction(DSLOpcode.FFT_SEED),
            DSLInstruction(DSLOpcode.INIT, param=width * 2),
            DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.TAKE, param=width),
            DSLInstruction(DSLOpcode.LOOP, param=iterations, body_a=[
                DSLInstruction(DSLOpcode.IF_PERIODIC,
                    body_a=[
                        DSLInstruction(DSLOpcode.FFT_SEED),
                        DSLInstruction(DSLOpcode.MUTATE, param=2),
                    ],
                    body_b=[
                        DSLInstruction(DSLOpcode.MUTATE, param=3),
                    ]
                ),
                DSLInstruction(DSLOpcode.SORT_MDL),
                DSLInstruction(DSLOpcode.TAKE, param=width),
            ]),
            DSLInstruction(DSLOpcode.SAVE_BEST),
        ]
    )


def random_restart_program(n_restarts: int = 5, per_restart: int = 8) -> SearchAlgorithmProgram:
    """RandomRestart encoded in DSL."""
    return SearchAlgorithmProgram(
        name="RandomRestart_DSL",
        instructions=[
            DSLInstruction(DSLOpcode.PARALLEL, param=n_restarts, body_a=[
                DSLInstruction(DSLOpcode.INIT, param=per_restart),
                DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
                DSLInstruction(DSLOpcode.SORT_MDL),
                DSLInstruction(DSLOpcode.TAKE, param=per_restart // 2),
            ]),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.TAKE, param=20),
        ]
    )


# ── Three new opcodes added Day 42 ────────────────────────────────────────────

import enum as _enum

_old_members = {name: member.value for name, member in DSLOpcode.__members__.items()}
_new_max = max(_old_members.values())

DSLOpcode = _enum.Enum('DSLOpcode', {
    **_old_members,
    'ANNEAL':     _new_max + 1,
    'ELITE_KEEP': _new_max + 2,
    'CROSS':      _new_max + 3,
})

# Rebuild OPCODE_BITS with new DSLOpcode (old keys are now stale objects)
OPCODE_BITS = {
    DSLOpcode.INIT:           4.0,
    DSLOpcode.BEAM:           3.0,
    DSLOpcode.MUTATE:         3.0,
    DSLOpcode.FFT_SEED:       2.0,
    DSLOpcode.MCMC:           4.0,
    DSLOpcode.GRAMMAR_FILTER: 2.0,
    DSLOpcode.SORT_MDL:       2.0,
    DSLOpcode.TAKE:           3.0,
    DSLOpcode.LOOP:           5.0,
    DSLOpcode.CLASSIFY_ENV:   3.0,
    DSLOpcode.IF_PERIODIC:    6.0,
    DSLOpcode.SAVE_BEST:      2.0,
    DSLOpcode.LOAD_BEST:      2.0,
    DSLOpcode.PARALLEL:       6.0,
    DSLOpcode.ANNEAL:         7.0,
    DSLOpcode.ELITE_KEEP:     3.0,
    DSLOpcode.CROSS:          4.0,
}


def anneal_program(
    width: int = 20,
    t_start: float = 5.0,
    t_end: float = 0.1,
    steps: int = 100,
) -> 'SearchAlgorithmProgram':
    """
    ANNEAL-based search program.
    Phase 1: beam search to find starting point
    Phase 2: simulated annealing from beam's best
    """
    return SearchAlgorithmProgram(
        name="Annealing_DSL",
        instructions=[
            DSLInstruction(DSLOpcode.INIT, param=width),
            DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.TAKE, param=width // 2),
            DSLInstruction(DSLOpcode.SAVE_BEST),
            DSLInstruction(DSLOpcode.ANNEAL, param=steps),
            DSLInstruction(DSLOpcode.LOAD_BEST),
            DSLInstruction(DSLOpcode.SORT_MDL),
        ]
    )


def elitist_restart_program(
    width: int = 20,
    n_restarts: int = 3,
    elite_k: int = 5,
) -> 'SearchAlgorithmProgram':
    """
    Elitist restart: preserve top-k across independent restarts.
    Better than random restart because elites carry over.
    """
    return SearchAlgorithmProgram(
        name="ElitistRestart_DSL",
        instructions=[
            DSLInstruction(DSLOpcode.INIT, param=width),
            DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.ELITE_KEEP, param=elite_k),
            DSLInstruction(DSLOpcode.LOOP, param=n_restarts, body_a=[
                DSLInstruction(DSLOpcode.INIT, param=width),
                DSLInstruction(DSLOpcode.LOAD_BEST),
                DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
                DSLInstruction(DSLOpcode.SORT_MDL),
                DSLInstruction(DSLOpcode.TAKE, param=width),
                DSLInstruction(DSLOpcode.ELITE_KEEP, param=elite_k),
            ]),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.TAKE, param=width),
        ]
    )


def crossover_beam_program(
    width: int = 20,
    iterations: int = 8,
    n_cross: int = 5,
) -> 'SearchAlgorithmProgram':
    """
    Beam search enhanced with crossover.
    Each iteration: mutate some, cross others, merge.
    This is a basic genetic algorithm over expression trees.
    """
    return SearchAlgorithmProgram(
        name="CrossoverBeam_DSL",
        instructions=[
            DSLInstruction(DSLOpcode.INIT, param=width),
            DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
            DSLInstruction(DSLOpcode.SORT_MDL),
            DSLInstruction(DSLOpcode.TAKE, param=width),
            DSLInstruction(DSLOpcode.LOOP, param=iterations, body_a=[
                DSLInstruction(DSLOpcode.MUTATE, param=2),
                DSLInstruction(DSLOpcode.CROSS, param=n_cross),
                DSLInstruction(DSLOpcode.SORT_MDL),
                DSLInstruction(DSLOpcode.TAKE, param=width),
            ]),
        ]
    )