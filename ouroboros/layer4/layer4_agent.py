"""
Layer4Agent — Proposes and evaluates novel search algorithm programs.

The four layers of OUROBOROS self-improvement:
  Layer 0: modify expressions (what the agent finds)
  Layer 1: modify hyperparameters (how hard it searches)
  Layer 2: modify MDL objective (what it optimizes)
  Layer 3: select search strategy (which algorithm it uses)
  Layer 4: write a new search algorithm in DSL (invent new strategies)

Layer 4 agent behavior:
  1. Start with the standard_beam_program as current algorithm
  2. Every K rounds: generate mutations of the current DSL program
  3. Run each candidate program on training data
  4. If a candidate beats the current program: propose it
  5. Layer4ProofMarket evaluates on OOD data
  6. If approved: the agent permanently uses the new algorithm
"""

from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from ouroboros.layer4.search_dsl import (
    SearchAlgorithmProgram, DSLInstruction, DSLOpcode,
    standard_beam_program, fft_guided_program,
)
from ouroboros.layer4.interpreter import AlgorithmInterpreter
from ouroboros.environments.base import Environment


@dataclass
class AlgorithmProposal:
    """An agent's proposal of a new search algorithm."""
    proposing_agent: str
    current_program: SearchAlgorithmProgram
    proposed_program: SearchAlgorithmProgram
    training_env_name: str
    current_best_cost: float
    proposed_best_cost: float
    current_time_seconds: float
    proposed_time_seconds: float

    @property
    def cost_improvement(self) -> float:
        return self.current_best_cost - self.proposed_best_cost

    @property
    def is_improvement(self) -> bool:
        return self.cost_improvement > 0.0

    @property
    def program_complexity_delta(self) -> float:
        """Change in program description bits (positive = more complex)."""
        return (self.proposed_program.description_bits() -
                self.current_program.description_bits())

    def description(self) -> str:
        return (
            f"AlgorithmProposal by {self.proposing_agent}\n"
            f"  Current:  {self.current_program.name} "
            f"({self.current_program.program_length()} instr, "
            f"{self.current_program.description_bits():.1f} bits)\n"
            f"  Proposed: {self.proposed_program.name} "
            f"({self.proposed_program.program_length()} instr, "
            f"{self.proposed_program.description_bits():.1f} bits)\n"
            f"  Cost improvement: {self.cost_improvement:.2f} bits\n"
            f"  Program: {self.proposed_program.to_string()[:80]}"
        )


@dataclass
class AlgorithmEvaluationResult:
    """Result of Layer4ProofMarket evaluating an algorithm proposal."""
    proposal: AlgorithmProposal
    approved: bool
    validation_env_name: str
    validation_current_cost: float
    validation_proposed_cost: float
    rejection_reason: Optional[str] = None

    @property
    def validation_improvement(self) -> float:
        return self.validation_current_cost - self.validation_proposed_cost

    def description(self) -> str:
        status = "✅ APPROVED" if self.approved else "❌ REJECTED"
        return (
            f"{status}: {self.proposal.proposed_program.name}\n"
            f"  Training improvement: {self.proposal.cost_improvement:.2f} bits\n"
            f"  Validation improvement: {self.validation_improvement:.2f} bits\n"
            f"  Verdict: {self.rejection_reason or 'Genuine improvement'}"
        )


class ProgramMutator:
    """
    Mutates DSL programs to generate novel algorithm proposals.
    
    Mutations:
      1. Parameter perturbation: change INIT(50) to INIT(35)
      2. Instruction insertion: add FFT_SEED before INIT
      3. Instruction deletion: remove MCMC step
      4. Body extension: add MUTATE(2) inside LOOP
      5. New instruction: add CLASSIFY_ENV at start
      6. Crossover: mix two existing programs
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def mutate(
        self,
        program: SearchAlgorithmProgram,
        mutation_type: str = "random",
    ) -> SearchAlgorithmProgram:
        """Generate a mutated version of the program."""
        new_prog = copy.deepcopy(program)
        new_prog.name = f"{program.name}_mut"

        if mutation_type == "random":
            mutation_type = self._rng.choice([
                "param_perturb", "insert", "delete", "extend_body",
                "add_classify", "add_fft", "add_mcmc",
            ])

        if mutation_type == "param_perturb" and new_prog.instructions:
            # Find a parameterized instruction and change its value
            parameterized = [
                (i, instr) for i, instr in enumerate(new_prog.instructions)
                if instr.param > 0
            ]
            if parameterized:
                idx, instr = self._rng.choice(parameterized)
                delta = self._rng.choice([-5, -3, -2, 2, 3, 5])
                new_prog.instructions[idx].param = max(1, instr.param + delta)

        elif mutation_type == "insert":
            # Insert a simple instruction at a random position
            new_instr = self._rng.choice([
                DSLInstruction(DSLOpcode.FFT_SEED),
                DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
                DSLInstruction(DSLOpcode.SAVE_BEST),
                DSLInstruction(DSLOpcode.LOAD_BEST),
                DSLInstruction(DSLOpcode.SORT_MDL),
                DSLInstruction(DSLOpcode.MUTATE, param=self._rng.randint(1, 5)),
            ])
            pos = self._rng.randint(0, len(new_prog.instructions))
            new_prog.instructions.insert(pos, new_instr)

        elif mutation_type == "delete" and len(new_prog.instructions) > 3:
            # Delete a random non-critical instruction
            deletable = [
                i for i, instr in enumerate(new_prog.instructions)
                if instr.opcode not in (DSLOpcode.SORT_MDL, DSLOpcode.TAKE)
            ]
            if deletable:
                idx = self._rng.choice(deletable)
                new_prog.instructions.pop(idx)

        elif mutation_type == "extend_body":
            # Add an instruction inside a LOOP body
            loops = [
                instr for instr in new_prog.instructions
                if instr.opcode == DSLOpcode.LOOP
            ]
            if loops:
                loop = self._rng.choice(loops)
                new_instr = DSLInstruction(
                    DSLOpcode.MUTATE, param=self._rng.randint(1, 3)
                )
                pos = self._rng.randint(0, len(loop.body_a))
                loop.body_a.insert(pos, new_instr)

        elif mutation_type == "add_classify":
            # Add CLASSIFY_ENV + IF_PERIODIC at the start
            if DSLOpcode.CLASSIFY_ENV not in [i.opcode for i in new_prog.instructions]:
                new_prog.instructions.insert(0, DSLInstruction(DSLOpcode.CLASSIFY_ENV))
                new_prog.instructions.insert(1, DSLInstruction(DSLOpcode.FFT_SEED))

        elif mutation_type == "add_fft":
            if DSLOpcode.FFT_SEED not in [i.opcode for i in new_prog.instructions]:
                new_prog.instructions.insert(0, DSLInstruction(DSLOpcode.FFT_SEED))

        elif mutation_type == "add_mcmc" and len(new_prog.instructions) > 2:
            # Add MCMC refinement at end
            if DSLOpcode.MCMC not in [i.opcode for i in new_prog.instructions]:
                new_prog.instructions.append(DSLInstruction(DSLOpcode.MCMC, param=50))

        return new_prog

    def crossover(
        self,
        prog_a: SearchAlgorithmProgram,
        prog_b: SearchAlgorithmProgram,
    ) -> SearchAlgorithmProgram:
        """Create a new program by combining parts of two existing ones."""
        n_a = len(prog_a.instructions)
        n_b = len(prog_b.instructions)
        if n_a == 0:
            return copy.deepcopy(prog_b)
        if n_b == 0:
            return copy.deepcopy(prog_a)

        split_a = self._rng.randint(1, n_a)
        split_b = self._rng.randint(0, n_b)
        new_instructions = (
            list(prog_a.instructions[:split_a]) +
            list(prog_b.instructions[split_b:])
        )
        return SearchAlgorithmProgram(
            name=f"crossover_{prog_a.name[:5]}_{prog_b.name[:5]}",
            instructions=new_instructions,
        )


class Layer4ProofMarket:
    """
    Adversarially evaluates DSL algorithm proposals.
    
    Approval criteria:
    1. Proposed algorithm finds lower MDL cost on training environment
    2. Improvement > min_improvement_bits
    3. Proposed algorithm does NOT perform worse on OOD validation environments
    4. Program is not excessively complex (description_bits delta < max_delta)
    """

    def __init__(
        self,
        validation_environments: List[Environment] = None,
        min_improvement_bits: float = 5.0,
        max_complexity_delta_bits: float = 20.0,
        time_budget_per_eval: float = 5.0,
        random_seed: int = 42,
    ):
        self.validation_envs = validation_environments or []
        self.min_improvement = min_improvement_bits
        self.max_complexity_delta = max_complexity_delta_bits
        self.time_budget = time_budget_per_eval
        self._rng = random.Random(random_seed)
        self._interpreter = AlgorithmInterpreter(time_budget_seconds=time_budget_per_eval)
        self._approved_programs: List[SearchAlgorithmProgram] = []

    def evaluate(
        self,
        proposal: AlgorithmProposal,
        training_env: Environment,
    ) -> AlgorithmEvaluationResult:
        """Evaluate a proposed algorithm."""

        # Check minimum improvement
        if proposal.cost_improvement < self.min_improvement:
            return AlgorithmEvaluationResult(
                proposal=proposal, approved=False,
                validation_env_name="threshold_check",
                validation_current_cost=proposal.current_best_cost,
                validation_proposed_cost=proposal.proposed_best_cost,
                rejection_reason=(
                    f"Improvement {proposal.cost_improvement:.2f} < "
                    f"threshold {self.min_improvement:.2f}"
                ),
            )

        # Check complexity: don't approve programs much more complex than current
        if proposal.program_complexity_delta > self.max_complexity_delta:
            return AlgorithmEvaluationResult(
                proposal=proposal, approved=False,
                validation_env_name="complexity_check",
                validation_current_cost=proposal.current_best_cost,
                validation_proposed_cost=proposal.proposed_best_cost,
                rejection_reason=(
                    f"Program too complex: +{proposal.program_complexity_delta:.1f} bits"
                ),
            )

        # OOD validation
        if self.validation_envs:
            val_env = self._rng.choice(self.validation_envs)
            val_obs = val_env.generate(200)

            _, current_cost, _ = self._interpreter.run(
                proposal.current_program, val_obs, val_env.alphabet_size
            )
            _, proposed_cost, _ = self._interpreter.run(
                proposal.proposed_program, val_obs, val_env.alphabet_size
            )

            if proposed_cost > current_cost + 10.0:
                return AlgorithmEvaluationResult(
                    proposal=proposal, approved=False,
                    validation_env_name=val_env.name,
                    validation_current_cost=current_cost,
                    validation_proposed_cost=proposed_cost,
                    rejection_reason=(
                        f"Performs worse OOD on {val_env.name}: "
                        f"+{proposed_cost - current_cost:.2f} bits"
                    ),
                )

            val_env_name = val_env.name
            val_current = current_cost
            val_proposed = proposed_cost
        else:
            val_env_name = "no_validation"
            val_current = proposal.current_best_cost
            val_proposed = proposal.proposed_best_cost

        result = AlgorithmEvaluationResult(
            proposal=proposal, approved=True,
            validation_env_name=val_env_name,
            validation_current_cost=val_current,
            validation_proposed_cost=val_proposed,
        )
        self._approved_programs.append(copy.deepcopy(proposal.proposed_program))
        return result

    @property
    def approved_programs(self) -> List[SearchAlgorithmProgram]:
        return list(self._approved_programs)


class Layer4Agent:
    """
    An agent capable of proposing entirely new search algorithms.
    
    Maintains a current DSL program. Every K rounds:
      1. Generate N mutations of the current program
      2. Run each mutation on training data
      3. If any mutation outperforms current: propose it
      4. Layer4ProofMarket evaluates OOD
      5. If approved: permanently switch to the new algorithm
    """

    def __init__(
        self,
        agent_id: str,
        initial_program: SearchAlgorithmProgram = None,
        proposal_interval: int = 8,
        n_candidate_mutations: int = 5,
        time_budget_per_eval: float = 3.0,
        random_seed: int = 42,
    ):
        self.agent_id = agent_id
        self.current_program = initial_program or standard_beam_program()
        self.proposal_interval = proposal_interval
        self.n_candidates = n_candidate_mutations
        self.time_budget = time_budget_per_eval
        self._mutator = ProgramMutator(seed=random_seed)
        self._interpreter = AlgorithmInterpreter(time_budget_seconds=time_budget_per_eval)
        self._rng = random.Random(random_seed)

        # Stats
        self.proposals_made = 0
        self.proposals_approved = 0
        self.program_history: List[str] = [self.current_program.name]
        self.cost_history: List[float] = []

    def run_round(
        self,
        env: Environment,
        market: Layer4ProofMarket,
        round_num: int,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run one round of Layer 4 self-improvement."""
        obs = env.generate(200)

        # Run current program
        expr, current_cost, current_time = self._interpreter.run(
            self.current_program, obs, env.alphabet_size
        )
        self.cost_history.append(current_cost)

        result = {
            "round": round_num,
            "agent_id": self.agent_id,
            "current_program": self.current_program.name,
            "current_cost": current_cost,
            "proposal_made": False,
            "proposal_approved": False,
        }

        # Every K rounds: try to improve the algorithm
        if round_num % self.proposal_interval == 0:
            self.proposals_made += 1

            # Generate candidate mutations
            candidates = []
            for _ in range(self.n_candidates):
                mutated = self._mutator.mutate(self.current_program)
                _, mut_cost, mut_time = self._interpreter.run(
                    mutated, obs, env.alphabet_size
                )
                candidates.append((mutated, mut_cost, mut_time))

            # Best candidate
            candidates.sort(key=lambda x: x[1])
            best_prog, best_cost, best_time = candidates[0]

            if best_cost < current_cost:
                result["proposal_made"] = True
                proposal = AlgorithmProposal(
                    proposing_agent=self.agent_id,
                    current_program=self.current_program,
                    proposed_program=best_prog,
                    training_env_name=env.name,
                    current_best_cost=current_cost,
                    proposed_best_cost=best_cost,
                    current_time_seconds=current_time,
                    proposed_time_seconds=best_time,
                )

                if verbose:
                    print(f"\n  [{self.agent_id}] ALGORITHM PROPOSAL:")
                    print(f"    {proposal.description()}")

                evaluation = market.evaluate(proposal, env)
                if evaluation.approved:
                    self.current_program = copy.deepcopy(best_prog)
                    self.program_history.append(best_prog.name)
                    self.proposals_approved += 1
                    result["proposal_approved"] = True
                    if verbose:
                        print(f"  [{self.agent_id}] ✅ New algorithm approved!")
                elif verbose:
                    print(f"  [{self.agent_id}] ❌ Rejected: {evaluation.rejection_reason}")

        return result