"""Tests for Layer 4 search algorithm self-modification."""
import pytest
from ouroboros.layer4.search_dsl import (
    DSLInstruction, DSLOpcode, SearchAlgorithmProgram,
    standard_beam_program, fft_guided_program, random_restart_program,
)
from ouroboros.layer4.interpreter import AlgorithmInterpreter, InterpreterContext
from ouroboros.layer4.layer4_agent import (
    ProgramMutator, AlgorithmProposal, Layer4Agent, Layer4ProofMarket,
)


class TestDSLInstruction:
    def test_simple_instruction_bits(self):
        instr = DSLInstruction(DSLOpcode.SORT_MDL)
        assert instr.description_bits() == 2.0

    def test_parameterized_bits(self):
        instr = DSLInstruction(DSLOpcode.INIT, param=50)
        bits = instr.description_bits()
        assert bits > 4.0  # base 4 + param bits

    def test_loop_bits_includes_body(self):
        body = [DSLInstruction(DSLOpcode.MUTATE, param=3),
                DSLInstruction(DSLOpcode.SORT_MDL)]
        loop = DSLInstruction(DSLOpcode.LOOP, param=10, body_a=body)
        bits = loop.description_bits()
        assert bits > 10.0  # should include body bits

    def test_to_string_simple(self):
        assert DSLInstruction(DSLOpcode.SORT_MDL).to_string() == "SORT_MDL"
        assert DSLInstruction(DSLOpcode.FFT_SEED).to_string() == "FFT_SEED"

    def test_to_string_parameterized(self):
        instr = DSLInstruction(DSLOpcode.INIT, param=50)
        assert "50" in instr.to_string()

    def test_to_string_loop(self):
        loop = DSLInstruction(DSLOpcode.LOOP, param=5, body_a=[
            DSLInstruction(DSLOpcode.MUTATE, param=3)
        ])
        s = loop.to_string()
        assert "LOOP" in s and "5" in s


class TestSearchAlgorithmProgram:
    def test_standard_beam_program(self):
        prog = standard_beam_program(width=20, iterations=5)
        assert prog.name == "BeamSearch_DSL"
        assert len(prog.instructions) > 0
        assert prog.description_bits() > 0

    def test_fft_guided_program(self):
        prog = fft_guided_program()
        assert prog.name == "FFTGuided_DSL"
        # Should contain CLASSIFY_ENV and FFT_SEED
        opcodes = [i.opcode for i in prog.instructions]
        assert DSLOpcode.CLASSIFY_ENV in opcodes
        assert DSLOpcode.FFT_SEED in opcodes

    def test_program_length(self):
        prog = standard_beam_program()
        assert prog.program_length() > 0

    def test_to_string_is_string(self):
        prog = standard_beam_program()
        s = prog.to_string()
        assert isinstance(s, str) and len(s) > 0

    def test_description_bits_positive(self):
        prog = random_restart_program()
        assert prog.description_bits() > 0


class TestProgramMutator:
    def test_param_perturb_changes_value(self):
        mutator = ProgramMutator(seed=1)
        prog = standard_beam_program(width=25)
        mutated = mutator.mutate(prog, "param_perturb")
        # At least one parameterized instruction should differ
        orig_params = [i.param for i in prog.instructions]
        mut_params = [i.param for i in mutated.instructions]
        assert orig_params != mut_params or len(prog.instructions) != len(mutated.instructions)

    def test_insert_adds_instruction(self):
        mutator = ProgramMutator(seed=2)
        prog = standard_beam_program()
        mutated = mutator.mutate(prog, "insert")
        assert len(mutated.instructions) == len(prog.instructions) + 1

    def test_delete_removes_instruction(self):
        mutator = ProgramMutator(seed=3)
        prog = standard_beam_program()
        mutated = mutator.mutate(prog, "delete")
        assert len(mutated.instructions) == len(prog.instructions) - 1 or \
               len(mutated.instructions) == len(prog.instructions)  # may not delete if none deletable

    def test_add_fft_adds_fft_seed(self):
        mutator = ProgramMutator(seed=4)
        prog = standard_beam_program()
        # Remove any existing FFT_SEED first
        prog.instructions = [i for i in prog.instructions if i.opcode != DSLOpcode.FFT_SEED]
        mutated = mutator.mutate(prog, "add_fft")
        opcodes = [i.opcode for i in mutated.instructions]
        assert DSLOpcode.FFT_SEED in opcodes

    def test_crossover_combines_programs(self):
        mutator = ProgramMutator(seed=5)
        prog_a = standard_beam_program(width=20)
        prog_b = fft_guided_program(width=15)
        crossed = mutator.crossover(prog_a, prog_b)
        assert len(crossed.instructions) > 0
        assert "crossover" in crossed.name

    def test_mutation_does_not_crash_randomly(self):
        mutator = ProgramMutator(seed=42)
        prog = standard_beam_program()
        for _ in range(20):
            mutated = mutator.mutate(prog)
            assert isinstance(mutated, SearchAlgorithmProgram)


class TestAlgorithmInterpreter:
    def test_standard_beam_runs(self):
        prog = standard_beam_program(width=5, iterations=2)
        interpreter = AlgorithmInterpreter(time_budget_seconds=3.0)
        obs = [(3*t+1)%7 for t in range(80)]
        expr, cost, elapsed = interpreter.run(prog, obs, alphabet_size=7)
        assert elapsed > 0
        assert cost >= 0

    def test_fft_guided_runs(self):
        prog = fft_guided_program(width=5, iterations=2)
        interpreter = AlgorithmInterpreter(time_budget_seconds=3.0)
        obs = [(3*t+1)%7 for t in range(80)]
        expr, cost, elapsed = interpreter.run(prog, obs, alphabet_size=7)
        assert elapsed > 0

    def test_time_budget_respected(self):
        import time
        prog = standard_beam_program(width=10, iterations=50)  # very long
        interpreter = AlgorithmInterpreter(time_budget_seconds=1.0)
        obs = [(3*t+1)%7 for t in range(100)]
        start = time.time()
        interpreter.run(prog, obs, alphabet_size=7)
        elapsed = time.time() - start
        assert elapsed < 3.0  # should stop within budget + overhead


class TestLayer4Agent:
    def test_initial_program_is_beam(self):
        agent = Layer4Agent("TEST")
        assert "Beam" in agent.current_program.name

    def test_proposals_tracked(self):
        agent = Layer4Agent("TEST")
        assert agent.proposals_made == 0
        assert agent.proposals_approved == 0

    def test_run_round_returns_dict(self):
        from ouroboros.environments.modular import ModularArithmeticEnv
        agent = Layer4Agent("TEST", proposal_interval=1, n_candidate_mutations=2,
                            time_budget_per_eval=1.5)
        env = ModularArithmeticEnv(modulus=7)
        market = Layer4ProofMarket(time_budget_per_eval=1.0)
        result = agent.run_round(env, market, round_num=1)
        assert "current_cost" in result
        assert "proposal_made" in result

    def test_program_history_tracked(self):
        agent = Layer4Agent("TEST")
        assert len(agent.program_history) == 1
        assert "Beam" in agent.program_history[0]


class TestAlgorithmProposal:
    def test_cost_improvement_computed(self):
        prog_a = standard_beam_program()
        prog_b = fft_guided_program()
        proposal = AlgorithmProposal(
            proposing_agent="A",
            current_program=prog_a,
            proposed_program=prog_b,
            training_env_name="test",
            current_best_cost=100.0,
            proposed_best_cost=80.0,
            current_time_seconds=1.0,
            proposed_time_seconds=1.5,
        )
        assert proposal.cost_improvement == pytest.approx(20.0)
        assert proposal.is_improvement

    def test_complexity_delta_computed(self):
        prog_a = standard_beam_program()
        prog_b = fft_guided_program()  # more complex
        proposal = AlgorithmProposal(
            proposing_agent="A",
            current_program=prog_a,
            proposed_program=prog_b,
            training_env_name="test",
            current_best_cost=100.0, proposed_best_cost=80.0,
            current_time_seconds=1.0, proposed_time_seconds=1.5,
        )
        delta = proposal.program_complexity_delta
        assert isinstance(delta, float)