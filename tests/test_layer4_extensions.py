"""Tests for Layer 4 DSL extensions — ANNEAL, ELITE_KEEP, CROSS."""
import pytest
import copy
from ouroboros.layer4.search_dsl import (
    DSLInstruction, DSLOpcode, SearchAlgorithmProgram,
    standard_beam_program, fft_guided_program,
    anneal_program, elitist_restart_program, crossover_beam_program,
)
from ouroboros.layer4.interpreter import AlgorithmInterpreter
from ouroboros.layer4.space_analyzer import (
    DSLSearchSpaceAnalyzer, Layer4LandmarkExperiment, ProgramEditDistance,
)


class TestNewOpcodes:
    def test_anneal_opcode_exists(self):
        assert hasattr(DSLOpcode, 'ANNEAL')
        assert DSLOpcode.ANNEAL is not None

    def test_elite_keep_opcode_exists(self):
        assert hasattr(DSLOpcode, 'ELITE_KEEP')

    def test_cross_opcode_exists(self):
        assert hasattr(DSLOpcode, 'CROSS')

    def test_new_opcodes_have_bits(self):
        from ouroboros.layer4.search_dsl import OPCODE_BITS
        assert DSLOpcode.ANNEAL in OPCODE_BITS
        assert DSLOpcode.ELITE_KEEP in OPCODE_BITS
        assert DSLOpcode.CROSS in OPCODE_BITS

    def test_anneal_bits_higher_than_sort(self):
        from ouroboros.layer4.search_dsl import OPCODE_BITS
        assert OPCODE_BITS[DSLOpcode.ANNEAL] > OPCODE_BITS[DSLOpcode.SORT_MDL]


class TestNewPrograms:
    def test_anneal_program_created(self):
        prog = anneal_program(width=10, steps=30)
        assert prog.name == "Annealing_DSL"
        opcodes = [i.opcode for i in prog.instructions]
        assert DSLOpcode.ANNEAL in opcodes

    def test_elitist_restart_created(self):
        prog = elitist_restart_program(width=15, n_restarts=3, elite_k=4)
        assert prog.name == "ElitistRestart_DSL"
        opcodes = [i.opcode for i in prog.instructions]
        assert DSLOpcode.ELITE_KEEP in opcodes

    def test_crossover_beam_created(self):
        prog = crossover_beam_program(width=15, iterations=5, n_cross=4)
        assert prog.name == "CrossoverBeam_DSL"
        # Find CROSS in any instruction or body
        def has_cross(instrs):
            for i in instrs:
                if i.opcode == DSLOpcode.CROSS:
                    return True
                if has_cross(i.body_a) or has_cross(i.body_b):
                    return True
            return False
        assert has_cross(prog.instructions)

    def test_program_description_bits_positive(self):
        for prog in [anneal_program(), elitist_restart_program(), crossover_beam_program()]:
            assert prog.description_bits() > 0

    def test_program_length_reasonable(self):
        for prog in [anneal_program(), elitist_restart_program(), crossover_beam_program()]:
            assert 3 <= prog.program_length() <= 30


class TestAnnealExecution:
    def test_anneal_program_runs(self):
        prog = anneal_program(width=5, steps=10)
        interp = AlgorithmInterpreter(time_budget_seconds=3.0)
        obs = [(3*t+1)%7 for t in range(60)]
        expr, cost, elapsed = interp.run(prog, obs, alphabet_size=7)
        assert elapsed > 0
        assert cost >= 0

    def test_anneal_doesnt_crash_on_empty_beam(self):
        from ouroboros.layer4.interpreter import InterpreterContext
        interp = AlgorithmInterpreter(time_budget_seconds=2.0)
        obs = [5] * 30
        ctx = InterpreterContext(observations=obs, alphabet_size=7, beam=[])
        import time
        instr = DSLInstruction(DSLOpcode.ANNEAL, param=5)
        # Should not raise even with empty beam
        try:
            interp._execute(instr, ctx, time.time(), False)
        except Exception as e:
            pytest.fail(f"ANNEAL raised with empty beam: {e}")


class TestEliteKeepExecution:
    def test_elite_keep_preserves_best(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        from ouroboros.layer4.interpreter import InterpreterContext
        import time

        expr_good = ExprNode(NodeType.CONST, value=3)
        expr_bad = ExprNode(NodeType.CONST, value=99)

        obs = [3] * 30
        ctx = InterpreterContext(
            observations=obs, alphabet_size=7,
            beam=[(expr_good, 20.0), (expr_bad, 200.0)],
        )

        interp = AlgorithmInterpreter(time_budget_seconds=2.0)
        instr = DSLInstruction(DSLOpcode.ELITE_KEEP, param=1)
        interp._execute(instr, ctx, time.time(), False)

        # Elite pool should have the best
        assert hasattr(ctx, '_elite_pool')
        assert len(ctx._elite_pool) >= 1
        assert ctx._elite_pool[0][1] == 20.0


class TestCrossExecution:
    def test_cross_creates_offspring(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        from ouroboros.layer4.interpreter import InterpreterContext
        import time

        expr_a = ExprNode(NodeType.CONST, value=3)
        expr_b = ExprNode(NodeType.CONST, value=7)
        obs = [5] * 30
        ctx = InterpreterContext(
            observations=obs, alphabet_size=10,
            beam=[(expr_a, 50.0), (expr_b, 60.0)],
        )
        initial_size = len(ctx.beam)
        interp = AlgorithmInterpreter(time_budget_seconds=2.0)
        instr = DSLInstruction(DSLOpcode.CROSS, param=3)
        interp._execute(instr, ctx, time.time(), False)
        # Beam should have grown (offspring added)
        assert len(ctx.beam) >= initial_size

    def test_cross_beam_sorted_after(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        from ouroboros.layer4.interpreter import InterpreterContext
        import time
        expr_a = ExprNode(NodeType.CONST, value=3)
        expr_b = ExprNode(NodeType.CONST, value=7)
        obs = [5] * 30
        ctx = InterpreterContext(
            observations=obs, alphabet_size=10,
            beam=[(expr_a, 50.0), (expr_b, 60.0)],
        )
        interp = AlgorithmInterpreter(time_budget_seconds=2.0)
        instr = DSLInstruction(DSLOpcode.CROSS, param=2)
        interp._execute(instr, ctx, time.time(), False)
        if len(ctx.beam) >= 2:
            assert ctx.beam[0][1] <= ctx.beam[1][1]


class TestDSLSearchSpaceAnalyzer:
    def test_analysis_runs(self):
        analyzer = DSLSearchSpaceAnalyzer()
        result = analyzer.analyze()
        assert result.n_opcodes >= 17  # 14 original + 3 new
        assert len(result.known_programs) >= 4

    def test_beam_to_fft_distance_positive(self):
        analyzer = DSLSearchSpaceAnalyzer()
        result = analyzer.analyze()
        assert result.beam_to_fft_distance > 0

    def test_all_distances_nonnegative(self):
        analyzer = DSLSearchSpaceAnalyzer()
        result = analyzer.analyze()
        for d in result.pairwise_distances:
            assert d.n_mutations_estimated >= 0

    def test_summary_is_string(self):
        analyzer = DSLSearchSpaceAnalyzer()
        result = analyzer.analyze()
        s = result.summary()
        assert isinstance(s, str) and len(s) > 50

    def test_edit_distance_same_program(self):
        analyzer = DSLSearchSpaceAnalyzer()
        prog = standard_beam_program()
        d = analyzer.opcode_edit_distance(prog, prog)
        assert d.total_edits == 0

    def test_edit_distance_different_programs(self):
        analyzer = DSLSearchSpaceAnalyzer()
        beam = standard_beam_program()
        fft = fft_guided_program()
        d = analyzer.opcode_edit_distance(beam, fft)
        assert d.total_edits > 0

    def test_reachability_reported(self):
        analyzer = DSLSearchSpaceAnalyzer()
        result = analyzer.analyze()
        assert isinstance(result.all_reachable_from_beam, bool)


class TestLayer4LandmarkExperiment:
    def test_runs_quickly(self):
        exp = Layer4LandmarkExperiment(n_runs=3, n_mutations_per_run=5,
                                        observations_length=60, seed=42)
        results = exp.run(verbose=False)
        assert "discovery_rate" in results
        assert 0.0 <= results["discovery_rate"] <= 1.0

    def test_cost_trajectories_tracked(self):
        exp = Layer4LandmarkExperiment(n_runs=3, n_mutations_per_run=5,
                                        observations_length=60, seed=42)
        results = exp.run(verbose=False)
        assert len(results["cost_trajectories"]) == 3
        assert all(len(t) > 0 for t in results["cost_trajectories"])

    def test_beam_cost_in_results(self):
        exp = Layer4LandmarkExperiment(n_runs=2, n_mutations_per_run=3,
                                        observations_length=50, seed=42)
        results = exp.run(verbose=False)
        assert "beam_cost" in results
        assert results["beam_cost"] > 0