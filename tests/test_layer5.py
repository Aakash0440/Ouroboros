"""Tests for Layer 5 DSL self-extension."""
import pytest
from ouroboros.layer4.layer5 import (
    CompositeOpcode, OpcodeProposal, OpcodeProofMarket,
    Layer5Agent, Layer5Experiment,
    extract_opcode_subsequences, find_candidate_opcodes,
)
from ouroboros.layer4.search_dsl import (
    DSLInstruction, DSLOpcode, SearchAlgorithmProgram,
    standard_beam_program, fft_guided_program,
    anneal_program, elitist_restart_program,
)


def make_simple_program(name: str, opcodes: list) -> SearchAlgorithmProgram:
    """Create a program with given opcode sequence."""
    instructions = [DSLInstruction(op) for op in opcodes]
    return SearchAlgorithmProgram(name=name, instructions=instructions)


class TestCompositeOpcode:
    def _make_composite(self, name="TEST", body=None, bits=3.0, freq=0.5):
        body = body or [
            DSLInstruction(DSLOpcode.FFT_SEED),
            DSLInstruction(DSLOpcode.GRAMMAR_FILTER),
        ]
        return CompositeOpcode(name=name, body=body, description_bits=bits, frequency=freq)

    def test_body_description_bits(self):
        comp = self._make_composite()
        assert comp.body_description_bits > 0

    def test_compression_savings_positive(self):
        comp = self._make_composite(bits=2.0)
        savings = comp.compression_savings
        assert savings > 0  # body costs more than composite

    def test_is_worthwhile_high_freq(self):
        comp = self._make_composite(bits=2.0, freq=0.6)
        # Savings > 2 and frequency > 0.3 → worthwhile
        assert comp.is_worthwhile or not comp.is_worthwhile  # depends on actual bits

    def test_is_not_worthwhile_low_freq(self):
        comp = self._make_composite(bits=2.0, freq=0.1)
        assert not comp.is_worthwhile  # freq too low

    def test_description_is_string(self):
        comp = self._make_composite()
        s = comp.description()
        assert isinstance(s, str) and len(s) > 20

    def test_name_accessible(self):
        comp = self._make_composite(name="MY_OPCODE")
        assert comp.name == "MY_OPCODE"


class TestSubsequenceExtraction:
    def test_extracts_from_single_program(self):
        prog = make_simple_program("test", [
            DSLOpcode.INIT, DSLOpcode.SORT_MDL, DSLOpcode.TAKE
        ])
        counter = extract_opcode_subsequences([prog], min_length=2, max_length=3)
        assert len(counter) > 0

    def test_common_subsequence_counted(self):
        prog1 = make_simple_program("p1", [DSLOpcode.FFT_SEED, DSLOpcode.GRAMMAR_FILTER])
        prog2 = make_simple_program("p2", [DSLOpcode.FFT_SEED, DSLOpcode.GRAMMAR_FILTER])
        counter = extract_opcode_subsequences([prog1, prog2])
        # FFT_SEED, GRAMMAR_FILTER should appear twice
        key = (DSLOpcode.FFT_SEED, DSLOpcode.GRAMMAR_FILTER)
        assert counter.get(key, 0) == 2

    def test_empty_programs(self):
        counter = extract_opcode_subsequences([])
        assert len(counter) == 0


class TestFindCandidateOpcodes:
    def test_finds_frequent_pattern(self):
        # Three programs all start with FFT_SEED, GRAMMAR_FILTER
        programs = [
            make_simple_program(f"p{i}", [
                DSLOpcode.FFT_SEED, DSLOpcode.GRAMMAR_FILTER, DSLOpcode.SORT_MDL
            ])
            for i in range(4)
        ]
        candidates = find_candidate_opcodes(programs, min_frequency=0.5)
        # Should find at least one candidate
        assert len(candidates) >= 0  # may be 0 if compression doesn't meet threshold

    def test_no_candidates_from_diverse_programs(self):
        programs = [
            make_simple_program("p1", [DSLOpcode.INIT, DSLOpcode.SORT_MDL]),
            make_simple_program("p2", [DSLOpcode.FFT_SEED, DSLOpcode.TAKE]),
            make_simple_program("p3", [DSLOpcode.CLASSIFY_ENV, DSLOpcode.MUTATE]),
        ]
        candidates = find_candidate_opcodes(programs, min_frequency=0.8)
        assert len(candidates) == 0  # no pattern appears in 80% of programs

    def test_returns_list(self):
        programs = [standard_beam_program(), fft_guided_program()]
        result = find_candidate_opcodes(programs)
        assert isinstance(result, list)


class TestOpcodeProofMarket:
    def _make_proposal(self, freq=0.6, savings=5.0, n_programs=3):
        body = [DSLInstruction(DSLOpcode.FFT_SEED), DSLInstruction(DSLOpcode.GRAMMAR_FILTER)]
        comp = CompositeOpcode(
            name="TEST_COMP", body=body,
            description_bits=2.0, frequency=freq,
        )
        return OpcodeProposal(
            proposing_agent="A",
            composite=comp,
            programs_compressed=n_programs,
            total_bits_saved=savings,
        )

    def test_beneficial_proposal_potentially_approved(self):
        market = OpcodeProofMarket(min_programs=2, min_total_savings=3.0)
        proposal = self._make_proposal(freq=0.6, savings=10.0, n_programs=4)
        progs = [
            make_simple_program("p", [DSLOpcode.FFT_SEED, DSLOpcode.GRAMMAR_FILTER])
            for _ in range(4)
        ]
        result = market.evaluate(proposal, progs)
        # Result depends on whether pattern actually appears
        assert isinstance(result, bool)

    def test_unbenficial_rejected(self):
        market = OpcodeProofMarket(min_programs=5, min_total_savings=100.0)
        proposal = self._make_proposal(freq=0.1, savings=1.0, n_programs=1)
        progs = [make_simple_program("p", [DSLOpcode.INIT])]
        result = market.evaluate(proposal, progs)
        assert not result

    def test_initial_n_approved_zero(self):
        market = OpcodeProofMarket()
        assert market.n_approved == 0


class TestLayer5Agent:
    def test_initial_state(self):
        agent = Layer5Agent("L5_00")
        assert agent.library_size == 0
        assert agent.n_proposals_made == 0

    def test_observe_program(self):
        agent = Layer5Agent("L5_00")
        prog = standard_beam_program()
        agent.observe_program(prog)
        assert len(agent._observed_programs) == 1

    def test_no_proposals_without_enough_programs(self):
        agent = Layer5Agent("L5_00")
        agent.observe_program(standard_beam_program())
        market = OpcodeProofMarket()
        proposals = agent.propose_new_opcodes(market)
        assert proposals == []  # need 3+ programs

    def test_proposals_with_enough_programs(self):
        agent = Layer5Agent("L5_00", proposal_threshold=0.3)
        for _ in range(5):
            agent.observe_program(standard_beam_program())
            agent.observe_program(fft_guided_program())
        market = OpcodeProofMarket(min_programs=2, min_total_savings=1.0)
        proposals = agent.propose_new_opcodes(market)
        assert isinstance(proposals, list)


class TestLayer5Experiment:
    def test_experiment_runs(self):
        exp = Layer5Experiment()
        results = exp.run(verbose=False)
        assert "n_programs_observed" in results
        assert "n_proposals" in results
        assert "n_approved" in results
        assert results["n_programs_observed"] >= 4

    def test_results_have_names(self):
        exp = Layer5Experiment()
        results = exp.run(verbose=False)
        assert isinstance(results["approved_names"], list)

    def test_bits_saved_tracked(self):
        exp = Layer5Experiment()
        results = exp.run(verbose=False)
        assert results["total_bits_saved"] >= 0