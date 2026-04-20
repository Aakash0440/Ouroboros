# tests/unit/test_base_agent.py

"""Unit tests for BaseAgent and NGramProgram."""

import pytest
from ouroboros.agents.base_agent import BaseAgent, NGramProgram, build_ngram_table
from ouroboros.environments import ModularArithmeticEnv, NoiseEnv


class TestNGramProgram:

    def test_predict_known_context(self):
        prog = NGramProgram(context_length=2)
        prog.table = {(0, 1): 2, (1, 2): 0, (2, 0): 1}
        assert prog.predict([0, 1]) == 2
        assert prog.predict([5, 1, 2]) == 0

    def test_predict_unknown_context_returns_int(self):
        prog = NGramProgram(context_length=1, table={(0,): 1})
        result = prog.predict([5])  # 5 not in table
        assert isinstance(result, int)

    def test_empty_history_returns_zero(self):
        prog = NGramProgram(context_length=2)
        prog.table = {(0, 1): 3}
        assert prog.predict([]) == 0

    def test_to_bytes_is_bytes(self):
        prog = NGramProgram(context_length=1, table={(0,): 1, (1,): 0})
        b = prog.to_bytes()
        assert isinstance(b, bytes)
        assert len(b) > 0

    def test_empty_table_serializes(self):
        prog = NGramProgram()
        b = prog.to_bytes()
        assert b == b"empty"


class TestBuildNgramTable:

    def test_periodic_stream(self):
        # 0,1,2,0,1,2,... — context (0,) should predict 1 most of time
        stream = [0, 1, 2] * 100
        table = build_ngram_table(stream, context_length=1)
        assert (0,) in table
        assert table[(0,)] == 1

    def test_table_covers_observed_contexts(self):
        stream = [0, 1, 0, 1, 0, 1]
        table = build_ngram_table(stream, context_length=1)
        assert (0,) in table
        assert (1,) in table

    def test_k2_context(self):
        stream = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        table = build_ngram_table(stream, context_length=2)
        assert (0, 1) in table
        assert table[(0, 1)] == 2


class TestBaseAgent:

    def test_initial_ratio_is_one(self):
        agent = BaseAgent(0, 7)
        assert agent.measure_compression_ratio() == 1.0

    def test_observe_extends_history(self):
        agent = BaseAgent(0, 2)
        agent.observe([0, 1, 0, 1])
        assert len(agent.observation_history) == 4

    def test_search_returns_finite_cost(self):
        env = ModularArithmeticEnv(7, 3, 1)
        env.reset(200)
        stream = env.peek_all()
        agent = BaseAgent(0, 7)
        agent.observe(stream)
        cost = agent.search_and_update()
        assert cost < float('inf')
        assert cost > 0

    def test_beats_random_on_modular_env(self):
        env = ModularArithmeticEnv(7, 3, 1)
        env.reset(500)
        stream = env.peek_all()
        agent = BaseAgent(0, 7, max_context_length=4)
        agent.observe(stream)
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        assert ratio < 1.0, f"Expected < 1.0, got {ratio:.4f}"

    def test_does_not_compress_noise(self):
        env = NoiseEnv(4, seed=7)
        env.reset(500)
        stream = env.peek_all()
        agent = BaseAgent(0, 4)
        agent.observe(stream)
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        # Noise should be hard to compress
        assert ratio > 0.80, f"Noise should not compress — got {ratio:.4f}"

    def test_status_dict_keys(self):
        agent = BaseAgent(0, 7)
        agent.observe([1, 2, 3, 4, 5])
        agent.search_and_update()
        s = agent.status_dict()
        for key in ['agent_id', 'observations', 'context_k',
                    'table_entries', 'compression_ratio']:
            assert key in s, f"Missing key: {key}"

    def test_set_history_replaces(self):
        agent = BaseAgent(0, 2)
        agent.observe([0, 1, 0])
        agent.set_history([1, 1, 1])
        assert agent.observation_history == [1, 1, 1]