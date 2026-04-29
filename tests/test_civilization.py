"""Tests for the Mathematical Civilization Simulation."""
import pytest
from ouroboros.civilization.simulator import (
    CivilizationSimulator, MathConcept, ConceptDiscovery,
    MATH_CONCEPTS, AgentSpecialization, CivilizationResult,
)


class TestMathConcept:
    def test_concept_is_indicated_by_node(self):
        concept = MathConcept("Test", "2024", 1, ["ISPRIME"])
        assert concept.is_indicated_by({"ISPRIME", "CONST"})
        assert not concept.is_indicated_by({"CONST", "TIME"})

    def test_all_concepts_have_indicators(self):
        for c in MATH_CONCEPTS:
            assert len(c.indicator_nodes) > 0

    def test_human_order_unique(self):
        orders = [c.human_order for c in MATH_CONCEPTS]
        assert len(set(orders)) == len(orders), "Duplicate human_order values"

    def test_concepts_cover_main_areas(self):
        names = {c.name for c in MATH_CONCEPTS}
        assert any("Modular" in n for n in names)
        assert any("Prime" in n for n in names)
        assert any("Calculus" in n or "Derivative" in n for n in names)
        assert any("Fourier" in n or "FFT" in n for n in names)


class TestCivilizationSimulator:
    def test_tiny_simulation_runs(self):
        sim = CivilizationSimulator(
            n_agents=4, n_rounds=5, stream_length=80,
            beam_width=5, n_iterations=2, verbose=False, report_every=10,
        )
        result = sim.run()
        assert result.n_agents == 4
        assert result.n_rounds == 5
        assert result.total_runtime_seconds > 0

    def test_result_has_discovery_order(self):
        sim = CivilizationSimulator(
            n_agents=4, n_rounds=8, stream_length=80,
            beam_width=5, n_iterations=2, verbose=False, report_every=10,
        )
        result = sim.run()
        assert isinstance(result.ouroboros_discovery_order, list)
        assert isinstance(result.human_discovery_order, list)

    def test_correlation_in_range(self):
        sim = CivilizationSimulator(
            n_agents=4, n_rounds=8, stream_length=80,
            beam_width=5, n_iterations=2, verbose=False, report_every=10,
        )
        result = sim.run()
        if len(result.ouroboros_discovery_order) >= 3:
            assert -1.0 <= result.order_correlation <= 1.0

    def test_extract_nodes_from_const_expr(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=5)
        sim = CivilizationSimulator(n_agents=2, n_rounds=2, verbose=False)
        nodes = sim._extract_nodes_from_expr(expr)
        assert "CONST" in nodes

    def test_extract_nodes_from_complex_expr(self):
        from ouroboros.synthesis.expr_node import ExprNode, NodeType, build_linear_modular
        expr = build_linear_modular(slope=3, intercept=1, modulus=7)
        sim = CivilizationSimulator(n_agents=2, n_rounds=2, verbose=False)
        nodes = sim._extract_nodes_from_expr(expr)
        assert "MOD" in nodes or "MUL" in nodes

    def test_spearman_correlation_identical(self):
        sim = CivilizationSimulator(n_agents=2, n_rounds=2, verbose=False)
        order = ["A", "B", "C", "D", "E"]
        rho = sim._spearman_correlation(order, order)
        assert abs(rho - 1.0) < 0.01

    def test_spearman_correlation_reversed(self):
        sim = CivilizationSimulator(n_agents=2, n_rounds=2, verbose=False)
        order_a = ["A", "B", "C", "D", "E"]
        order_b = ["E", "D", "C", "B", "A"]
        rho = sim._spearman_correlation(order_a, order_b)
        assert rho < -0.8

    def test_spearman_correlation_short(self):
        sim = CivilizationSimulator(n_agents=2, n_rounds=2, verbose=False)
        rho = sim._spearman_correlation(["A"], ["A"])
        assert rho == 0.0  # too short for meaningful correlation

    def test_result_summary_is_string(self):
        sim = CivilizationSimulator(
            n_agents=4, n_rounds=5, stream_length=80,
            beam_width=5, n_iterations=2, verbose=False, report_every=10,
        )
        result = sim.run()
        s = result.summary()
        assert isinstance(s, str) and len(s) > 100

    def test_environment_suite_built(self):
        sim = CivilizationSimulator(n_agents=4, n_rounds=2, verbose=False)
        envs = sim._build_environment_suite()
        assert len(envs) >= 10
        # Should have different types
        names = [e.name for e in envs]
        assert any("Modular" in n for n in names)
        assert any("Fib" in n or "Tribonacci" in n for n in names)

    def test_no_duplicate_concepts_in_order(self):
        sim = CivilizationSimulator(
            n_agents=4, n_rounds=8, stream_length=80,
            beam_width=5, n_iterations=2, verbose=False, report_every=10,
        )
        result = sim.run()
        order = result.ouroboros_discovery_order
        assert len(order) == len(set(order)), "Duplicate concepts in order"

    def test_concept_discovery_fields(self):
        concept = MATH_CONCEPTS[0]
        from ouroboros.synthesis.expr_node import ExprNode, NodeType
        expr = ExprNode(NodeType.CONST, value=5)
        disc = ConceptDiscovery(
            concept=concept, round_discovered=5,
            agent_id="A0", environment_name="TestEnv",
            expression_str="CONST(5)", mdl_cost=45.0,
        )
        assert disc.round_discovered == 5
        assert disc.concept.name == concept.name