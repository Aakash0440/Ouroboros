"""Tests for the mathematical grammar constraint system."""
import pytest
from ouroboros.grammar.math_grammar import MathGrammar, DEFAULT_GRAMMAR, ANY_TYPES
from ouroboros.nodes.extended_nodes import ExtNodeType, NodeCategory


class TestMathGrammar:
    def test_deriv_allows_continuous(self):
        allowed = DEFAULT_GRAMMAR.allowed_child_categories(ExtNodeType.DERIV, 0)
        assert NodeCategory.CALCULUS in allowed
        assert NodeCategory.TRANSCEND in allowed

    def test_bool_and_requires_logical_children(self):
        allowed = DEFAULT_GRAMMAR.allowed_child_categories(ExtNodeType.BOOL_AND, 0)
        assert NodeCategory.LOGICAL in allowed
        assert NodeCategory.CALCULUS not in allowed

    def test_fft_amp_requires_terminal_for_freq(self):
        allowed = DEFAULT_GRAMMAR.allowed_child_categories(ExtNodeType.FFT_AMP, 1)
        assert NodeCategory.TERMINAL in allowed
        assert len(allowed) == 1  # only TERMINAL for freq index

    def test_effective_branching_lower_than_60(self):
        bf = DEFAULT_GRAMMAR.effective_branching_factor(0)
        # Should be much less than 60 (the unconstrained case)
        assert bf < 44

    def test_permissive_grammar_allows_all(self):
        from ouroboros.grammar.math_grammar import PERMISSIVE_GRAMMAR
        allowed = PERMISSIVE_GRAMMAR.allowed_child_categories(ExtNodeType.DERIV, 0)
        assert allowed == ANY_TYPES

    def test_category_of_known_type(self):
        cat = DEFAULT_GRAMMAR.category_of(ExtNodeType.DERIV)
        assert cat == NodeCategory.CALCULUS

    def test_search_space_smaller_with_grammar(self):
        unconstrained = 60 ** 16
        grammar_size = DEFAULT_GRAMMAR.search_space_size(max_depth=4)
        assert grammar_size < unconstrained