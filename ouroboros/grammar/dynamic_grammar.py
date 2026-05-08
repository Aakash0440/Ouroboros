"""
ouroboros.grammar.dynamic_grammar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
DynamicGrammar: registers verified primitives as new callable node types
and injects them into MathGrammar's _RULES so the beam search can use them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any

from ouroboros.grammar.math_grammar import _RULES, _SCALAR, _CONST, MathGrammar
from ouroboros.nodes.extended_nodes import NODE_SPECS, NodeCategory


# ── Pseudo node type ──────────────────────────────────────────────────────────
# ExtNodeType is a static enum so we can't extend it at runtime.
# We create lightweight objects that satisfy the .name / .category / .arity
# interface expected by MathGrammar and the beam search.

@dataclass
class DynamicNodeType:
    name: str
    category: NodeCategory
    arity: int
    fn: Callable[[int], float]          # fn(index_1based) -> value
    description: str = ""

    # Makes it hashable so it can be used as a dict key
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, DynamicNodeType) and self.name == other.name

    def __repr__(self):
        return f"DynamicNodeType({self.name})"


# ── Registry ──────────────────────────────────────────────────────────────────

class DynamicGrammar:
    """
    Accepts VerificationResult objects and registers them as usable nodes.

    Usage
    -----
    dg = DynamicGrammar()
    dg.register(verification_result, name="my_func")

    # Then pass dg to the beam search so it can evaluate dynamic nodes:
    value = dg.evaluate(node_type, t)
    """

    def __init__(self):
        self._registry: Dict[str, DynamicNodeType] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        verification_result: Any,
        name: Optional[str] = None,
    ) -> Optional[DynamicNodeType]:
        """
        Register a verified primitive. Returns the DynamicNodeType if
        successful, None if the result is invalid.
        """
        if not getattr(verification_result, "is_valid", False):
            return None

        impl_str = getattr(verification_result, "python_implementation", None)
        if not impl_str:
            return None

        # Build callable from implementation string
        fn = self._compile(impl_str)
        if fn is None:
            return None

        # Pick a name
        node_name = (name or f"DYN_{len(self._registry)}").upper()
        if node_name in self._registry:
            return self._registry[node_name]   # already registered

        props = getattr(verification_result, "verified_properties", [])
        node = DynamicNodeType(
            name=node_name,
            category=NodeCategory.ARITHMETIC,   # default; could refine
            arity=0,                            # terminal: takes no children, uses t
            fn=fn,
            description=", ".join(props),
        )

        # Inject into NODE_SPECS so MathGrammar can iterate over it
        NODE_SPECS[node] = _make_node_spec(node)

        # Inject into _RULES so grammar allows it as a terminal
        _RULES[node_name] = (frozenset(), None, None)

        self._registry[node_name] = node
        return node

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, node_type: DynamicNodeType, t: int) -> float:
        """Evaluate a dynamic node at time index t (1-based for the fn)."""
        try:
            return float(node_type.fn(t + 1))   # fn expects 1-based index
        except Exception:
            return 0.0

    def get(self, name: str) -> Optional[DynamicNodeType]:
        return self._registry.get(name.upper())

    def all_nodes(self) -> List[DynamicNodeType]:
        return list(self._registry.values())

    def __len__(self):
        return len(self._registry)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compile(impl_str: str) -> Optional[Callable]:
        """Compile the implementation string into a callable."""
        ns: dict = {}
        try:
            exec(impl_str, ns)
        except Exception:
            return None

        # PrimitiveVerifier puts the callable name as the last line
        last_line = impl_str.strip().splitlines()[-1].strip()

        if last_line.startswith("lambda"):
            try:
                fn = eval(last_line, ns)
            except Exception:
                return None
        else:
            fn = (
                ns.get(last_line)
                or ns.get("_liouville_impl")
                or ns.get("_completely_mult_impl")
                or ns.get("_multiplicative_impl")
            )

        if not callable(fn):
            return None

        # Smoke test
        try:
            fn(1); fn(2); fn(3)
        except Exception:
            return None

        return fn


# ── NodeSpec shim ─────────────────────────────────────────────────────────────

def _make_node_spec(node: DynamicNodeType):
    """
    Create a minimal NodeSpec-compatible object for NODE_SPECS injection.
    Mirrors the SimpleNamespace pattern used by extended_nodes.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        name=node.name,
        category=node.category,
        arity=node.arity,
        description=node.description,
        window_arg=False,
        const_arg=False,
        output_type="scalar",
    )


# ── Global instance ───────────────────────────────────────────────────────────
DYNAMIC_GRAMMAR = DynamicGrammar()