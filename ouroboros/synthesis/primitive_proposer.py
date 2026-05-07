"""
ouroboros.synthesis.primitive_proposer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
PrimitiveProposer: analyses a numeric sequence and, when it detects
structure not captured by the current best expression, proposes a new
named primitive along with detected algebraic properties.

Properties detected
-------------------
* MULTIPLICATIVE   – f(mn) = f(m)f(n) for coprime m, n
* COMPLETELY_MULTIPLICATIVE – f(mn) = f(m)f(n) for all m, n
* ADDITIVE         – f(mn) = f(m)+f(n)
* PERIODIC         – f(n+k) = f(n)
* ALTERNATING      – values alternate sign
* BOUNDED          – |f(n)| ≤ C for some C
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


# ── Helpers ────────────────────────────────────────────────────────────────────

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _coprime(a: int, b: int) -> bool:
    return _gcd(a, b) == 1


def _is_multiplicative(vals: list[float], tol: float = 0.5) -> bool:
    """Check f(mn) ≈ f(m)*f(n) for coprime pairs with valid indices."""
    n = len(vals)
    checks = 0
    hits = 0
    for m in range(2, min(n + 1, 12)):
        for k in range(2, min(n + 1, 12)):
            if _coprime(m, k) and m * k <= n:
                lhs = vals[m * k - 1]
                rhs = vals[m - 1] * vals[k - 1]
                if abs(lhs - rhs) < tol * (abs(rhs) + 1):
                    hits += 1
                checks += 1
    return checks > 0 and hits / checks >= 0.70


def _is_completely_multiplicative(vals: list[float], tol: float = 0.5) -> bool:
    """Check f(mn) ≈ f(m)*f(n) for ALL pairs (not just coprime)."""
    n = len(vals)
    checks = 0
    hits = 0
    for m in range(2, min(n + 1, 10)):
        for k in range(2, min(n + 1, 10)):
            if m * k <= n:
                lhs = vals[m * k - 1]
                rhs = vals[m - 1] * vals[k - 1]
                if abs(lhs - rhs) < tol * (abs(rhs) + 1):
                    hits += 1
                checks += 1
    return checks > 0 and hits / checks >= 0.70


def _is_additive(vals: list[float], tol: float = 0.5) -> bool:
    n = len(vals)
    checks = 0; hits = 0
    for m in range(2, min(n + 1, 10)):
        for k in range(2, min(n + 1, 10)):
            if _coprime(m, k) and m * k <= n:
                lhs = vals[m * k - 1]
                rhs = vals[m - 1] + vals[k - 1]
                if abs(lhs - rhs) < tol * (abs(rhs) + 1):
                    hits += 1
                checks += 1
    return checks > 0 and hits / checks >= 0.70


def _is_periodic(vals: list[float], max_period: int = 20, tol: float = 0.5) -> bool:
    n = len(vals)
    for p in range(1, min(max_period + 1, n // 2)):
        ok = all(
            abs(vals[i] - vals[i + p]) < tol * (abs(vals[i]) + 1)
            for i in range(n - p)
        )
        if ok:
            return True
    return False


def _is_alternating(vals: list[float]) -> bool:
    return all(
        (vals[i] > 0) != (vals[i + 1] > 0)
        for i in range(len(vals) - 1)
        if vals[i] != 0 and vals[i + 1] != 0
    )


def _detect_properties(vals: list[float]) -> list[str]:
    props: list[str] = []
    if _is_completely_multiplicative(vals):
        props.append("COMPLETELY_MULTIPLICATIVE")
        props.append("MULTIPLICATIVE")          # subset relation
    elif _is_multiplicative(vals):
        props.append("MULTIPLICATIVE")
    if _is_additive(vals):
        props.append("ADDITIVE")
    if _is_periodic(vals):
        props.append("PERIODIC")
    if _is_alternating(vals):
        props.append("ALTERNATING")
    if all(abs(v) <= 1.0 + 1e-9 for v in vals):
        props.append("BOUNDED")
    return props


def _sequence_mdl(vals: list[float]) -> float:
    """Very rough MDL estimate in bits."""
    if not vals:
        return 0.0
    var = sum(v ** 2 for v in vals) / len(vals)
    sigma = math.sqrt(max(var, 1e-10))
    return len(vals) * (0.5 * math.log2(2 * math.pi * math.e * sigma ** 2) + 1)


def _name_from_properties(props: list[str]) -> str:
    if "COMPLETELY_MULTIPLICATIVE" in props:
        return "completely_multiplicative_func"
    if "MULTIPLICATIVE" in props and "ALTERNATING" in props:
        return "alternating_multiplicative_func"
    if "MULTIPLICATIVE" in props:
        return "multiplicative_func"
    if "PERIODIC" in props:
        return "periodic_func"
    if "ADDITIVE" in props:
        return "additive_func"
    return "novel_arithmetic_func"


# ── Public data classes ────────────────────────────────────────────────────────

@dataclass
class PrimitiveSpecification:
    proposed_name: str
    detected_properties: List[str]
    sequence_sample: List[float]
    estimated_mdl_gain: float = 0.0
    description: str = ""


@dataclass
class PrimitiveProposal:
    specification: PrimitiveSpecification
    trigger_reason: str = "stuck"

    # convenience pass-throughs so callers can do proposal.detected_properties
    @property
    def detected_properties(self) -> List[str]:
        return self.specification.detected_properties

    @property
    def proposed_name(self) -> str:
        return self.specification.proposed_name


# ── Main class ─────────────────────────────────────────────────────────────────

class PrimitiveProposer:
    """
    Monitors MDL cost and proposes new primitives when improvement stalls.

    Parameters
    ----------
    stuck_threshold : float
        MDL cost (bits) above which the proposer considers the search stuck.
    min_compression_gain : float
        Minimum estimated gain (bits) to bother proposing.
    """

    def __init__(
        self,
        stuck_threshold: float = 100.0,
        min_compression_gain: float = 10.0,
    ):
        self.stuck_threshold = stuck_threshold
        self.min_compression_gain = min_compression_gain

    # ------------------------------------------------------------------
    def maybe_propose(
        self,
        sequence: Sequence[float | int],
        best_expr: Any,
        mdl_cost: float,
    ) -> Optional[PrimitiveProposal]:
        """
        Analyse *sequence* and return a PrimitiveProposal if warranted,
        or None if the search is not stuck / no useful primitive found.
        """
        vals = [float(v) for v in sequence]

        # Is the search stuck?
        baseline = _sequence_mdl(vals)
        gain = mdl_cost - baseline * 0.5   # how much we could save

        if mdl_cost < self.stuck_threshold and gain < self.min_compression_gain:
            return None

        props = _detect_properties(vals)
        if not props:
            return None

        name = _name_from_properties(props)
        spec = PrimitiveSpecification(
            proposed_name=name,
            detected_properties=props,
            sequence_sample=vals[:20],
            estimated_mdl_gain=max(0.0, gain),
            description=(
                f"Auto-proposed primitive '{name}' with properties: "
                + ", ".join(props)
            ),
        )
        return PrimitiveProposal(specification=spec, trigger_reason="stuck")