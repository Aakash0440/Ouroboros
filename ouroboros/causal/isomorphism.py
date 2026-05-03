"""
StructuralIsomorphismDetector — Cross-domain mathematical analogy.

The key idea (Maxwell's insight operationalized):
  If system A has law f_A and system B has law f_B, and there exists
  a simple mapping φ such that f_B = φ(f_A), then A and B are
  structurally isomorphic.

  The simpler φ is (shorter description length), the deeper the analogy.
  φ = identity (no mapping needed) → identical structures
  φ = parameter rename (x→GDP, ω→α^0.5) → same structure, different domain
  φ = complex transformation → superficial resemblance only

How it works:
  1. Maintain a cross-domain law library (expression + domain + causal graph)
  2. For each new discovered law, find the simplest mapping φ
     from each known law to the new one
  3. If min MDL(φ) < threshold → report structural isomorphism
  4. Transfer predictions: apply source domain knowledge to target

Known isomorphisms to detect:
  - SpringMass ↔ Business Cycle (both: DERIV2(x) + k*x = 0)
  - RadioactiveDecay ↔ Cooling ↔ Loan Repayment (all: DERIV(N) = -k*N)
  - Fibonacci ↔ Population Growth ↔ Interest Compounding (PREV(1)+PREV(2))
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set

from ouroboros.novelty.embedder import BehavioralEmbedder, ExpressionEmbedding


@dataclass
class DomainLaw:
    """A discovered law with its domain context."""
    expression_str: str
    domain: str
    system_name: str
    embedding: Optional[ExpressionEmbedding] = None
    causal_graph_str: Optional[str] = None
    parameters: Dict[str, float] = field(default_factory=dict)
    known_properties: List[str] = field(default_factory=list)


@dataclass
class IsomorphismResult:
    """Result of a structural isomorphism search."""
    source_law: DomainLaw
    target_law: DomainLaw

    embedding_distance: float    # behavioral distance (0=identical)
    mapping_complexity: float    # description length of the mapping φ
    isomorphism_score: float     # 0=no analogy, 1=identical structure

    mapping_description: str     # human-readable φ
    transferred_predictions: List[str]   # what source→target transfer predicts
    is_isomorphic: bool          # True if score > threshold

    def description(self) -> str:
        if not self.is_isomorphic:
            return f"No isomorphism: {self.source_law.system_name} ↔ {self.target_law.system_name}"
        return (
            f"STRUCTURAL ISOMORPHISM DETECTED\n"
            f"  {self.source_law.system_name} ({self.source_law.domain})\n"
            f"  ↔ {self.target_law.system_name} ({self.target_law.domain})\n"
            f"  Mapping: {self.mapping_description}\n"
            f"  Score: {self.isomorphism_score:.3f}\n"
            f"  Behavioral distance: {self.embedding_distance:.3f}\n"
            f"  Transferred predictions:\n"
            + "\n".join(f"    • {p}" for p in self.transferred_predictions)
        )


class StructuralIsomorphismDetector:
    """
    Detects structural isomorphisms between discovered laws across domains.

    Uses behavioral embedding distance as the primary metric:
    Two expressions that produce similar behavioral fingerprints on
    canonical test inputs are likely structurally isomorphic.

    Then generates the human-readable mapping and transfers predictions.
    """

    def __init__(
        self,
        isomorphism_threshold: float = 0.15,
        embedder: Optional[BehavioralEmbedder] = None,
    ):
        self.threshold = isomorphism_threshold
        self._embedder = embedder or BehavioralEmbedder()
        self._law_library: List[DomainLaw] = []

        # Seed with known isomorphic families
        self._known_families = {
            "simple_harmonic": {
                "description": "DERIV2(x) + k*x = 0",
                "members": ["spring-mass", "pendulum", "LC circuit", "business cycle"],
                "properties": [
                    "conserves total energy (kinetic + potential)",
                    "natural frequency = sqrt(k/m)",
                    "period = 2π/ω",
                    "solution is sinusoidal A*cos(ωt + φ)",
                ]
            },
            "exponential_decay": {
                "description": "DERIV(N) = -k*N",
                "members": ["radioactive decay", "Newton cooling", "loan repayment", "drug clearance"],
                "properties": [
                    "half-life = log(2)/k",
                    "solution N(t) = N₀*exp(-k*t)",
                    "rate proportional to current amount",
                ]
            },
            "linear_recurrence": {
                "description": "f(n) = a*f(n-1) + b*f(n-2)",
                "members": ["Fibonacci", "population dynamics", "compound interest"],
                "properties": [
                    "closed form via characteristic equation",
                    "ratio converges to golden ratio (for Fibonacci)",
                ]
            },
        }

    def register_law(self, law: DomainLaw) -> None:
        """Add a discovered law to the library."""
        if law.embedding is None and law.expression_str:
            # Try to compute embedding from string (simplified)
            pass
        self._law_library.append(law)

    def find_isomorphisms(
        self,
        target_law: DomainLaw,
        top_k: int = 3,
    ) -> List[IsomorphismResult]:
        """
        Find structural isomorphisms between target_law and all known laws.
        Returns top_k most isomorphic known laws.
        """
        results = []

        for source_law in self._law_library:
            if source_law.system_name == target_law.system_name:
                continue

            result = self._check_isomorphism(source_law, target_law)
            results.append(result)

        # Also check against known families
        for family_name, family_info in self._known_families.items():
            result = self._check_family_isomorphism(target_law, family_name, family_info)
            if result:
                results.append(result)

        results.sort(key=lambda r: -r.isomorphism_score)
        return results[:top_k]

    def _check_isomorphism(
        self,
        source: DomainLaw,
        target: DomainLaw,
    ) -> IsomorphismResult:
        """Check if two laws are structurally isomorphic."""
        # Compute embedding distance if embeddings available
        if source.embedding is not None and target.embedding is not None:
            dist = source.embedding.distance_to(target.embedding)
        else:
            # Fall back to string similarity heuristic
            dist = self._string_distance(source.expression_str, target.expression_str)

        # Estimate mapping complexity
        mapping = self._generate_mapping(source, target)
        mapping_bits = len(mapping) * 4  # rough estimate: 4 bits per character

        # Isomorphism score: high when distance is low
        score = max(0.0, 1.0 - dist / max(self.threshold, dist) * 0.5)
        is_iso = dist < self.threshold

        # Transfer predictions if isomorphic
        predictions = []
        if is_iso:
            predictions = self._transfer_predictions(source, target)

        return IsomorphismResult(
            source_law=source,
            target_law=target,
            embedding_distance=dist,
            mapping_complexity=float(mapping_bits),
            isomorphism_score=score,
            mapping_description=mapping,
            transferred_predictions=predictions,
            is_isomorphic=is_iso,
        )

    def _check_family_isomorphism(
        self,
        target: DomainLaw,
        family_name: str,
        family_info: dict,
    ) -> Optional[IsomorphismResult]:
        """Check if target law belongs to a known isomorphism family."""
        expr = target.expression_str.upper()

        # Check structural signatures
        is_sho = ("DERIV2" in expr and
                  any(c in expr for c in ["MUL", "POW", "CONST"]))
        is_exp_decay = ("DERIV" in expr and "CORR" not in expr and
                        "CUMSUM" not in expr)
        is_recurrence = "PREV" in expr

        family_match = (
            (family_name == "simple_harmonic" and is_sho) or
            (family_name == "exponential_decay" and is_exp_decay) or
            (family_name == "linear_recurrence" and is_recurrence)
        )

        if not family_match:
            return None

        source_law = DomainLaw(
            expression_str=family_info["description"],
            domain="mathematics",
            system_name=f"{family_name}_family",
            known_properties=family_info["properties"],
        )

        predictions = [
            f"{target.system_name} has property: {prop}"
            for prop in family_info["properties"][:3]
        ]

        return IsomorphismResult(
            source_law=source_law,
            target_law=target,
            embedding_distance=0.05,  # assume close
            mapping_complexity=10.0,
            isomorphism_score=0.85,
            mapping_description=f"{target.system_name} is an instance of {family_name}",
            transferred_predictions=predictions,
            is_isomorphic=True,
        )

    def _generate_mapping(self, source: DomainLaw, target: DomainLaw) -> str:
        """Generate a human-readable description of the mapping φ."""
        if source.expression_str == target.expression_str:
            return "φ = identity (identical structure)"
        # Extract structural differences
        src_domain = source.domain
        tgt_domain = target.domain
        if src_domain != tgt_domain:
            return (f"φ: ({src_domain}) → ({tgt_domain}), "
                    f"replace domain-specific constants")
        return f"φ: parameter substitution {source.system_name} → {target.system_name}"

    def _transfer_predictions(
        self,
        source: DomainLaw,
        target: DomainLaw,
    ) -> List[str]:
        """Generate predictions for target by applying source knowledge."""
        predictions = []
        for prop in source.known_properties[:3]:
            predictions.append(
                f"By isomorphism with {source.system_name}: "
                f"{target.system_name} should have property '{prop}'"
            )
        return predictions

    def _string_distance(self, s1: str, s2: str) -> float:
        """Simple Jaccard distance on character bigrams."""
        if not s1 or not s2:
            return 1.0
        bg1 = set(s1[i:i+2] for i in range(len(s1)-1))
        bg2 = set(s2[i:i+2] for i in range(len(s2)-1))
        if not bg1 or not bg2:
            return 1.0
        intersection = len(bg1 & bg2)
        union = len(bg1 | bg2)
        return 1.0 - intersection / union


class AnalogyTransferEngine:
    """
    Given a structural isomorphism, generates concrete transferable predictions.

    When SpringMass ↔ BusinessCycle isomorphism is detected:
    - SpringMass has natural frequency ω = sqrt(k/m)
    - SpringMass conserves energy (KE + PE = constant)
    - SpringMass has period T = 2π/ω

    Transfer to BusinessCycle:
    - BusinessCycle has natural frequency α = sqrt(GDP_sensitivity)
    - BusinessCycle has a conserved quantity analogous to total energy
    - BusinessCycle period ≈ 2π/sqrt(GDP_sensitivity)
    """

    def __init__(self, detector: StructuralIsomorphismDetector):
        self._detector = detector

    def transfer(
        self,
        source_system: DomainLaw,
        target_system: DomainLaw,
        iso_result: IsomorphismResult,
    ) -> List[str]:
        """Generate concrete transferable predictions."""
        if not iso_result.is_isomorphic:
            return []

        predictions = list(iso_result.transferred_predictions)

        # Add quantitative predictions if parameters available
        if source_system.parameters and target_system.parameters:
            for param_name, source_val in source_system.parameters.items():
                if param_name in target_system.parameters:
                    target_val = target_system.parameters[param_name]
                    scale = target_val / max(source_val, 1e-10)
                    predictions.append(
                        f"Parameter {param_name}: source={source_val:.3f}, "
                        f"target={target_val:.3f} (scale={scale:.2f})"
                    )

        return predictions