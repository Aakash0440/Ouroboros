"""
OuroborosClient — Python client for the OUROBOROS web API.

Usage:
    client = OuroborosClient("http://localhost:8000")
    result = client.discover([1, 4, 0, 3, 6, 2, 5, 1, 4, 0], alphabet_size=7)
    print(result.expression)  # "(3*t+1) % 7"
    print(result.mdl_cost)    # 45.21
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class DiscoveryResult:
    """Result from the OUROBOROS discovery API."""
    expression: Optional[str]
    mdl_cost: float
    compression_ratio: float
    math_family: str
    confidence: float
    verified_law: str
    lean4_theorem: Optional[str]
    runtime_seconds: float
    n_observations: int
    alphabet_size_used: int
    from_cache: bool

    def __str__(self) -> str:
        return (
            f"Expression: {self.expression}\n"
            f"MDL cost: {self.mdl_cost:.3f} bits\n"
            f"Compression ratio: {self.compression_ratio:.4f}\n"
            f"Family: {self.math_family} (confidence={self.confidence:.2f})\n"
            f"Physics law: {self.verified_law}\n"
            f"Runtime: {self.runtime_seconds:.2f}s"
        )


class OuroborosClient:
    """
    Client for the OUROBOROS web API.
    
    Can use either httpx (async) or the built-in http.client (sync).
    Falls back to direct Python call if base_url is None.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip('/') if base_url else None
        self.timeout = timeout

    def discover(
        self,
        observations: List[float],
        alphabet_size: Optional[int] = None,
        beam_width: int = 20,
        max_depth: int = 4,
        n_iterations: int = 10,
        time_budget_seconds: float = 10.0,
        verify_physics_laws: bool = True,
        return_lean4: bool = False,
    ) -> DiscoveryResult:
        """
        Discover a symbolic expression for the observation sequence.
        
        If base_url is None: runs discovery locally (no HTTP overhead).
        If base_url is set: POSTs to the API server.
        """
        if self.base_url is None:
            return self._discover_local(
                observations, alphabet_size, beam_width, max_depth,
                n_iterations, time_budget_seconds, verify_physics_laws, return_lean4,
            )
        return self._discover_remote(
            observations, alphabet_size, beam_width, max_depth,
            n_iterations, time_budget_seconds, verify_physics_laws, return_lean4,
        )

    def _discover_local(
        self,
        observations, alphabet_size, beam_width, max_depth,
        n_iterations, time_budget_seconds, verify_physics_laws, return_lean4,
    ) -> DiscoveryResult:
        """Run discovery locally without HTTP."""
        from ouroboros.api.server import (
            _run_discovery, _run_law_verification, _generate_lean4_stub,
        )
        from ouroboros.physics.law_signature import PhysicsLaw

        alphabet_size = alphabet_size or (max(int(v) for v in observations) + 2)
        result = _run_discovery(
            observations, alphabet_size, beam_width, max_depth,
            n_iterations, time_budget_seconds,
        )

        verified_law = "NONE"
        if verify_physics_laws:
            try:
                law_result = _run_law_verification(observations)
                verified_law = law_result["primary_law"]
            except Exception:
                pass

        lean4 = None
        if return_lean4 and result.get("expression"):
            lean4 = _generate_lean4_stub(result["expression"], result["math_family"])

        return DiscoveryResult(
            expression=result.get("expression"),
            mdl_cost=result.get("mdl_cost", 9999.0),
            compression_ratio=result.get("compression_ratio", 1.0),
            math_family=result.get("math_family", "MIXED"),
            confidence=result.get("confidence", 0.0),
            verified_law=verified_law,
            lean4_theorem=lean4,
            runtime_seconds=result.get("runtime_seconds", 0.0),
            n_observations=len(observations),
            alphabet_size_used=alphabet_size,
            from_cache=False,
        )

    def _discover_remote(
        self,
        observations, alphabet_size, beam_width, max_depth,
        n_iterations, time_budget_seconds, verify_physics_laws, return_lean4,
    ) -> DiscoveryResult:
        """POST to the remote API server."""
        import urllib.request
        import urllib.error

        payload = {
            "observations": observations,
            "alphabet_size": alphabet_size,
            "beam_width": beam_width,
            "max_depth": max_depth,
            "n_iterations": n_iterations,
            "time_budget_seconds": time_budget_seconds,
            "verify_physics_laws": verify_physics_laws,
            "return_lean4": return_lean4,
        }

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/discover",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = json.loads(e.read().decode())
            raise RuntimeError(f"API error {e.code}: {error_body.get('detail', str(e))}")

        return DiscoveryResult(
            expression=body.get("expression"),
            mdl_cost=body.get("mdl_cost", 9999.0),
            compression_ratio=body.get("compression_ratio", 1.0),
            math_family=body.get("math_family", "MIXED"),
            confidence=body.get("confidence", 0.0),
            verified_law=body.get("verified_law", "NONE"),
            lean4_theorem=body.get("lean4_theorem"),
            runtime_seconds=body.get("runtime_seconds", 0.0),
            n_observations=len(observations),
            alphabet_size_used=body.get("alphabet_size_used", 0),
            from_cache=body.get("from_cache", False),
        )

    def verify_law(self, observations: List[float]) -> dict:
        """Check if sequence satisfies a known physics law."""
        if self.base_url is None:
            from ouroboros.api.server import _run_law_verification
            return _run_law_verification(observations)

        import urllib.request
        payload = json.dumps({"observations": observations}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/verify_law",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode())
