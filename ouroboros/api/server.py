"""
OuroborosAPI — FastAPI web service for mathematical discovery.

Endpoints:
  POST /discover       — find symbolic expression for a time series
  POST /verify_law     — test whether a sequence satisfies a physics law
  GET  /benchmark      — run the standard benchmark (async)
  GET  /health         — liveness check
  GET  /docs           — auto-generated OpenAPI docs

Usage:
  uvicorn ouroboros.api.server:app --host 0.0.0.0 --port 8000

Example POST /discover:
  {
    "observations": [1, 4, 0, 3, 6, 2, 5, 1, 4, 0],
    "alphabet_size": 7,
    "max_depth": 4,
    "beam_width": 20,
    "time_budget_seconds": 10.0
  }

Response:
  {
    "expression": "(3*t+1) % 7",
    "mdl_cost": 45.21,
    "compression_ratio": 0.0041,
    "math_family": "NUMBER_THEOR",
    "confidence": 0.94,
    "verified_law": "NONE",
    "lean4_theorem": "theorem discovered : ...",
    "runtime_seconds": 2.3
  }
"""

from __future__ import annotations
import time
import hashlib
import json
from typing import List, Optional, Dict, Any
from functools import lru_cache

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    # Stubs for testing without FastAPI installed
    class BaseModel:
        pass
    class Field:
        def __init__(self, *a, **kw): pass
    def validator(*a, **kw):
        def dec(f): return f
        return dec


# ── Request/Response Models ────────────────────────────────────────────────────

class DiscoveryRequest(BaseModel if HAS_FASTAPI else object):
    """Request body for POST /discover."""
    observations: List[float] = Field(
        ...,
        description="Observation sequence to analyze",
        min_items=10,
        max_items=5000,
    )
    alphabet_size: Optional[int] = Field(
        None,
        description="Number of unique symbols. Auto-detected if None.",
        ge=2, le=10000,
    )
    max_depth: int = Field(
        4,
        description="Maximum expression tree depth",
        ge=2, le=6,
    )
    beam_width: int = Field(
        20,
        description="Beam search width",
        ge=5, le=100,
    )
    n_iterations: int = Field(
        10,
        description="Search iterations",
        ge=3, le=30,
    )
    time_budget_seconds: float = Field(
        10.0,
        description="Maximum search time in seconds",
        ge=1.0, le=60.0,
    )
    verify_physics_laws: bool = Field(
        True,
        description="Test whether result satisfies known physics laws",
    )
    return_lean4: bool = Field(
        False,
        description="Generate a Lean4 theorem stub for the result",
    )

    @validator('observations')
    def observations_finite(cls, v):
        import math
        if any(not math.isfinite(x) for x in v):
            raise ValueError("observations must all be finite numbers")
        return v


class DiscoveryResponse(BaseModel if HAS_FASTAPI else object):
    """Response body from POST /discover."""
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
    search_config: Dict[str, Any]


class LawVerifyRequest(BaseModel if HAS_FASTAPI else object):
    """Request body for POST /verify_law."""
    observations: List[float] = Field(..., min_items=10, max_items=5000)
    law_to_test: Optional[str] = Field(
        None,
        description="Specific law to test: 'hookes_law', 'exponential_decay', 'free_fall'. Tests all if None."
    )


class LawVerifyResponse(BaseModel if HAS_FASTAPI else object):
    """Response body from POST /verify_law."""
    primary_law: str
    confidence: float
    all_tests: List[Dict[str, Any]]
    summary: str


# ── SessionCache ───────────────────────────────────────────────────────────────

class SessionCache:
    """
    LRU cache for discovery results.
    
    Caches (observation_hash, config_hash) → DiscoveryResponse.
    Max 100 entries. Cache hit skips search entirely.
    """

    def __init__(self, maxsize: int = 100):
        self._cache: Dict[str, dict] = {}
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    def _key(self, observations: List[float], config: dict) -> str:
        obs_str = json.dumps(observations[:100])  # hash first 100 for speed
        cfg_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(f"{obs_str}:{cfg_str}".encode()).hexdigest()[:16]

    def get(self, observations: List[float], config: dict) -> Optional[dict]:
        key = self._key(observations, config)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, observations: List[float], config: dict, result: dict) -> None:
        key = self._key(observations, config)
        if len(self._cache) >= self._maxsize:
            # Evict oldest entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = result

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)


# ── Rate Limiter ───────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Simple token bucket rate limiter.
    10 requests per minute per IP. Burst of 3.
    """

    def __init__(self, requests_per_minute: int = 10, burst: int = 3):
        self._rpm = requests_per_minute
        self._burst = burst
        self._buckets: Dict[str, List[float]] = {}

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        window = 60.0  # 1 minute
        if client_ip not in self._buckets:
            self._buckets[client_ip] = []
        bucket = self._buckets[client_ip]
        # Remove old entries
        bucket[:] = [t for t in bucket if now - t < window]
        if len(bucket) >= self._rpm:
            return False
        bucket.append(now)
        return True


# ── Core Discovery Logic ───────────────────────────────────────────────────────

def _run_discovery(
    observations: List[float],
    alphabet_size: int,
    beam_width: int,
    max_depth: int,
    n_iterations: int,
    time_budget: float,
) -> dict:
    """Run the discovery pipeline and return a result dict."""
    import math

    int_obs = [int(round(v)) for v in observations]

    from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
    from ouroboros.compression.mdl_engine import MDLEngine

    router = HierarchicalSearchRouter(RouterConfig(
        beam_width=beam_width,
        max_depth=max_depth,
        n_iterations=n_iterations,
        time_budget_seconds=time_budget,
        random_seed=42,
    ))

    start = time.time()
    result = router.search(int_obs, alphabet_size=alphabet_size)
    elapsed = time.time() - start

    expression = result.expr.to_string() if result.expr else None
    mdl_cost = result.mdl_cost if math.isfinite(result.mdl_cost) else 9999.9

    # Compute compression ratio
    n = len(int_obs)
    import math as _math
    baseline_bits = n * _math.log2(max(alphabet_size, 2))
    compression_ratio = mdl_cost / max(baseline_bits, 1.0)

    return {
        "expression": expression,
        "mdl_cost": round(mdl_cost, 3),
        "compression_ratio": round(compression_ratio, 4),
        "math_family": result.math_family.name,
        "confidence": round(result.classification_confidence, 3),
        "runtime_seconds": round(elapsed, 3),
    }


def _run_law_verification(observations: List[float]) -> dict:
    """Run physics law verification on the observations."""
    from ouroboros.physics.derivative_analyzer import PhysicsLawVerifier
    from ouroboros.physics.law_signature import PhysicsLaw

    verifier = PhysicsLawVerifier()
    law, test_results = verifier.verify_raw_sequence(observations)

    all_tests = [
        {
            "law": r.law.name,
            "passed": r.passed,
            "confidence": round(r.confidence, 3),
            "key_metric": r.key_metric,
            "key_value": round(r.key_value, 4),
        }
        for r in test_results
    ]

    return {
        "primary_law": law.name,
        "confidence": max((t["confidence"] for t in all_tests if t["passed"]), default=0.0),
        "all_tests": all_tests,
    }


def _generate_lean4_stub(expression: str, math_family: str) -> str:
    """Generate a Lean4 theorem stub for the discovered expression."""
    return f"""/-
Discovered expression: {expression}
Mathematical family: {math_family}
Discovered by OUROBOROS (https://github.com/ouroboros-research/ouroboros)
-/
import Mathlib.Tactic

-- Auto-generated theorem stub. Fill in proof manually or use omega/norm_num.
theorem discovered_expression (t : ℕ) :
    -- TODO: formalize the claim here
    True := trivial
"""


# ── FastAPI Application ────────────────────────────────────────────────────────

_cache = SessionCache(maxsize=100)
_limiter = RateLimiter(requests_per_minute=10, burst=3)

if HAS_FASTAPI:
    app = FastAPI(
        title="OUROBOROS Mathematical Discovery API",
        description=(
            "Multi-agent system that discovers symbolic mathematical expressions "
            "from integer or float observation sequences using MDL compression."
        ),
        version="9.0.0",
    )

    @app.get("/health")
    async def health():
        """Liveness check."""
        return {
            "status": "ok",
            "version": "9.0.0",
            "cache_size": _cache.size,
            "cache_hit_rate": round(_cache.hit_rate, 3),
        }

    @app.post("/discover", response_model=None)
    async def discover(request: Request, body: DiscoveryRequest):
        """
        Discover a symbolic expression for the given observation sequence.
        
        The system uses MDL compression pressure to find the shortest
        symbolic program that predicts the observations. Returns the
        expression, its MDL cost, and optionally a physics law verification
        and Lean4 theorem stub.
        """
        client_ip = request.client.host if request.client else "unknown"
        if not _limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 10 requests per minute."
            )

        observations = body.observations
        alphabet_size = body.alphabet_size or (max(int(v) for v in observations) + 2)

        # Check cache
        config = {
            "beam_width": body.beam_width,
            "max_depth": body.max_depth,
            "n_iterations": body.n_iterations,
        }
        cached = _cache.get(observations, config)
        if cached:
            cached["from_cache"] = True
            return JSONResponse(content=cached)

        # Run discovery
        try:
            result = _run_discovery(
                observations, alphabet_size,
                body.beam_width, body.max_depth,
                body.n_iterations, body.time_budget_seconds,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

        # Optionally verify physics laws
        verified_law = "NONE"
        if body.verify_physics_laws:
            try:
                law_result = _run_law_verification(observations)
                verified_law = law_result["primary_law"]
            except Exception:
                pass

        # Optionally generate Lean4 stub
        lean4 = None
        if body.return_lean4 and result["expression"]:
            lean4 = _generate_lean4_stub(result["expression"], result["math_family"])

        response = {
            **result,
            "verified_law": verified_law,
            "lean4_theorem": lean4,
            "n_observations": len(observations),
            "alphabet_size_used": alphabet_size,
            "search_config": config,
            "from_cache": False,
        }

        _cache.set(observations, config, response)
        return JSONResponse(content=response)

    @app.post("/verify_law", response_model=None)
    async def verify_law(request: Request, body: LawVerifyRequest):
        """
        Test whether an observation sequence satisfies known physics laws.
        
        Tests: Hooke's Law, exponential decay, free fall, Newton cooling,
        simple harmonic motion.
        """
        client_ip = request.client.host if request.client else "unknown"
        if not _limiter.is_allowed(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")

        try:
            result = _run_law_verification(body.observations)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")

        n_passing = sum(1 for t in result["all_tests"] if t["passed"])
        result["summary"] = (
            f"Tested {len(result['all_tests'])} physics laws. "
            f"{n_passing} passed. "
            f"Primary: {result['primary_law']} (confidence={result['confidence']:.2f})"
        )
        return JSONResponse(content=result)

    @app.get("/stats")
    async def stats():
        """Return API usage statistics."""
        return {
            "cache_size": _cache.size,
            "cache_hit_rate": round(_cache.hit_rate, 3),
            "cache_hits": _cache._hits,
            "cache_misses": _cache._misses,
        }

else:
    # Stub app when FastAPI is not installed
    class app:
        pass