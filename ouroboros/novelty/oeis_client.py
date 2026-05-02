"""
OEISClient — Query the On-Line Encyclopedia of Integer Sequences.

The OEIS has ~370,000 integer sequences with formulas, references, and
cross-links. For OUROBOROS, it is the primary source for:
  1. Checking if a discovered integer sequence is already known
  2. Getting the formula name if it is known
  3. Finding related sequences to seed the registry

OEIS API:
  https://oeis.org/search?q=[search_terms]&fmt=json
  https://oeis.org/A000045/b045.txt  — first N terms of sequence A000045

Rate limiting:
  OEIS requests are generous but we cache aggressively to be respectful.
  Cache: SQLite database, keyed by (sequence_prefix_hash, search_terms).
  TTL: 30 days (OEIS sequences rarely change).

Usage:
  client = OEISClient(cache_path="results/oeis_cache.db")
  result = client.search_sequence([1, 1, 2, 3, 5, 8, 13, 21, 34])
  # result.oeis_id = "A000045"
  # result.name = "Fibonacci numbers"
  # result.formula = "a(n) = a(n-1) + a(n-2)"
"""

from __future__ import annotations
import hashlib
import json
import sqlite3
import time
import urllib.request
import urllib.parse
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


OEIS_API_BASE = "https://oeis.org/search"
OEIS_SEQUENCE_BASE = "https://oeis.org"
REQUEST_DELAY_SECONDS = 0.5   # be polite to OEIS
CACHE_TTL_DAYS = 30


@dataclass
class OEISResult:
    """Result from an OEIS query."""
    found: bool
    oeis_id: Optional[str]      # e.g. "A000045"
    name: Optional[str]          # e.g. "Fibonacci numbers"
    description: Optional[str]
    formula: Optional[str]       # best formula from OEIS
    example_values: List[int]    # first terms of the sequence
    offset: int                  # which index the sequence starts at
    keywords: List[str]          # OEIS keywords: "nonn", "easy", "core", etc.
    references: List[str]        # citations
    from_cache: bool

    @property
    def is_well_known(self) -> bool:
        return self.found and "core" in self.keywords

    @property
    def is_finite(self) -> bool:
        return self.found and "fini" in self.keywords

    def description_str(self) -> str:
        if not self.found:
            return "Not found in OEIS"
        parts = [f"OEIS {self.oeis_id}: {self.name}"]
        if self.formula:
            parts.append(f"Formula: {self.formula}")
        if self.keywords:
            parts.append(f"Keywords: {', '.join(self.keywords[:5])}")
        return "\n".join(parts)


class OEISCache:
    """SQLite-backed cache for OEIS queries."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS oeis_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.commit()

    def get(self, key: str) -> Optional[dict]:
        ttl_seconds = CACHE_TTL_DAYS * 86400
        cutoff = time.time() - ttl_seconds
        row = self._conn.execute(
            "SELECT value FROM oeis_cache WHERE key=? AND timestamp>?",
            (key, cutoff)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO oeis_cache VALUES (?,?,?)",
            (key, json.dumps(value), time.time())
        )
        self._conn.commit()

    def key_for_sequence(self, terms: List[int]) -> str:
        prefix = ",".join(str(t) for t in terms[:8])
        return hashlib.md5(prefix.encode()).hexdigest()

    def key_for_search(self, query: str) -> str:
        return hashlib.md5(query.encode()).hexdigest()


class OEISClient:
    """
    Client for the OEIS API with local caching.

    All queries are cached for 30 days. Network is only hit on cache misses.
    Falls back gracefully if the network is unavailable.
    """

    def __init__(
        self,
        cache_path: str = "results/oeis_cache.db",
        timeout_seconds: float = 10.0,
        verbose: bool = False,
    ):
        self._cache = OEISCache(cache_path)
        self._timeout = timeout_seconds
        self._verbose = verbose
        self._last_request_time = 0.0
        self._n_requests = 0
        self._n_cache_hits = 0

    def _rate_limit(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SECONDS:
            time.sleep(REQUEST_DELAY_SECONDS - elapsed)
        self._last_request_time = time.time()

    def _fetch(self, url: str) -> Optional[dict]:
        """Fetch URL, return parsed JSON or None on failure."""
        self._rate_limit()
        self._n_requests += 1
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "OUROBOROS-Research/1.0 (mathematical-discovery-system)"}
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            if self._verbose:
                print(f"  OEIS fetch failed: {e}")
            return None

    def search_sequence(
        self,
        terms: List[int],
        max_terms_for_search: int = 8,
    ) -> OEISResult:
        """
        Search OEIS for a sequence matching the given terms.
        Returns the best match if found, otherwise OEISResult(found=False).
        """
        if len(terms) < 3:
            return OEISResult(False, None, None, None, None, [], 0, [], [], False)

        search_terms = terms[:max_terms_for_search]
        cache_key = self._cache.key_for_sequence(search_terms)

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._n_cache_hits += 1
            return self._parse_oeis_response(cached, from_cache=True)

        # Build search query: comma-separated terms
        query = ",".join(str(t) for t in search_terms)
        url = f"{OEIS_API_BASE}?q={urllib.parse.quote(query)}&fmt=json&start=0"

        data = self._fetch(url)
        if data is None:
            return OEISResult(False, None, None, None, None, [], 0, [], [], False)

        self._cache.set(cache_key, data)
        return self._parse_oeis_response(data, from_cache=False)

    def lookup_id(self, oeis_id: str) -> OEISResult:
        """Look up a specific OEIS sequence by ID (e.g., 'A000045')."""
        cache_key = f"id:{oeis_id}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._n_cache_hits += 1
            return self._parse_oeis_response(cached, from_cache=True)

        url = f"{OEIS_API_BASE}?q=id:{oeis_id}&fmt=json"
        data = self._fetch(url)
        if data is None:
            return OEISResult(False, None, None, None, None, [], 0, [], [], False)

        self._cache.set(cache_key, data)
        return self._parse_oeis_response(data, from_cache=False)

    def _parse_oeis_response(self, data: dict, from_cache: bool) -> OEISResult:
        """Parse OEIS API JSON response into OEISResult."""
        results = data.get("results")
        if not results:
            return OEISResult(False, None, None, None, None, [], 0, [], [], from_cache)

        # Take the first (best) result
        r = results[0]
        oeis_id = f"A{r.get('number', 0):06d}"
        name = r.get("name", "")

        # Extract formula
        formula_list = r.get("formula", [])
        formula = formula_list[0] if formula_list else None

        # Extract example values
        values_str = r.get("data", "")
        try:
            example_values = [int(v) for v in values_str.split(",") if v.strip()]
        except ValueError:
            example_values = []

        keywords = r.get("keyword", "").split(",")
        references = r.get("reference", [])[:3]
        offset_str = r.get("offset", "0,1").split(",")[0]
        try:
            offset = int(offset_str)
        except ValueError:
            offset = 0

        return OEISResult(
            found=True,
            oeis_id=oeis_id,
            name=name,
            description=r.get("comment", [""])[0] if r.get("comment") else None,
            formula=formula,
            example_values=example_values[:20],
            offset=offset,
            keywords=keywords,
            references=references,
            from_cache=from_cache,
        )

    @property
    def cache_hit_rate(self) -> float:
        total = self._n_requests + self._n_cache_hits
        return self._n_cache_hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "n_api_requests": self._n_requests,
            "n_cache_hits": self._n_cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
        }