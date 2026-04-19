"""
KnowledgeBase — persistent SQLite store for discovered axioms.

Schema:
    axioms:          All promoted proto-axioms across all runs
    environments:    Environment registry (parameters + alphabet size)
    run_history:     Log of all experimental runs
    expression_cache: Fingerprint → expression string cache

Usage:
    kb = KnowledgeBase('ouroboros_knowledge.db')

    # Save a discovered axiom
    kb.save_axiom(axiom, environment_name='ModArith(7,3,1)')

    # Load all high-confidence axioms for an environment
    priors = kb.load_priors_for_environment('ModArith(7,3,1)')

    # Get prior expressions to seed beam search
    seed_exprs = kb.get_seed_expressions(alphabet_size=7)

    # Save a complete run's results
    kb.save_run(run_id, environment, results_dict)

Why SQLite?
    - No server needed (file-based)
    - Built into Python stdlib
    - ACID transactions (no corruption on crash)
    - Fast enough for thousands of axioms
    - Human-readable with any SQLite viewer
"""

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ouroboros.compression.program_synthesis import ExprNode, BeamSearchSynthesizer, C
from ouroboros.emergence.proto_axiom_pool import ProtoAxiom
from ouroboros.utils.logger import get_logger


@dataclass
class StoredAxiom:
    """
    An axiom as stored in the database.

    Fields:
        axiom_id: Database primary key (auto-increment)
        expression_str: Human-readable expression string
        fingerprint_hash: SHA-256 of fingerprint tuple (for fast dedup)
        confidence: [0,1] confidence score
        compression_ratio: MDL ratio at discovery
        environment_name: Which environment it came from
        alphabet_size: Symbol alphabet size
        times_confirmed: How many runs have re-discovered this
        times_survived_market: Proof market survival count
        first_seen: Unix timestamp of first discovery
        last_seen: Unix timestamp of most recent confirmation
    """
    axiom_id: int
    expression_str: str
    fingerprint_hash: str
    confidence: float
    compression_ratio: float
    environment_name: str
    alphabet_size: int
    times_confirmed: int = 1
    times_survived_market: int = 0
    first_seen: float = 0.0
    last_seen: float = 0.0

    def to_expr_node(self) -> Optional[ExprNode]:
        """Parse expression string back to ExprNode (best-effort)."""
        # We use beam search on a dummy sequence to regenerate the expression.
        # In practice, store the expression bytes and use a proper deserializer.
        # For now: return None and let caller handle.
        return None  # TODO: implement proper deserializer in Day 15+

    def __repr__(self) -> str:
        return (f"StoredAxiom({self.axiom_id}: "
                f"{self.expression_str!r} "
                f"conf={self.confidence:.3f} "
                f"confirmed={self.times_confirmed}x)")


class KnowledgeBase:
    """
    Persistent axiom knowledge base backed by SQLite.

    Args:
        db_path: Path to SQLite database file
                 Default: 'ouroboros_knowledge.db' in project root
    """

    CREATE_AXIOMS = """
    CREATE TABLE IF NOT EXISTS axioms (
        axiom_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        expression_str      TEXT NOT NULL,
        expression_bytes    BLOB,
        fingerprint_hash    TEXT NOT NULL UNIQUE,
        fingerprint_json    TEXT,
        confidence          REAL NOT NULL,
        compression_ratio   REAL NOT NULL,
        environment_name    TEXT NOT NULL,
        alphabet_size       INTEGER NOT NULL,
        times_confirmed     INTEGER DEFAULT 1,
        times_survived      INTEGER DEFAULT 0,
        times_rejected      INTEGER DEFAULT 0,
        first_seen          REAL NOT NULL,
        last_seen           REAL NOT NULL,
        notes               TEXT DEFAULT ''
    )
    """

    CREATE_RUNS = """
    CREATE TABLE IF NOT EXISTS run_history (
        run_id          TEXT PRIMARY KEY,
        environment     TEXT NOT NULL,
        num_agents      INTEGER,
        stream_length   INTEGER,
        best_ratio      REAL,
        axioms_found    INTEGER DEFAULT 0,
        phase           INTEGER DEFAULT 1,
        started_at      REAL NOT NULL,
        elapsed_seconds REAL,
        config_json     TEXT,
        results_json    TEXT
    )
    """

    CREATE_PRIORS = """
    CREATE TABLE IF NOT EXISTS search_priors (
        prior_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        expression_str  TEXT NOT NULL,
        alphabet_size   INTEGER NOT NULL,
        source_axiom_id INTEGER,
        priority        REAL DEFAULT 1.0,
        created_at      REAL NOT NULL
    )
    """

    def __init__(self, db_path: str = 'ouroboros_knowledge.db'):
        self.db_path = db_path
        self.logger = get_logger('KnowledgeBase')
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self.logger.info(f"KnowledgeBase opened: {db_path}")

    def _create_tables(self) -> None:
        c = self._conn.cursor()
        c.execute(self.CREATE_AXIOMS)
        c.execute(self.CREATE_RUNS)
        c.execute(self.CREATE_PRIORS)
        self._conn.commit()

    def _fingerprint_hash(self, fingerprint: Tuple) -> str:
        """SHA-256 hash of a fingerprint tuple for fast deduplication."""
        fp_str = json.dumps(list(fingerprint), separators=(',', ':'))
        return hashlib.sha256(fp_str.encode()).hexdigest()

    # ─── Axiom storage ───────────────────────────────────────────────────

    def save_axiom(
        self,
        axiom: ProtoAxiom,
        environment_name: str,
        alphabet_size: int
    ) -> int:
        """
        Save a proto-axiom to the knowledge base.

        If the same fingerprint already exists, increment times_confirmed
        and update confidence/compression_ratio if improved.

        Returns:
            axiom_id (database primary key)
        """
        fp_hash = self._fingerprint_hash(axiom.fingerprint)
        now = time.time()

        c = self._conn.cursor()
        existing = c.execute(
            "SELECT axiom_id, confidence, compression_ratio, times_confirmed "
            "FROM axioms WHERE fingerprint_hash = ?",
            (fp_hash,)
        ).fetchone()

        if existing:
            # Update existing — increment confirmed count
            new_conf = max(existing['confidence'], axiom.confidence)
            new_ratio = min(existing['compression_ratio'], axiom.compression_ratio)
            new_confirmed = existing['times_confirmed'] + 1

            c.execute("""
                UPDATE axioms
                SET confidence=?, compression_ratio=?, times_confirmed=?, last_seen=?
                WHERE axiom_id=?
            """, (new_conf, new_ratio, new_confirmed, now, existing['axiom_id']))
            self._conn.commit()

            self.logger.info(
                f"Axiom {existing['axiom_id']} confirmed again "
                f"({new_confirmed}x): {axiom.expression.to_string()!r}"
            )
            return existing['axiom_id']

        else:
            # Insert new axiom
            c.execute("""
                INSERT INTO axioms
                (expression_str, expression_bytes, fingerprint_hash, fingerprint_json,
                 confidence, compression_ratio, environment_name, alphabet_size,
                 times_confirmed, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (
                axiom.expression.to_string(),
                axiom.expression.to_bytes(),
                fp_hash,
                json.dumps(list(axiom.fingerprint)),
                axiom.confidence,
                axiom.compression_ratio,
                environment_name,
                alphabet_size,
                now, now
            ))
            self._conn.commit()
            new_id = c.lastrowid

            self.logger.info(
                f"New axiom saved (id={new_id}): "
                f"{axiom.expression.to_string()!r}  "
                f"conf={axiom.confidence:.3f}  "
                f"env={environment_name}"
            )
            return new_id

    def record_market_result(
        self,
        axiom_id: int,
        survived: bool
    ) -> None:
        """Update proof market survival count for an axiom."""
        col = 'times_survived' if survived else 'times_rejected'
        self._conn.execute(
            f"UPDATE axioms SET {col} = {col} + 1 WHERE axiom_id = ?",
            (axiom_id,)
        )
        self._conn.commit()

    # ─── Loading priors ──────────────────────────────────────────────────

    def load_axioms_for_environment(
        self,
        environment_name: str,
        min_confidence: float = 0.30,
        min_confirmed: int = 1,
        limit: int = 20
    ) -> List[StoredAxiom]:
        """
        Load all high-quality axioms for a specific environment.

        Args:
            environment_name: Must match exactly what was saved
            min_confidence: Minimum confidence threshold
            min_confirmed: Minimum times confirmed across runs
            limit: Maximum axioms to return

        Returns:
            List of StoredAxiom sorted by confidence (highest first)
        """
        rows = self._conn.execute("""
            SELECT * FROM axioms
            WHERE environment_name = ?
              AND confidence >= ?
              AND times_confirmed >= ?
            ORDER BY confidence DESC
            LIMIT ?
        """, (environment_name, min_confidence, min_confirmed, limit)).fetchall()

        return [self._row_to_stored_axiom(r) for r in rows]

    def load_all_axioms(
        self,
        min_confidence: float = 0.20,
        limit: int = 100
    ) -> List[StoredAxiom]:
        """Load all axioms above confidence threshold, any environment."""
        rows = self._conn.execute("""
            SELECT * FROM axioms
            WHERE confidence >= ?
            ORDER BY confidence DESC, times_confirmed DESC
            LIMIT ?
        """, (min_confidence, limit)).fetchall()
        return [self._row_to_stored_axiom(r) for r in rows]

    def get_seed_expressions_for_search(
        self,
        alphabet_size: int,
        top_k: int = 10
    ) -> List[str]:
        """
        Return expression strings that can seed a new beam search.

        These are the high-confidence expressions from previous runs
        at the same alphabet size. An agent starting a new run can
        evaluate these expressions first before searching from scratch.

        Args:
            alphabet_size: Symbol alphabet of the new run
            top_k: How many seeds to return

        Returns:
            List of expression strings (best first)
        """
        rows = self._conn.execute("""
            SELECT expression_str, confidence, times_confirmed
            FROM axioms
            WHERE alphabet_size = ?
            ORDER BY confidence * times_confirmed DESC
            LIMIT ?
        """, (alphabet_size, top_k)).fetchall()
        return [r['expression_str'] for r in rows]

    # ─── Run history ─────────────────────────────────────────────────────

    def save_run(
        self,
        run_id: str,
        environment: str,
        results: dict,
        config: dict = None,
        phase: int = 1
    ) -> None:
        """Save a complete experimental run to history."""
        self._conn.execute("""
            INSERT OR REPLACE INTO run_history
            (run_id, environment, num_agents, stream_length,
             best_ratio, axioms_found, phase, started_at,
             elapsed_seconds, config_json, results_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            environment,
            results.get('num_agents', 0),
            results.get('stream_length', 0),
            results.get('best_ratio', 1.0),
            len(results.get('axioms_promoted', [])),
            phase,
            time.time(),
            results.get('elapsed_seconds', 0),
            json.dumps(config or {}),
            json.dumps(results, default=str)
        ))
        self._conn.commit()

    # ─── Statistics ──────────────────────────────────────────────────────

    def statistics(self) -> dict:
        """Return summary statistics of the knowledge base."""
        c = self._conn.cursor()
        total = c.execute("SELECT COUNT(*) FROM axioms").fetchone()[0]
        high_conf = c.execute(
            "SELECT COUNT(*) FROM axioms WHERE confidence >= 0.6"
        ).fetchone()[0]
        multi_confirmed = c.execute(
            "SELECT COUNT(*) FROM axioms WHERE times_confirmed >= 3"
        ).fetchone()[0]
        runs = c.execute("SELECT COUNT(*) FROM run_history").fetchone()[0]
        envs = c.execute(
            "SELECT COUNT(DISTINCT environment_name) FROM axioms"
        ).fetchone()[0]
        return {
            'total_axioms': total,
            'high_confidence_axioms': high_conf,
            'multi_confirmed_axioms': multi_confirmed,
            'total_runs': runs,
            'distinct_environments': envs,
        }

    def summary(self) -> str:
        stats = self.statistics()
        lines = [
            f"KnowledgeBase: {self.db_path}",
            f"  Total axioms:         {stats['total_axioms']}",
            f"  High confidence:      {stats['high_confidence_axioms']}",
            f"  Multi-confirmed:      {stats['multi_confirmed_axioms']}",
            f"  Total runs logged:    {stats['total_runs']}",
            f"  Distinct environments:{stats['distinct_environments']}",
        ]
        return '\n'.join(lines)

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _row_to_stored_axiom(self, row: sqlite3.Row) -> StoredAxiom:
        return StoredAxiom(
            axiom_id=row['axiom_id'],
            expression_str=row['expression_str'],
            fingerprint_hash=row['fingerprint_hash'],
            confidence=row['confidence'],
            compression_ratio=row['compression_ratio'],
            environment_name=row['environment_name'],
            alphabet_size=row['alphabet_size'],
            times_confirmed=row['times_confirmed'],
            times_survived_market=row['times_survived'],
            first_seen=row['first_seen'],
            last_seen=row['last_seen'],
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def save_axiom_from_dict(
        self,
        ax_dict: dict,
        environment_name: str,
        alphabet_size: int
    ) -> int:
        """Save an axiom from a results dict (no ProtoAxiom object needed)."""
        expr_str = ax_dict['expression']
        confidence = ax_dict.get('confidence', 0.5)
        compression_ratio = ax_dict.get('compression_ratio', 1.0)
        
        # Use expression string as fingerprint (good enough for dedup)
        fp_hash = hashlib.sha256(expr_str.encode()).hexdigest()
        now = time.time()

        c = self._conn.cursor()
        existing = c.execute(
            "SELECT axiom_id, confidence, times_confirmed FROM axioms "
            "WHERE fingerprint_hash = ?", (fp_hash,)
        ).fetchone()

        if existing:
            new_conf = max(existing['confidence'], confidence)
            new_confirmed = existing['times_confirmed'] + 1
            c.execute("""
                UPDATE axioms SET confidence=?, times_confirmed=?, last_seen=?
                WHERE axiom_id=?
            """, (new_conf, new_confirmed, now, existing['axiom_id']))
            self._conn.commit()
            return existing['axiom_id']
        else:
            c.execute("""
                INSERT INTO axioms
                (expression_str, fingerprint_hash, confidence, compression_ratio,
                environment_name, alphabet_size, times_confirmed, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            """, (expr_str, fp_hash, confidence, compression_ratio,
                environment_name, alphabet_size, now, now))
            self._conn.commit()
        return c.lastrowid