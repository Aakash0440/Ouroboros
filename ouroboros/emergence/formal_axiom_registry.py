"""
Formal Axiom Registry — stores axioms with their verification status.

Extends KnowledgeBase with:
    - verification_method: 'lean4' | 'empirical' | 'unverified'
    - lean4_proof_term: The actual Lean4 proof term (if proved)
    - lean4_theorem_name: The theorem name in the formal library
    - formal_confidence: Separate from empirical confidence
        empirical: confidence from agent consensus
        lean4: confidence from formal proof (1.0 if proved, 0 if refuted)
        combined: 0.7 * formal + 0.3 * empirical

This registry is the bridge between OUROBOROS (empirical AI system)
and Lean4 (formal mathematics). Axioms that pass both are:
    1. Discovered empirically (compression pressure)
    2. Verified formally (Lean4 proof)
These are the strongest possible axioms — discovered AND proved.

Eventually: export these to Mathlib4 as machine-discovered lemmas.
"""

import json
import time
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from ouroboros.core.knowledge_base import KnowledgeBase, StoredAxiom
from ouroboros.proof_market.lean4_bridge import (
    VerificationResult, VerificationReport, Lean4Translator, Lean4Runner
)
from ouroboros.compression.program_synthesis import ExprNode
from ouroboros.utils.logger import get_logger


@dataclass
class FormalAxiom:
    """
    An axiom with full verification status.

    Combines empirical discovery data with formal verification status.
    """
    stored: StoredAxiom
    verification_method: str = 'unverified'  # lean4 | empirical | unverified
    lean4_proof_term: str = ''
    lean4_theorem_name: str = ''
    lean4_verified: bool = False
    empirical_verified: bool = False
    formal_confidence: float = 0.0

    @property
    def combined_confidence(self) -> float:
        if not self.lean4_verified and not self.empirical_verified:
            return 0.0
        formal = 1.0 if self.lean4_verified else 0.5
        return 0.7 * formal + 0.3 * self.stored.confidence

    @property
    def is_fully_verified(self) -> bool:
        return self.lean4_verified and self.empirical_verified

    def __repr__(self) -> str:
        v = "LEAN4+EMP" if self.is_fully_verified else \
            "LEAN4" if self.lean4_verified else \
            "EMP" if self.empirical_verified else "UNVERIFIED"
        return (f"FormalAxiom({self.stored.axiom_id}: "
                f"{self.stored.expression_str!r} "
                f"[{v}] "
                f"combined={self.combined_confidence:.3f})")


class FormalAxiomRegistry:
    """
    Registry combining KnowledgeBase with formal verification.

    Usage:
        registry = FormalAxiomRegistry('ouroboros_formal.db')
        registry.verify_and_register(axiom, stream, alphabet_size)
        best = registry.get_fully_verified_axioms()
    """

    FORMAL_TABLE = """
    CREATE TABLE IF NOT EXISTS formal_verification (
        axiom_id            INTEGER PRIMARY KEY,
        verification_method TEXT DEFAULT 'unverified',
        lean4_verified      INTEGER DEFAULT 0,
        empirical_verified  INTEGER DEFAULT 0,
        lean4_proof_term    TEXT DEFAULT '',
        lean4_theorem_name  TEXT DEFAULT '',
        formal_confidence   REAL DEFAULT 0.0,
        verified_at         REAL DEFAULT 0.0,
        lean4_output        TEXT DEFAULT ''
    )
    """

    def __init__(self, db_path: str = 'ouroboros_formal.db'):
        self.kb = KnowledgeBase(db_path)
        self.translator = Lean4Translator()
        self.runner = Lean4Runner()
        self.logger = get_logger('FormalAxiomRegistry')

        # Add formal verification table
        self.kb._conn.execute(self.FORMAL_TABLE)
        self.kb._conn.commit()

    def verify_and_register(
        self,
        axiom,  # ProtoAxiom
        test_stream: List[int],
        alphabet_size: int,
        environment_name: str
    ) -> Tuple[int, VerificationReport]:
        """
        Save axiom to KB and run formal verification.

        Returns:
            (axiom_id, verification_report)
        """
        # Save to KB
        axiom_id = self.kb.save_axiom(axiom, environment_name, alphabet_size)

        # Formal verification
        expr = axiom.expression
        short_stream = test_stream[:min(40, len(test_stream))]

        if self.runner.is_available():
            # Lean4 path
            script = self.translator.build_verification_script(
                expr, short_stream, alphabet_size
            )
            report = self.runner.run_script(script)
        else:
            # Empirical fallback
            preds = expr.predict_sequence(len(short_stream), alphabet_size)
            errors = sum(p != a for p, a in zip(preds, short_stream))
            if errors == 0:
                report = VerificationReport(
                    VerificationResult.EMPIRICAL_NONE, method='empirical'
                )
            else:
                report = VerificationReport(
                    VerificationResult.EMPIRICAL_CE,
                    method='empirical',
                    counterexample=f'{errors} errors'
                )

        # Save verification result
        lean4_verified = report.result == VerificationResult.PROVED
        empirical_verified = report.result == VerificationResult.EMPIRICAL_NONE
        formal_conf = 1.0 if lean4_verified else (0.8 if empirical_verified else 0.0)
        method = report.method

        self.kb._conn.execute("""
            INSERT OR REPLACE INTO formal_verification
            (axiom_id, verification_method, lean4_verified, empirical_verified,
             lean4_proof_term, formal_confidence, verified_at, lean4_output)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            axiom_id, method,
            int(lean4_verified), int(empirical_verified),
            report.proof_term,
            formal_conf, time.time(),
            report.lean4_output[:1000]
        ))
        self.kb._conn.commit()

        self.logger.info(
            f"Axiom {axiom_id} verified: method={method} "
            f"lean4={lean4_verified} empirical={empirical_verified} "
            f"conf={formal_conf:.2f}"
        )

        return axiom_id, report

    def get_fully_verified_axioms(
        self,
        require_lean4: bool = False
    ) -> List[FormalAxiom]:
        """
        Return axioms that passed verification.

        Args:
            require_lean4: If True, only return Lean4-proved axioms.
                          If False, include empirically verified too.
        """
        if require_lean4:
            where = "fv.lean4_verified = 1"
        else:
            where = "(fv.lean4_verified = 1 OR fv.empirical_verified = 1)"

        rows = self.kb._conn.execute(f"""
            SELECT a.*, fv.verification_method, fv.lean4_verified,
                   fv.empirical_verified, fv.lean4_proof_term,
                   fv.formal_confidence
            FROM axioms a
            JOIN formal_verification fv ON a.axiom_id = fv.axiom_id
            WHERE {where}
            ORDER BY fv.formal_confidence DESC, a.confidence DESC
        """).fetchall()

        result = []
        for row in rows:
            stored = self.kb._row_to_stored_axiom(row)
            formal = FormalAxiom(
                stored=stored,
                verification_method=row['verification_method'],
                lean4_verified=bool(row['lean4_verified']),
                empirical_verified=bool(row['empirical_verified']),
                lean4_proof_term=row['lean4_proof_term'] or '',
                formal_confidence=row['formal_confidence'],
            )
            result.append(formal)
        return result

    def export_lean4_library(self, output_path: str) -> None:
        """
        Export all formally verified axioms as a Lean4 library file.

        The exported file can be imported into Lean4 projects and
        eventually contributed to Mathlib4 as machine-discovered lemmas.
        """
        axioms = self.get_fully_verified_axioms(require_lean4=False)

        lines = [
            "-- Machine-discovered mathematical axioms",
            "-- Generated by OUROBOROS — DO NOT EDIT BY HAND",
            "-- Source: compression-pressure MDL agent society",
            "",
            "import Mathlib.Data.Nat.Basic",
            "import Mathlib.Tactic",
            "",
            "namespace OUROBOROS.Discovered",
            "",
        ]

        for i, ax in enumerate(axioms):
            lines += [
                f"-- Axiom {ax.stored.axiom_id}: {ax.stored.expression_str}",
                f"-- Environment: {ax.stored.environment_name}",
                f"-- Confidence: {ax.combined_confidence:.3f}",
                f"-- Verified by: {ax.verification_method}",
                f"-- Times confirmed: {ax.stored.times_confirmed}",
                f"",
                f"def axiom_{ax.stored.axiom_id}_fn (t : Nat) : Nat :=",
                f"  -- {ax.stored.expression_str}",
                f"  sorry -- Formal expression to be filled by Lean4Translator",
                f"",
            ]

        lines += ["end OUROBOROS.Discovered"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"Exported {len(axioms)} axioms to {output_path}")

    def registry_summary(self) -> str:
        stats = self.kb.statistics()
        lean4_count = self.kb._conn.execute(
            "SELECT COUNT(*) FROM formal_verification WHERE lean4_verified=1"
        ).fetchone()[0]
        empirical_count = self.kb._conn.execute(
            "SELECT COUNT(*) FROM formal_verification WHERE empirical_verified=1"
        ).fetchone()[0]
        lines = [
            "FormalAxiomRegistry:",
            f"  Total axioms:           {stats['total_axioms']}",
            f"  Lean4-proved:           {lean4_count}",
            f"  Empirically-verified:   {empirical_count}",
            f"  Total runs:             {stats['total_runs']}",
            f"  Lean4 available:        {self.runner.is_available()}",
        ]
        return '\n'.join(lines)