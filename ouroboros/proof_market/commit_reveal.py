"""
Cryptographic commit-reveal protocol for the OUROBOROS proof market.

The collusion problem:
    In a naive adversarial market, agents can coordinate off-market.
    Agent A proposes a bad modification → tells friends B,C,D to approve
    → no real verification happens → bad modifications get through.

The solution: commit-reveal (standard in cryptographic auction theory).

    COMMIT phase:
        Each agent computes their counterexample attempt.
        They hash it: H = SHA256(counterexample_bytes || salt)
        They submit ONLY the hash. Cannot see others' hashes.
        Salt is a random 256-bit nonce — prevents preimage attacks.

    REVEAL phase:
        All agents publish (counterexample_bytes, salt).
        Anyone can verify: SHA256(counterexample_bytes || salt) == H

    VERIFY phase:
        Valid reveals are checked for counterexample quality.
        Invalid reveals (hash mismatch) → agent penalized.

Why collusion is now impossible:
    - Agents must commit BEFORE seeing others' work.
    - Changing answer after seeing others = hash won't match = penalty.
    - To collude, agents would need to agree on the hash in advance,
      but they can't know the hash without knowing the counterexample,
      which they can't know before independently searching.

This module implements:
    make_commitment(agent_id, counterexample, round_id) → Commitment
    reveal_commitment(commitment) → (counterexample_bytes, salt)
    verify_reveal(commitment) → bool
    PublicCommitmentRecord — what others can see (hash only, no CE)
"""

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class Commitment:
    """
    An agent's cryptographic commitment to a counterexample.

    Private fields (agent keeps secret until reveal):
        counterexample: The actual counterexample bytes
        salt: Random nonce used in hash

    Public fields (visible to all during COMMIT phase):
        agent_id, commitment_hash, round_id, committed_at

    After REVEAL phase:
        revealed = True
        valid = True/False (set by verify_reveal)
    """
    # Public (visible immediately)
    agent_id: int
    commitment_hash: str        # SHA-256 hex — 64 chars
    round_id: str
    committed_at: float = field(default_factory=time.time)

    # Private (revealed in REVEAL phase)
    counterexample: Optional[bytes] = None
    salt: Optional[str] = None

    # Set after verification
    revealed: bool = False
    valid: bool = False

    def public_view(self) -> dict:
        """What other agents can see during COMMIT phase."""
        return {
            'agent_id': self.agent_id,
            'commitment_hash': self.commitment_hash,
            'round_id': self.round_id,
            'committed_at': self.committed_at,
            'revealed': self.revealed,
        }

    def __repr__(self) -> str:
        status = "revealed+valid" if (self.revealed and self.valid) else \
                 "revealed+invalid" if self.revealed else "committed"
        return (f"Commitment(agent={self.agent_id}, "
                f"hash={self.commitment_hash[:8]}..., "
                f"status={status})")


def make_commitment(
    agent_id: int,
    counterexample: bytes,
    round_id: str
) -> Commitment:
    """
    Create a cryptographic commitment to a counterexample.

    The agent commits to their counterexample without revealing it.
    The random salt prevents:
        1. Preimage attacks (others can't reverse-engineer CE from hash)
        2. Rainbow table attacks (same CE gives different hash each time)

    Security guarantee:
        Given only the commitment_hash, it is computationally infeasible
        to recover counterexample or find a different CE with the same hash.

    Args:
        agent_id: Agent making the commitment
        counterexample: The counterexample bytes (KEPT SECRET until reveal)
        round_id: Proof market round identifier

    Returns:
        Commitment object — agent holds this privately.
        Only commitment_hash is shared with others.
    """
    # 256-bit random salt — cryptographically secure
    salt = secrets.token_hex(32)

    # H = SHA-256(counterexample || salt)
    h = hashlib.sha256()
    h.update(counterexample)
    h.update(salt.encode('utf-8'))
    commitment_hash = h.hexdigest()

    return Commitment(
        agent_id=agent_id,
        commitment_hash=commitment_hash,
        round_id=round_id,
        counterexample=counterexample,
        salt=salt,
    )


def make_null_commitment(agent_id: int, round_id: str) -> Commitment:
    """
    Commitment for "I found no counterexample."

    Agents who genuinely find no counterexample still must commit —
    otherwise silence could indicate they're waiting to see others' work.
    The null commitment uses a fixed sentinel bytes value.
    """
    return make_commitment(agent_id, b'__NULL_NO_COUNTEREXAMPLE__', round_id)


def reveal_commitment(commitment: Commitment) -> Tuple[bytes, str]:
    """
    Reveal phase: publish the counterexample and salt.

    After calling this, the commitment is public.
    Others can verify using verify_reveal().

    Returns:
        (counterexample_bytes, salt_hex)
    """
    if commitment.counterexample is None:
        raise ValueError(f"Agent {commitment.agent_id}: no counterexample to reveal")
    commitment.revealed = True
    return commitment.counterexample, commitment.salt


def verify_reveal(commitment: Commitment) -> bool:
    """
    Verify that a revealed counterexample matches its commitment hash.

    Called during VERIFY phase after all reveals are in.

    Returns True if SHA-256(counterexample || salt) == commitment_hash.
    Sets commitment.valid accordingly.

    A False result means either:
        1. The agent tampered with their answer after seeing others'
        2. Data corruption
        3. Implementation bug on the agent's side
    All three result in the same penalty.
    """
    if not commitment.revealed:
        return False
    if commitment.counterexample is None or commitment.salt is None:
        return False

    h = hashlib.sha256()
    h.update(commitment.counterexample)
    h.update(commitment.salt.encode('utf-8'))
    expected = h.hexdigest()

    is_valid = (expected == commitment.commitment_hash)
    commitment.valid = is_valid
    return is_valid


def is_null_commitment(commitment: Commitment) -> bool:
    """Check if this is a null (no counterexample found) commitment."""
    if commitment.counterexample is None:
        return True
    return commitment.counterexample == b'__NULL_NO_COUNTEREXAMPLE__'


@dataclass
class RoundState:
    """
    Tracks the state of one proof market round.

    Lifecycle: OPEN → COMMIT → REVEAL → VERIFY → RESOLVED

    OPEN:     Proposal accepted, commit window starts
    COMMIT:   Agents submit commitment hashes (deadline: commit_deadline)
    REVEAL:   Agents publish counterexamples (deadline: reveal_deadline)
    VERIFY:   Reveals validated, adjudication run
    RESOLVED: Final outcome recorded
    """
    round_id: str
    proposer_id: int
    proposal_bytes: bytes
    proposal_description: str

    # Timing
    phase: str = 'OPEN'
    created_at: float = field(default_factory=time.time)

    # Commitments: agent_id → Commitment
    commitments: dict = field(default_factory=dict)

    # Results
    counterexamples_verified: int = 0
    modification_approved: bool = False
    bounty_distributions: dict = field(default_factory=dict)
    resolution_notes: str = ''

    def advance_to_commit(self) -> None:
        self.phase = 'COMMIT'

    def advance_to_reveal(self) -> None:
        self.phase = 'REVEAL'

    def advance_to_verify(self) -> None:
        self.phase = 'VERIFY'

    def resolve(self, approved: bool, notes: str = '') -> None:
        self.modification_approved = approved
        self.resolution_notes = notes
        self.phase = 'RESOLVED'

    def public_commitments(self) -> dict:
        """Only hashes visible during COMMIT phase."""
        return {aid: c.public_view()
                for aid, c in self.commitments.items()}

    def __repr__(self) -> str:
        return (f"RoundState(id={self.round_id}, "
                f"phase={self.phase}, "
                f"proposer={self.proposer_id}, "
                f"commits={len(self.commitments)})")