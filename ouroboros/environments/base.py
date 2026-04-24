"""
Observation environment base class.

An environment produces a stream of integer symbols ∈ {0..alphabet_size-1}.
Agents observe the stream and try to find programs that predict it.

Why integer sequences?
- Compression is measured in bits — discrete alphabets make this clean
- Kolmogorov complexity applies to discrete strings
- Arithmetic structure (mod, +, *) maps naturally to integer sequences
- If modular arithmetic EMERGES, it emerges in the integer domain

Environment contract:
    env.reset(length)        → generate a fresh stream of `length` symbols
    env.observe(n)           → return next n symbols, advance position
    env.peek_all()           → return all symbols without advancing
    env.naive_description_length() → bits needed with no compression
"""

from abc import ABC
from typing import List, Optional
import numpy as np


class ObservationEnvironment(ABC):
    """
    Abstract base for all OUROBOROS environments.

    Subclasses implement _generate_stream() to define the
    underlying mathematical structure.

    Args:
        alphabet_size: Number of distinct symbols (e.g. 7 for mod-7 env)
        seed: Random seed for reproducible streams
        name: Human-readable name for the environment (defaults to class name)
    """

    def __init__(self, alphabet_size: int = 256, seed: int = 42, name: Optional[str] = None):
        if not isinstance(getattr(type(self), 'alphabet_size', None), property):
            if not isinstance(getattr(type(self), 'alphabet_size', None), property):
                self.alphabet_size = alphabet_size
        self.name = name or self.__class__.__name__
        self.rng = np.random.default_rng(seed)
        self._stream: List[int] = []
        self._position: int = 0

    def generate(self, length: int, start: int = 0) -> List[int]:
        """Convenience method: generate a stream without affecting internal state."""
        backup_rng = np.random.default_rng(int(self.rng.integers(0, 2**32)))
        backup_stream = self._stream
        backup_pos = self._position
        self._stream = self._generate_stream(length)
        result = list(self._stream)
        self._stream = backup_stream
        self._position = backup_pos
        return result

    def _generate_stream(self, length: int = 1000) -> List[int]:
        """Generate observation stream of given length."""
        return []

    def reset(self, stream_length: int = 10_000) -> None:
        """
        Generate a fresh stream and reset read position to 0.
        Call this at the start of every episode.
        """
        self._stream = self._generate_stream(stream_length)
        self._position = 0

    def observe(self, n: int = 1) -> List[int]:
        """
        Return next n symbols, advancing position.
        Returns fewer than n symbols if stream is exhausted.
        """
        start = self._position
        end = min(self._position + n, len(self._stream))
        self._position = end
        return self._stream[start:end]

    def peek_all(self) -> List[int]:
        """
        Return entire stream without advancing position.
        Used by agents to load their full observation history.
        """
        return list(self._stream)

    def peek(self, n: int) -> List[int]:
        """Return next n symbols WITHOUT advancing position."""
        end = min(self._position + n, len(self._stream))
        return self._stream[self._position:end]

    @property
    def position(self) -> int:
        """Current read position."""
        return self._position

    @property
    def length(self) -> int:
        """Total stream length."""
        return len(self._stream)

    @property
    def remaining(self) -> int:
        """Symbols remaining to observe."""
        return max(0, self.length - self._position)

    @property
    def exhausted(self) -> bool:
        """True if all symbols have been observed."""
        return self._position >= len(self._stream)

    def naive_description_length(self) -> float:
        """
        Bits needed if we assume uniform distribution over alphabet.
        = len(stream) * log2(alphabet_size)

        This is the WORST CASE — no compression at all.
        Any agent that does better than this has found structure.
        """
        import math
        return len(self._stream) * math.log2(self.alphabet_size)

    def __len__(self) -> int:
        return len(self._stream)

    def __repr__(self) -> str:
        return (
            f"{self.name}("
            f"alphabet={self.alphabet_size}, "
            f"len={len(self._stream)}, "
            f"pos={self._position})"
        )


# Alias for backwards compatibility
Environment = ObservationEnvironment