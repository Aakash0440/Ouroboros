# ouroboros/compression/__init__.py
from ouroboros.compression.mdl import (
    entropy_bits,
    naive_bits,
    compression_ratio,
    MDLCost,
)
__all__ = ['entropy_bits', 'naive_bits', 'compression_ratio', 'MDLCost']