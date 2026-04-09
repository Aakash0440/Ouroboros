# ouroboros/utils/logger.py

"""
Structured logging for OUROBOROS.

Two components:
1. get_logger(name) → Rich-formatted console logger
2. MetricsWriter     → JSON-lines file writer for metrics

Every experiment writes to its own timestamped run directory.
Metrics are JSONL for easy pandas/analysis later.

Usage:
    logger = get_logger('Phase1')
    logger.info("Starting compression loop")
    logger.debug("Agent 3: ratio=0.045")

    writer = MetricsWriter('experiments/phase1/runs/run_001')
    writer.write(step=100, agent_id=3, compression_ratio=0.045)
    writer.close()
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.logging import RichHandler
from rich.console import Console

console = Console()


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a Rich-formatted logger.

    Args:
        name: Logger name (shown in output)
        level: Logging level (default INFO)

    Returns:
        Configured logger with Rich formatting
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
        )
        handler.setLevel(level)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger
    logger.propagate = False
    return logger


class MetricsWriter:
    """
    Writes metrics as JSON-lines to a run directory.

    JSON-lines format: one JSON object per line.
    Easy to read back with pandas: pd.read_json(path, lines=True)

    Args:
        run_dir: Directory for this run's outputs
        filename: Metrics file name (default 'metrics.jsonl')

    Usage:
        writer = MetricsWriter('experiments/phase1/runs/run_001')
        writer.write(step=100, agent_id=0, compression_ratio=0.73)
        writer.write(step=200, agent_id=0, compression_ratio=0.41)
        writer.close()
    """

    def __init__(self, run_dir: str, filename: str = 'metrics.jsonl'):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / filename
        self._file = open(self.metrics_path, 'a', encoding='utf-8')
        self._count = 0

        # Write run metadata header
        meta = {
            'type': 'run_metadata',
            'timestamp': datetime.now().isoformat(),
            'run_dir': str(self.run_dir),
        }
        self._file.write(json.dumps(meta) + '\n')
        self._file.flush()

    def write(self, step: int, **kwargs: Any) -> None:
        """
        Write one metrics record.

        Args:
            step: Current step number (required)
            **kwargs: Any key-value metrics to record

        Example:
            writer.write(step=100, agent_id=0, ratio=0.45, expr="(t*3+1) mod 7")
        """
        record = {
            'step': step,
            'ts': time.time(),
            **kwargs,
        }
        self._file.write(json.dumps(record) + '\n')
        self._file.flush()
        self._count += 1

    def close(self) -> None:
        """Close the metrics file."""
        if not self._file.closed:
            self._file.close()

    @property
    def path(self) -> Path:
        return self.metrics_path

    @property
    def num_records(self) -> int:
        return self._count

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return f"MetricsWriter(path={self.metrics_path}, records={self._count})"

    # Context manager support for convenient `with MetricsWriter(...) as w:`
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        # Do not suppress exceptions
        return False


def make_run_dir(base_dir: str = 'experiments/runs', prefix: str = 'run') -> str:
    """
    Create a unique timestamped run directory.

    Returns:
        Path string like 'experiments/runs/run_20240115_143022'
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
