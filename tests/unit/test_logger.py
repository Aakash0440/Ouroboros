# tests/unit/test_logger.py

"""Tests for logging infrastructure."""

import json
import os
import tempfile
import pytest
from ouroboros.utils.logger import get_logger, MetricsWriter, make_run_dir


class TestMetricsWriter:

    def test_creates_file_on_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir, 'test_metrics.jsonl')
            writer.close()
            assert os.path.exists(os.path.join(tmpdir, 'test_metrics.jsonl'))

    def test_writes_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)
            writer.write(step=1, agent_id=0, ratio=0.5)
            writer.write(step=2, agent_id=1, ratio=0.4)
            writer.close()

            lines = open(writer.path).readlines()
            # First line is metadata, then 2 records
            assert len(lines) == 3

    def test_records_are_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)
            writer.write(step=10, value=3.14)
            writer.close()

            with open(writer.path) as f:
                lines = f.readlines()
            # Parse each line — should not raise
            for line in lines:
                obj = json.loads(line)
                assert isinstance(obj, dict)

    def test_step_is_in_record(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = MetricsWriter(tmpdir)
            writer.write(step=42, test_val=99)
            writer.close()

            with open(writer.path) as f:
                records = [json.loads(l) for l in f.readlines()]
            data_records = [r for r in records if r.get('step') == 42]
            assert len(data_records) == 1
            assert data_records[0]['test_val'] == 99

    def test_make_run_dir_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = make_run_dir(tmpdir, 'test_run')
            assert os.path.isdir(run_dir)
            assert 'test_run' in run_dir


class TestGetLogger:
    def test_returns_logger(self):
        import logging
        logger = get_logger('test_logger')
        assert isinstance(logger, logging.Logger)

    def test_same_name_same_logger(self):
        l1 = get_logger('my_logger')
        l2 = get_logger('my_logger')
        assert l1 is l2