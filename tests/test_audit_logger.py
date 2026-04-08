"""Tests for AuditLogger."""

import threading

import pytest

from major_tom.config import Config
from major_tom.memory.audit_logger import AuditLogger


class TestAuditLogger:
    def test_log_creates_file(self, tmp_config):
        logger = AuditLogger()
        logger.log("Brain", "TEST_EVENT", {"key": "value"})

        assert logger.log_file.exists()
        content = logger.log_file.read_text()
        assert "TEST_EVENT" in content
        assert "key:" in content
        assert "value" in content

    def test_log_format(self, tmp_config):
        logger = AuditLogger()
        logger.log("Eye", "VLM_CALL", {"model": "test", "tokens": "100"})

        content = logger.log_file.read_text()
        assert "[Eye]" in content
        assert "[VLM_CALL]" in content
        assert "model:" in content

    def test_concurrent_logging(self, tmp_config):
        logger = AuditLogger()
        errors = []

        def log_many(thread_id):
            try:
                for i in range(20):
                    logger.log("Test", f"EVENT_{thread_id}_{i}", {"data": str(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_many, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        content = logger.log_file.read_text()
        # Should have entries from all threads
        assert "EVENT_0_" in content
        assert "EVENT_4_" in content
