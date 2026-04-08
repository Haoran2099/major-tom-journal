"""Tests for TaskBlockManager."""

import threading

import pytest

from major_tom.config import Config
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.memory.task_block_manager import TaskBlockManager


class TestTaskBlockManager:
    def test_switch_task(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)

        mgr.update({"type": "INFO", "timestamp": "10:00", "content": "task A work"})
        mgr.switch_task("TaskB")

        assert mgr.current_task_id == "TaskB"
        assert len(mgr.active_history) == 0  # New task starts empty

        # Original task should be persisted
        assert (Config.MEMORY_ROOT / "startup.md").exists()

    def test_switch_back_restores_history(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)

        mgr.update({"type": "INFO", "timestamp": "10:00", "content": "original work"})
        mgr.switch_task("TaskB")
        mgr.update({"type": "INFO", "timestamp": "10:05", "content": "task B work"})
        mgr.switch_task("startup")

        assert mgr.current_task_id == "startup"
        assert len(mgr.active_history) > 0
        assert any("original work" in h.get("content", "") for h in mgr.active_history)

    def test_no_switch_to_same_task(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.current_task_id = "TaskA"
        mgr.switch_task("TaskA")  # Should be no-op
        assert mgr.current_task_id == "TaskA"

    def test_add_log_to_specific_task_active(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.current_task_id = "TaskA"

        mgr.add_log_to_specific_task("TaskA", {
            "type": "VLM", "timestamp": "10:00", "content": "visual data"
        })
        assert len(mgr.active_history) == 1

    def test_add_log_to_specific_task_background(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.current_task_id = "TaskA"

        mgr.add_log_to_specific_task("TaskB", {
            "type": "VLM", "timestamp": "10:00", "content": "background data"
        })

        # Should not pollute active history
        assert len(mgr.active_history) == 0

        # Should write to TaskB file
        bg_file = Config.MEMORY_ROOT / "TaskB.md"
        assert bg_file.exists()
        content = bg_file.read_text()
        assert "background data" in content

    def test_concurrent_writes(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.current_task_id = "Main"
        errors = []

        def write_main():
            try:
                for i in range(20):
                    mgr.update({"type": "INFO", "timestamp": "10:00", "content": f"main_{i}"})
            except Exception as e:
                errors.append(e)

        def write_bg():
            try:
                for i in range(20):
                    mgr.add_log_to_specific_task("Background", {
                        "type": "VLM", "timestamp": "10:00", "content": f"bg_{i}"
                    })
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=write_main)
        t2 = threading.Thread(target=write_bg)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors

    def test_get_context_summary_empty(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        summary = mgr.get_context_summary()
        assert "No recent actions" in summary

    def test_get_context_summary_with_history(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.update({"type": "INFO", "timestamp": "10:00", "content": "working on code"})
        summary = mgr.get_context_summary()
        assert "working on code" in summary

    def test_md_roundtrip(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.update({"type": "INFO", "timestamp": "10:00", "content": "test entry"})
        mgr._persist_task("startup")

        loaded = mgr._load_task("startup")
        assert len(loaded) > 0
        assert any("test entry" in e.get("content", "") for e in loaded)

    def test_sanitize_task_id(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.add_log_to_specific_task("", {"type": "INFO", "timestamp": "10:00", "content": "test"})
        # Empty id should become General_Task
        assert (Config.MEMORY_ROOT / "General_Task.md").exists()
