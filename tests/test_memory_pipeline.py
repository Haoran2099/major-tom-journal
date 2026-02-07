"""Integration test: task switch -> persist -> concurrent VLM writes."""

import threading

import pytest

from major_tom.config import Config
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.memory.task_block_manager import TaskBlockManager


class TestMemoryPipeline:
    def test_switch_and_persist(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)

        # Work on task A
        mgr.update({"type": "INFO", "timestamp": "10:00", "content": "coding main.py"})
        mgr.update({"type": "VLM", "timestamp": "10:01", "content": "visual: Python code"})

        # Switch to task B
        mgr.switch_task("Safari_Research")
        assert mgr.current_task_id == "Safari_Research"
        assert len(mgr.active_history) == 0

        # Verify task A persisted
        a_file = Config.MEMORY_ROOT / "startup.md"
        assert a_file.exists()
        content = a_file.read_text()
        assert "coding main.py" in content

        # Work on task B
        mgr.update({"type": "INFO", "timestamp": "10:05", "content": "reading arxiv paper"})

        # Switch back to task A
        mgr.switch_task("startup")
        assert mgr.current_task_id == "startup"
        assert any("coding main.py" in h.get("content", "") for h in mgr.active_history)

    def test_concurrent_vlm_writes_no_pollution(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)
        mgr.current_task_id = "VS_Code_Python"

        # Simulate VLM writing to previous task while main thread is on new task
        errors = []

        def vlm_write():
            """Simulate VLM worker writing to previous task."""
            try:
                for i in range(10):
                    mgr.add_log_to_specific_task("Safari_Research", {
                        "type": "VLM", "timestamp": "10:00",
                        "content": f"Safari visual data {i}",
                    })
            except Exception as e:
                errors.append(e)

        def main_write():
            """Main thread writing to current task."""
            try:
                for i in range(10):
                    mgr.update({
                        "type": "INFO", "timestamp": "10:00",
                        "content": f"VS Code activity {i}",
                    })
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=vlm_write)
        t2 = threading.Thread(target=main_write)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors

        # Check current task not polluted with Safari data
        for entry in mgr.active_history:
            assert "Safari visual data" not in entry.get("content", "")

        # Check background file has Safari data
        safari_file = Config.MEMORY_ROOT / "Safari_Research.md"
        assert safari_file.exists()
        safari_content = safari_file.read_text()
        assert "Safari visual data" in safari_content

    def test_rapid_task_switching(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)

        tasks = ["TaskA", "TaskB", "TaskC", "TaskA", "TaskB", "TaskA"]

        for task in tasks:
            mgr.update({
                "type": "INFO", "timestamp": "10:00",
                "content": f"work on {task}",
            })
            mgr.switch_task(task)

        # Should end on TaskA
        assert mgr.current_task_id == "TaskA"

        # All task files should exist
        for task in set(tasks):
            file = Config.MEMORY_ROOT / f"{task}.md"
            assert file.exists(), f"{task}.md should exist"

    def test_memory_isolation_after_switch_back(self, tmp_config):
        logger = MarkdownStreamLogger()
        mgr = TaskBlockManager(logger)

        # Task A: add coding entries
        mgr.switch_task("Coding")
        mgr.update({"type": "INFO", "timestamp": "10:00", "content": "writing Python code"})
        mgr.update({"type": "INFO", "timestamp": "10:01", "content": "debugging function"})

        # Task B: add reading entries
        mgr.switch_task("Reading")
        mgr.update({"type": "INFO", "timestamp": "10:05", "content": "reading research paper"})
        mgr.update({"type": "INFO", "timestamp": "10:06", "content": "taking notes on ML"})

        # Switch back to A
        mgr.switch_task("Coding")
        summary = mgr.get_context_summary()

        # Should contain coding content
        assert "Python code" in summary or "debugging" in summary
        # Should NOT contain reading content
        assert "research paper" not in summary
        assert "notes on ML" not in summary
