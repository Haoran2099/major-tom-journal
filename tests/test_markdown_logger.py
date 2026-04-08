"""Tests for MarkdownStreamLogger."""

import datetime

import pytest

from major_tom.config import Config
from major_tom.memory.markdown_logger import MarkdownStreamLogger


class TestMarkdownStreamLogger:
    def test_creates_daily_file(self, tmp_config):
        logger = MarkdownStreamLogger()
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        file_path = Config.LOG_ROOT / f"{today}.md"
        assert file_path.exists()
        content = file_path.read_text()
        assert "Journal:" in content

    def test_log_vlm_analysis(self, tmp_config):
        logger = MarkdownStreamLogger()
        entry = logger.log("VLM_ANALYSIS", "Code analysis result", {"app": "VS Code"})

        assert entry["type"] == "VLM_ANALYSIS"
        assert entry["content"] == "Code analysis result"
        assert entry["context"]["app"] == "VS Code"

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        content = (Config.LOG_ROOT / f"{today}.md").read_text()
        assert "Visual: VS Code" in content

    def test_log_text_snapshot(self, tmp_config):
        logger = MarkdownStreamLogger()
        logger.log("TEXT_SNAPSHOT", "File content here", {"app": "VS Code"})

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        content = (Config.LOG_ROOT / f"{today}.md").read_text()
        assert "Text: VS Code" in content

    def test_log_idle_start(self, tmp_config):
        logger = MarkdownStreamLogger()
        logger.log("IDLE_START", "Inactive > 200s")

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        content = (Config.LOG_ROOT / f"{today}.md").read_text()
        assert "Away" in content

    def test_log_task_switch(self, tmp_config):
        logger = MarkdownStreamLogger()
        logger.log("TASK_SWITCH", "Switch: A -> B")

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        content = (Config.LOG_ROOT / f"{today}.md").read_text()
        assert "Switch: A -> B" in content

    def test_log_file_modified(self, tmp_config):
        logger = MarkdownStreamLogger()
        logger.log("FILE_MODIFIED", "Edited: test.py")

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        content = (Config.LOG_ROOT / f"{today}.md").read_text()
        assert "`Edited: test.py`" in content

    def test_log_returns_structured_data(self, tmp_config):
        logger = MarkdownStreamLogger()
        entry = logger.log("VLM_ANALYSIS", "test", {"app": "Safari", "title": "Google"})

        assert "timestamp" in entry
        assert entry["type"] == "VLM_ANALYSIS"
        assert entry["context"]["app"] == "Safari"

    def test_entry_types_all_write(self, tmp_config):
        logger = MarkdownStreamLogger()
        types = ["VLM_ANALYSIS", "TEXT_SNAPSHOT", "IDLE_START", "FILE_MODIFIED", "TASK_SWITCH", "CUSTOM"]
        for t in types:
            logger.log(t, f"content for {t}")

        today = datetime.datetime.now().strftime("%Y-%m-%d")
        content = (Config.LOG_ROOT / f"{today}.md").read_text()
        assert "CUSTOM" in content
