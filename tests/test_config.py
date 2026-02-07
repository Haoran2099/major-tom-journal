"""Tests for Config class."""

import json

import pytest

from major_tom.config import Config


class TestConfig:
    def test_load_valid_config(self, sample_config_json, tmp_path):
        orig_log = Config.LOG_ROOT
        orig_mem = Config.MEMORY_ROOT
        orig_cfg = Config.CONFIG_PATH
        try:
            Config.CONFIG_PATH = sample_config_json
            Config.load_config()
            assert Config.BRAIN_MODEL == "test-brain:1b"
            assert Config.EYE_MODEL == "test-eye:1b"
            assert Config.SAMPLE_INTERVAL == 5
            assert Config.SEMANTIC_ENABLED is True
            assert Config.SEMANTIC_THRESHOLD == 0.5
            assert "SKIP" in Config.SEMANTIC_ROUTES
            assert Config.CONTEXT_ROUTING_ENABLED is True
        finally:
            Config.LOG_ROOT = orig_log
            Config.MEMORY_ROOT = orig_mem
            Config.CONFIG_PATH = orig_cfg

    def test_load_missing_config(self, tmp_path):
        orig = Config.CONFIG_PATH
        try:
            Config.CONFIG_PATH = tmp_path / "nonexistent.json"
            # Should not raise, just use defaults
            Config.load_config()
        finally:
            Config.CONFIG_PATH = orig

    def test_load_invalid_json(self, tmp_path):
        bad_config = tmp_path / "bad.json"
        bad_config.write_text("not valid json{{{")
        orig = Config.CONFIG_PATH
        try:
            Config.CONFIG_PATH = bad_config
            Config.load_config()  # Should not raise
        finally:
            Config.CONFIG_PATH = orig

    def test_save_route_new(self, tmp_config):
        config_path = tmp_config / "config.json"
        config_path.write_text("{}")
        Config.CONFIG_PATH = config_path

        Config.SEMANTIC_ROUTES = {"SKIP": [], "SNAPSHOT": []}
        Config.save_route("SKIP", "New test pattern")

        assert "New test pattern" in Config.SEMANTIC_ROUTES["SKIP"]

        with open(config_path) as f:
            data = json.load(f)
        assert "New test pattern" in data["semantic_router"]["routes"]["SKIP"]

    def test_save_route_duplicate(self, tmp_config):
        config_path = tmp_config / "config.json"
        config_path.write_text("{}")
        Config.CONFIG_PATH = config_path

        Config.SEMANTIC_ROUTES = {"SKIP": ["Existing"]}
        Config.save_route("SKIP", "Existing")
        # Should not duplicate
        assert Config.SEMANTIC_ROUTES["SKIP"].count("Existing") == 1

    def test_defaults(self):
        assert Config.SAMPLE_INTERVAL >= 1
        assert Config.IDLE_THRESHOLD > 0
        assert Config.VLM_COOLDOWN > 0
        assert 0 < Config.VISUAL_DIFF_THRESHOLD <= 1.0
