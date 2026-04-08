"""Shared test fixtures for Major Tom Journal."""

import json
import shutil
from pathlib import Path
from typing import Dict

import pytest

from major_tom.config import Config
from major_tom.llm.base import EmbeddingResponse, LLMResponse
from major_tom.llm.mock_backend import MockBackend


@pytest.fixture
def mock_llm():
    """Provide a MockBackend instance."""
    backend = MockBackend()
    yield backend
    backend.reset()


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary config environment and patch Config paths."""
    log_root = tmp_path / "Record"
    memory_root = tmp_path / "Memory"
    monitor_path = tmp_path / "Documents"
    log_root.mkdir()
    memory_root.mkdir()
    monitor_path.mkdir()

    # Save originals
    orig_log = Config.LOG_ROOT
    orig_mem = Config.MEMORY_ROOT
    orig_mon = Config.MONITOR_PATH
    orig_cfg = Config.CONFIG_PATH

    Config.LOG_ROOT = log_root
    Config.MEMORY_ROOT = memory_root
    Config.MONITOR_PATH = monitor_path
    Config.CONFIG_PATH = tmp_path / "config.json"

    yield tmp_path

    # Restore
    Config.LOG_ROOT = orig_log
    Config.MEMORY_ROOT = orig_mem
    Config.MONITOR_PATH = orig_mon
    Config.CONFIG_PATH = orig_cfg


@pytest.fixture
def sample_config_json(tmp_path) -> Path:
    """Write a sample config.json and return its path."""
    config_data = {
        "paths": {
            "log_root": str(tmp_path / "Record"),
            "memory_root": str(tmp_path / "Memory"),
            "monitor_path": str(tmp_path / "Documents"),
        },
        "parameters": {
            "sample_interval": 5,
            "idle_threshold": 180,
            "vlm_cooldown": 60,
        },
        "models": {
            "brain_model": "test-brain:1b",
            "eye_model": "test-eye:1b",
            "embedding_model": "test-embed:1b",
        },
        "semantic_router": {
            "enabled": True,
            "similarity_threshold": 0.5,
            "routes": {
                "SKIP": ["System settings"],
                "SNAPSHOT": ["Writing code"],
            },
        },
        "context_routing": {
            "enabled": True,
            "method": "keyword",
            "apps": {
                "Safari": {
                    "Research": ["arxiv", "paper"],
                    "Entertainment": ["youtube", "bilibili"],
                },
            },
            "default_suffix": "General",
        },
    }
    (tmp_path / "Record").mkdir(exist_ok=True)
    (tmp_path / "Memory").mkdir(exist_ok=True)
    (tmp_path / "Documents").mkdir(exist_ok=True)

    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_data, f)
    return config_path


@pytest.fixture
def sample_decision() -> Dict:
    """Return a sample brain decision dict."""
    return {
        "action": "SNAPSHOT",
        "reason": "User is coding",
        "prompt": "Summarize code",
        "learn_pattern": False,
        "new_pattern_phrase": "",
        "next_check_delay": 10,
        "region_mode": "ACTIVE_WINDOW",
        "updated_summary": "User is writing Python code",
        "source": "LLM_BRAIN",
        "total_tokens": 150,
    }


@pytest.fixture
def snapshot_response():
    """Return a mock LLM response for SNAPSHOT decisions."""
    return LLMResponse(
        text=json.dumps({
            "action": "SNAPSHOT",
            "reason": "User is writing code",
            "prompt": "Summarize code",
            "learn_pattern": False,
            "new_pattern_phrase": "",
            "next_check_delay": 10,
            "region_mode": "ACTIVE_WINDOW",
            "updated_summary": "User is writing Python code",
        }),
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        model="test-brain",
        latency_ms=100.0,
    )


@pytest.fixture
def skip_response():
    """Return a mock LLM response for SKIP decisions."""
    return LLMResponse(
        text=json.dumps({
            "action": "SKIP",
            "reason": "System idle",
            "prompt": "",
            "learn_pattern": False,
            "new_pattern_phrase": "",
            "next_check_delay": 30,
            "region_mode": "ACTIVE_WINDOW",
            "updated_summary": "System idle",
        }),
        prompt_tokens=80,
        completion_tokens=30,
        total_tokens=110,
        model="test-brain",
        latency_ms=50.0,
    )
