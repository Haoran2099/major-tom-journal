"""Tests for IntelligentContextRouter."""

import json

import numpy as np
import pytest

from major_tom.brain.context_router import IntelligentContextRouter
from major_tom.config import Config
from major_tom.llm.base import EmbeddingResponse, LLMResponse
from major_tom.llm.mock_backend import MockBackend
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.memory.task_block_manager import TaskBlockManager


class TestIntelligentContextRouter:
    @pytest.fixture
    def router_components(self, tmp_config):
        Config.SEMANTIC_ENABLED = False  # Disable semantic to test LLM path
        mock = MockBackend()
        logger = MarkdownStreamLogger()
        manager = TaskBlockManager(logger)
        return mock, logger, manager

    def test_semantic_path(self, tmp_config):
        Config.SEMANTIC_ENABLED = True
        Config.SEMANTIC_THRESHOLD = 0.3
        Config.SEMANTIC_ROUTES = {"SNAPSHOT": ["Writing code"]}

        mock = MockBackend()
        vec = np.ones(768).tolist()
        mock.set_embedding_response(EmbeddingResponse(vector=vec, prompt_tokens=5, model="mock"))

        logger = MarkdownStreamLogger()
        manager = TaskBlockManager(logger)
        router = IntelligentContextRouter(logger, manager, mock)

        decision = router._make_heavy_decision("VS Code", "main.py", {"kpm": 100}, "key1")
        assert decision["source"] == "SEMANTIC"

    def test_llm_path(self, router_components, snapshot_response):
        mock, logger, manager = router_components
        mock.set_generate_response(snapshot_response)

        router = IntelligentContextRouter(logger, manager, mock)
        decision = router._make_heavy_decision("VS Code", "main.py", {"kpm": 100}, "key_llm")

        assert decision["action"] == "SNAPSHOT"
        assert decision["source"] == "LLM_BRAIN"
        assert mock.generate_count >= 1

    def test_json_parse_error_returns_skip(self, router_components):
        mock, logger, manager = router_components
        mock.set_generate_response(LLMResponse(
            text="not valid json",
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            model="test", latency_ms=10.0,
        ))

        router = IntelligentContextRouter(logger, manager, mock)
        decision = router._make_heavy_decision("App", "Title", {}, "key_err")
        assert decision["action"] == "SKIP"
        assert decision["source"] == "ERROR"

    def test_cache_hit(self, router_components, skip_response):
        mock, logger, manager = router_components
        mock.set_generate_response(skip_response)

        router = IntelligentContextRouter(logger, manager, mock)
        router.cache["VS Code :: main.py"] = {
            "action": "SKIP", "source": "CACHE", "next_check_delay": 5
        }

        decisions = []
        router.decide_async("VS Code", "main.py - editor", {}, lambda d: decisions.append(d))

        assert len(decisions) == 1
        assert decisions[0]["source"] == "CACHE"
        # No LLM call should have been made
        assert mock.generate_count == 0

    def test_reset_working_state(self, router_components):
        mock, logger, manager = router_components
        router = IntelligentContextRouter(logger, manager, mock)

        router.working_state["summary"] = "old summary"
        router.reset_working_state("NewTask")

        assert "NewTask" in router.working_state["summary"]

    def test_learn_pattern_validation(self, router_components):
        mock, logger, manager = router_components

        # Set up response with learn_pattern=True but bad phrase
        bad_phrase_response = LLMResponse(
            text=json.dumps({
                "action": "SNAPSHOT",
                "reason": "test",
                "prompt": "test",
                "learn_pattern": True,
                "new_pattern_phrase": "ab",  # Too short
                "next_check_delay": 5,
                "region_mode": "ACTIVE_WINDOW",
                "updated_summary": "test",
            }),
            prompt_tokens=10, completion_tokens=10, total_tokens=20,
            model="test", latency_ms=10.0,
        )
        mock.set_generate_response(bad_phrase_response)

        router = IntelligentContextRouter(logger, manager, mock)
        Config.SEMANTIC_ENABLED = False
        router.semantic_router.enabled = False
        decision = router._make_heavy_decision("App", "Title", {}, "key_bad")
        # Should not crash, pattern should be rejected silently
        assert decision["action"] == "SNAPSHOT"
