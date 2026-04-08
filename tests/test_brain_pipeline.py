"""Integration test: SemanticGating -> Router -> decision flow."""

import json

import numpy as np
import pytest

from major_tom.brain.context_router import IntelligentContextRouter
from major_tom.config import Config
from major_tom.llm.base import EmbeddingResponse, LLMResponse
from major_tom.llm.mock_backend import MockBackend
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.memory.task_block_manager import TaskBlockManager


class TestBrainPipeline:
    @pytest.fixture(autouse=True)
    def _setup_config(self, tmp_config):
        Config.SEMANTIC_ENABLED = True
        Config.SEMANTIC_THRESHOLD = 0.3
        Config.SEMANTIC_ROUTES = {
            "SKIP": ["System settings", "File explorer"],
            "SNAPSHOT": ["Writing code", "Reading papers"],
        }

    def test_semantic_hit_skips_llm(self):
        mock = MockBackend()
        # All embeddings return same vector -> cosine sim = 1.0 -> always hits
        vec = np.ones(768).tolist()
        mock.set_embedding_response(EmbeddingResponse(vector=vec, prompt_tokens=5, model="mock"))
        mock.set_generate_response(LLMResponse(
            text='{"action": "SNAPSHOT"}',
            prompt_tokens=100, completion_tokens=50, total_tokens=150,
            model="mock", latency_ms=500.0,
        ))

        logger = MarkdownStreamLogger()
        manager = TaskBlockManager(logger)
        router = IntelligentContextRouter(logger, manager, mock)

        decision = router._make_heavy_decision("Finder", "Documents", {"kpm": 0}, "test_key")

        assert decision["source"] == "SEMANTIC"
        # generate should NOT have been called (semantic intercepted)
        assert mock.generate_count == 0
        # But embed should have been called
        assert mock.embed_count > 0

    def test_semantic_miss_falls_through_to_llm(self):
        Config.SEMANTIC_ENABLED = False  # Disable semantic layer entirely

        mock = MockBackend()
        mock.set_generate_response(LLMResponse(
            text=json.dumps({
                "action": "SNAPSHOT",
                "reason": "User is coding",
                "prompt": "Analyze code",
                "learn_pattern": False,
                "new_pattern_phrase": "",
                "next_check_delay": 10,
                "region_mode": "ACTIVE_WINDOW",
                "updated_summary": "User coding in Python",
            }),
            prompt_tokens=200, completion_tokens=50, total_tokens=250,
            model="mock", latency_ms=1000.0,
        ))

        logger = MarkdownStreamLogger()
        manager = TaskBlockManager(logger)
        router = IntelligentContextRouter(logger, manager, mock)

        decision = router._make_heavy_decision("VS Code", "main.py", {"kpm": 100}, "test_key_2")

        assert decision["source"] == "LLM_BRAIN"
        assert decision["action"] == "SNAPSHOT"
        assert mock.generate_count == 1

    def test_cache_prevents_repeated_calls(self):
        mock = MockBackend()
        Config.SEMANTIC_ENABLED = False

        mock.set_generate_response(LLMResponse(
            text=json.dumps({
                "action": "SKIP", "reason": "idle", "prompt": "",
                "next_check_delay": 30, "region_mode": "ACTIVE_WINDOW",
                "updated_summary": "idle",
            }),
            prompt_tokens=100, completion_tokens=30, total_tokens=130,
            model="mock", latency_ms=500.0,
        ))

        logger = MarkdownStreamLogger()
        manager = TaskBlockManager(logger)
        router = IntelligentContextRouter(logger, manager, mock)

        # Use the normalized key that decide_async will compute
        cache_key = router._normalize_key("Finder", "Home")

        # First call via _make_heavy_decision - caches SKIP result
        d1 = router._make_heavy_decision("Finder", "Home", {}, cache_key)
        assert d1["action"] == "SKIP"
        calls_after_first = mock.generate_count

        # Second call via decide_async - should hit cache synchronously
        results = []
        router.decide_async("Finder", "Home", {}, lambda d: results.append(d))
        assert len(results) == 1
        assert results[0]["source"] == "CACHE"
        assert mock.generate_count == calls_after_first

    def test_full_pipeline_with_task_switch(self):
        mock = MockBackend()
        Config.SEMANTIC_ENABLED = False

        mock.set_generate_response(LLMResponse(
            text=json.dumps({
                "action": "SNAPSHOT", "reason": "coding", "prompt": "code",
                "next_check_delay": 10, "region_mode": "ACTIVE_WINDOW",
                "updated_summary": "Coding in Python",
            }),
            prompt_tokens=100, completion_tokens=30, total_tokens=130,
            model="mock", latency_ms=500.0,
        ))

        logger = MarkdownStreamLogger()
        manager = TaskBlockManager(logger)
        router = IntelligentContextRouter(logger, manager, mock)

        # Task A
        d1 = router._make_heavy_decision("VS Code", "main.py", {"kpm": 100}, "key_a")
        assert "Coding" in router.working_state["summary"]

        # Switch task
        manager.switch_task("Safari_Research")
        router.reset_working_state("Safari_Research")

        assert "Safari_Research" in router.working_state["summary"]
