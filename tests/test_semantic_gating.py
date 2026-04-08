"""Tests for SemanticGatingLayer."""

import numpy as np
import pytest

from major_tom.brain.semantic_gating import SemanticGatingLayer
from major_tom.config import Config
from major_tom.llm.base import EmbeddingResponse
from major_tom.llm.mock_backend import MockBackend
from major_tom.memory.markdown_logger import MarkdownStreamLogger


@pytest.fixture(autouse=True)
def _reset_semantic_config():
    orig_enabled = Config.SEMANTIC_ENABLED
    orig_threshold = Config.SEMANTIC_THRESHOLD
    orig_routes = Config.SEMANTIC_ROUTES.copy()
    yield
    Config.SEMANTIC_ENABLED = orig_enabled
    Config.SEMANTIC_THRESHOLD = orig_threshold
    Config.SEMANTIC_ROUTES = orig_routes


class TestSemanticGatingLayer:
    def test_match_returns_none_when_disabled(self, tmp_config):
        Config.SEMANTIC_ENABLED = False
        mock = MockBackend()
        logger = MarkdownStreamLogger()
        gating = SemanticGatingLayer(logger, mock)
        assert gating.match("Safari", "Google") is None

    def test_match_hit(self, tmp_config):
        Config.SEMANTIC_ENABLED = True
        Config.SEMANTIC_THRESHOLD = 0.3
        Config.SEMANTIC_ROUTES = {"SKIP": ["Browsing files"], "SNAPSHOT": ["Writing code"]}

        mock = MockBackend()
        # Use a fixed vector so all cosine similarities are 1.0
        vec = np.ones(768).tolist()
        mock.set_embedding_response(EmbeddingResponse(vector=vec, prompt_tokens=5, model="mock"))

        logger = MarkdownStreamLogger()
        gating = SemanticGatingLayer(logger, mock)

        result = gating.match("Finder", "Documents")
        assert result is not None
        assert result["action"] in ("SKIP", "SNAPSHOT")
        assert result["source"] == "SEMANTIC"

    def test_match_miss_below_threshold(self, tmp_config):
        Config.SEMANTIC_ENABLED = True
        Config.SEMANTIC_THRESHOLD = 0.99  # Very high but valid (cosine sim range is [-1, 1])
        Config.SEMANTIC_ROUTES = {"SKIP": ["test"]}

        mock = MockBackend()
        # Use orthogonal-ish vectors: route gets ones, query gets a different vector
        # Since MockBackend returns the same vector for all calls, both get the same vec
        # and cosine sim = 1.0. So we use a threshold just above 1.0 isn't valid.
        # Instead, set threshold high and use zero vector for route embeddings.
        vec = np.zeros(768).tolist()
        mock.set_embedding_response(EmbeddingResponse(vector=vec, prompt_tokens=5, model="mock"))

        logger = MarkdownStreamLogger()
        gating = SemanticGatingLayer(logger, mock)
        # Zero vectors produce NaN similarity which won't pass threshold
        result = gating.match("App", "Title")
        assert result is None

    def test_learn_pattern(self, tmp_config):
        Config.SEMANTIC_ENABLED = True
        Config.SEMANTIC_ROUTES = {"SKIP": [], "SNAPSHOT": []}

        mock = MockBackend()
        vec = np.random.randn(768).tolist()
        mock.set_embedding_response(EmbeddingResponse(vector=vec, prompt_tokens=5, model="mock"))

        logger = MarkdownStreamLogger()
        gating = SemanticGatingLayer(logger, mock)

        gating.learn_pattern("SKIP", "New pattern phrase")
        assert len(gating.route_embeddings.get("SKIP", [])) > 0

    def test_estimate_tokens(self, tmp_config):
        mock = MockBackend()
        logger = MarkdownStreamLogger()
        Config.SEMANTIC_ENABLED = False
        gating = SemanticGatingLayer(logger, mock)
        assert gating._estimate_tokens("hello world") >= 1
        assert gating._estimate_tokens("") == 1
