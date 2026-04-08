"""Tests for ContextClassifier."""

import pytest

from major_tom.brain.context_classifier import ContextClassifier
from major_tom.config import Config
from major_tom.llm.base import EmbeddingResponse
from major_tom.llm.mock_backend import MockBackend


@pytest.fixture(autouse=True)
def _reset_config():
    orig_enabled = Config.CONTEXT_ROUTING_ENABLED
    orig_method = Config.CONTEXT_ROUTING_METHOD
    orig_apps = Config.CONTEXT_ROUTING_APPS
    orig_suffix = Config.CONTEXT_ROUTING_DEFAULT_SUFFIX
    yield
    Config.CONTEXT_ROUTING_ENABLED = orig_enabled
    Config.CONTEXT_ROUTING_METHOD = orig_method
    Config.CONTEXT_ROUTING_APPS = orig_apps
    Config.CONTEXT_ROUTING_DEFAULT_SUFFIX = orig_suffix


class TestContextClassifier:
    def test_keyword_match(self):
        Config.CONTEXT_ROUTING_ENABLED = True
        Config.CONTEXT_ROUTING_METHOD = "keyword"
        Config.CONTEXT_ROUTING_APPS = {
            "Safari": {"Research": ["arxiv", "paper"], "Fun": ["youtube"]},
        }
        classifier = ContextClassifier()
        assert classifier.classify_task_id("Safari", "Reading arxiv papers") == "Safari_Research"
        assert classifier.classify_task_id("Safari", "Watching youtube") == "Safari_Fun"

    def test_keyword_no_match_returns_default(self):
        Config.CONTEXT_ROUTING_ENABLED = True
        Config.CONTEXT_ROUTING_METHOD = "keyword"
        Config.CONTEXT_ROUTING_APPS = {"Safari": {"Research": ["arxiv"]}}
        Config.CONTEXT_ROUTING_DEFAULT_SUFFIX = "General"

        classifier = ContextClassifier()
        result = classifier.classify_task_id("Safari", "random page")
        assert result == "Safari_General"

    def test_disabled_returns_app(self):
        Config.CONTEXT_ROUTING_ENABLED = False
        classifier = ContextClassifier()
        assert classifier.classify_task_id("Safari", "anything") == "Safari"

    def test_unknown_app_returns_app(self):
        Config.CONTEXT_ROUTING_ENABLED = True
        Config.CONTEXT_ROUTING_APPS = {}
        classifier = ContextClassifier()
        assert classifier.classify_task_id("Notes", "My note") == "Notes"

    def test_semantic_classification_with_mock(self):
        Config.CONTEXT_ROUTING_ENABLED = True
        Config.CONTEXT_ROUTING_METHOD = "semantic"
        Config.CONTEXT_ROUTING_APPS = {
            "Safari": {"Research": ["academic paper"]},
        }

        mock = MockBackend()
        # Set a non-zero vector for meaningful cosine similarity
        import numpy as np
        vec = np.random.randn(768).tolist()
        mock.set_embedding_response(EmbeddingResponse(vector=vec, prompt_tokens=5, model="mock"))

        classifier = ContextClassifier(llm_backend=mock)
        result = classifier.classify_task_id("Safari", "Reading academic papers")
        # With identical vectors for all embeddings, similarity will be 1.0
        assert result.startswith("Safari_")
        assert mock.embed_count >= 1

    def test_semantic_fallback_on_no_backend(self):
        Config.CONTEXT_ROUTING_ENABLED = True
        Config.CONTEXT_ROUTING_METHOD = "semantic"
        Config.CONTEXT_ROUTING_APPS = {"Safari": {"Research": ["paper"]}}
        Config.CONTEXT_ROUTING_DEFAULT_SUFFIX = "General"

        classifier = ContextClassifier(llm_backend=None)
        result = classifier.classify_task_id("Safari", "test")
        assert result == "Safari_General"
