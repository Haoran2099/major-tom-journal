"""Vector Router with Dynamic Learning capabilities."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from major_tom.config import Config
from major_tom.llm.base import LLMBackend
from major_tom.memory.markdown_logger import MarkdownStreamLogger

logger = logging.getLogger(__name__)


class SemanticGatingLayer:
    """Embedding-based router that classifies contexts before invoking the LLM Brain."""

    def __init__(self, md_logger: MarkdownStreamLogger, llm_backend: LLMBackend):
        self.logger = md_logger
        self._llm = llm_backend
        self.enabled = Config.SEMANTIC_ENABLED
        self.route_embeddings: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        if self.enabled:
            try:
                logger.info("Initializing embeddings model: %s", Config.EMBEDDING_MODEL)
                self._llm.embed(Config.EMBEDDING_MODEL, "test")
                self._reindex_vectors()
                logger.info("Semantic Vectors Ready.")
            except Exception as e:
                logger.error("Failed to init embeddings: %s", e)
                self.enabled = False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get numpy embedding vector via LLMBackend."""
        try:
            resp = self._llm.embed(Config.EMBEDDING_MODEL, text)
            if resp.vector:
                return np.array(resp.vector, dtype=np.float32)
        except Exception as e:
            logger.error("Embed error: %s", e)
        return None

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _reindex_vectors(self) -> None:
        """Re-compute embeddings for all routes."""
        self.route_embeddings = {}
        for action, sentences in Config.SEMANTIC_ROUTES.items():
            self.route_embeddings[action] = []
            for phrase in sentences:
                vec = self._get_embedding(phrase)
                if vec is not None:
                    self.route_embeddings[action].append((phrase, vec))

    def learn_pattern(self, action: str, phrase: str) -> None:
        """Dynamic Learning API for adding new patterns."""
        if not self.enabled:
            return

        Config.save_route(action, phrase)
        try:
            vec = self._get_embedding(phrase)
            if vec is not None:
                if action not in self.route_embeddings:
                    self.route_embeddings[action] = []
                self.route_embeddings[action].append((phrase, vec))
                logger.info("Semantic Firewall updated with: '%s'", phrase)
        except Exception as e:
            logger.error("Vector update failed: %s", e)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate tokens for embedding call."""
        return max(1, len(text) // 4)

    def match(self, app: str, title: str) -> Optional[Dict]:
        """Match current context against learned patterns."""
        if not self.enabled:
            return None

        try:
            query_text = f"Using application {app} to {title}"
            dynamic_cost = self._estimate_tokens(query_text)
            query_vec = self._get_embedding(query_text)
            if query_vec is None:
                return None

            best_score = -1.0
            best_action = None
            best_phrase = ""

            for action, items in self.route_embeddings.items():
                for phrase, target_vec in items:
                    score = self._cosine_similarity(query_vec, target_vec)
                    if score > best_score:
                        best_score = score
                        best_action = action
                        best_phrase = phrase

            if best_score >= Config.SEMANTIC_THRESHOLD:
                return {
                    "action": best_action,
                    "prompt": f"Analyze {app}",
                    "reason": f"Semantic: {best_phrase} ({best_score:.2f})",
                    "source": "SEMANTIC",
                    "total_tokens": dynamic_cost,
                }
        except Exception as e:
            logger.error("Match error: %s", e)
        return None
