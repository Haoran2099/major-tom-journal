"""Dynamic sub-task classifier for tab-based and project-based applications."""

import logging
from typing import Dict, List, Optional

import numpy as np

from major_tom.config import Config
from major_tom.llm.base import LLMBackend

logger = logging.getLogger(__name__)


class ContextClassifier:
    """Routes context to different memory files based on window title/content."""

    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        self._llm = llm_backend
        self._keyword_embedding_cache: Dict[str, np.ndarray] = {}

    def classify_task_id(self, app: str, title: str) -> str:
        """Classify the task ID based on app name and window title."""
        if not Config.CONTEXT_ROUTING_ENABLED:
            return app

        app_rules = Config.CONTEXT_ROUTING_APPS.get(app)
        if not app_rules:
            return app

        if Config.CONTEXT_ROUTING_METHOD == "keyword":
            title_lower = title.lower()
            for category, keywords in app_rules.items():
                if any(kw.lower() in title_lower for kw in keywords):
                    return f"{app}_{category}"

        elif Config.CONTEXT_ROUTING_METHOD == "semantic" and self._llm is not None:
            return self._semantic_classify(app, title, app_rules)

        return f"{app}_{Config.CONTEXT_ROUTING_DEFAULT_SUFFIX}"

    def _semantic_classify(
        self, app: str, title: str, app_rules: Dict[str, List[str]]
    ) -> str:
        """Use embeddings to find the best matching category."""
        try:
            title_resp = self._llm.embed(Config.EMBEDDING_MODEL, title)
            title_vec = np.array(title_resp.vector, dtype=np.float32)
            if len(title_vec) == 0:
                return f"{app}_{Config.CONTEXT_ROUTING_DEFAULT_SUFFIX}"

            best_score = -1.0
            best_category = Config.CONTEXT_ROUTING_DEFAULT_SUFFIX

            for category, keywords in app_rules.items():
                for kw in keywords:
                    if kw not in self._keyword_embedding_cache:
                        kw_resp = self._llm.embed(Config.EMBEDDING_MODEL, kw)
                        self._keyword_embedding_cache[kw] = np.array(
                            kw_resp.vector, dtype=np.float32
                        )
                    kw_vec = self._keyword_embedding_cache[kw]
                    if len(kw_vec) > 0:
                        score = float(
                            np.dot(title_vec, kw_vec)
                            / (np.linalg.norm(title_vec) * np.linalg.norm(kw_vec))
                        )
                        if score > best_score:
                            best_score = score
                            best_category = category

            if best_score >= 0.6:
                return f"{app}_{best_category}"

        except Exception as e:
            logger.error("Classifier error: %s", e)

        return f"{app}_{Config.CONTEXT_ROUTING_DEFAULT_SUFFIX}"
