"""Ablation manager for selectively disabling components in experiments."""

import logging
from typing import Any, Dict, List, Optional

from major_tom.config import Config
from major_tom.experiments.config import ExperimentConfig
from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse
from major_tom.memory.markdown_logger import MarkdownStreamLogger

logger = logging.getLogger(__name__)


class NoOpSemanticGating:
    """Stub that always returns None (semantic gating disabled)."""

    def __init__(self, *args, **kwargs):
        self.enabled = False
        self.route_embeddings: Dict = {}

    def match(self, app: str, title: str) -> None:
        return None

    def learn_pattern(self, action: str, phrase: str) -> None:
        pass


class GlobalMemoryManager:
    """M0: Single global memory file, no task isolation."""

    def __init__(self, md_logger: MarkdownStreamLogger):
        self.logger = md_logger
        self.storage_path = Config.MEMORY_ROOT
        self.active_history: List[Dict] = []
        self.current_task_id = "global"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        import threading
        self._lock = threading.Lock()

    def update(self, log_entry: Dict) -> None:
        with self._lock:
            self.active_history.append(log_entry)
            from major_tom.constants import HISTORY_MAX_SIZE
            if len(self.active_history) > HISTORY_MAX_SIZE:
                self.active_history.pop(0)

    def add_log_to_specific_task(self, target_task_id: str, log_entry: Dict) -> None:
        self.update(log_entry)

    def switch_task(self, new_task_id: str, reason: str = "Context Switch") -> None:
        # No-op: global memory doesn't isolate tasks
        pass

    def get_context_summary(self) -> str:
        if not self.active_history:
            return "> (No recent actions)"
        from major_tom.constants import CONTEXT_HISTORY_LINES
        lines = ["# Global Context"]
        for h in self.active_history[-CONTEXT_HISTORY_LINES:]:
            content = h.get("content", "")[:500]
            lines.append(f"- {content}")
        return "\n".join(lines)

    def _persist_task(self, task_id: str) -> None:
        file_path = self.storage_path / "global.md"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("# Global Memory\n\n")
                for entry in self.active_history:
                    content = entry.get("content", "")
                    f.write(f"- {content}\n")
        except OSError:
            pass


class AppOnlyMemoryManager:
    """M1: Memory isolated by app name only, no sub-context distinction."""

    def __init__(self, md_logger: MarkdownStreamLogger):
        self.logger = md_logger
        self.storage_path = Config.MEMORY_ROOT
        self.current_task_id = "startup"
        self.active_history: List[Dict] = []
        self._histories: Dict[str, List[Dict]] = {}
        self.storage_path.mkdir(parents=True, exist_ok=True)
        import threading
        self._lock = threading.Lock()

    def _app_key(self, task_id: str) -> str:
        return task_id.split("_")[0] if "_" in task_id else task_id

    def update(self, log_entry: Dict) -> None:
        self.add_log_to_specific_task(self.current_task_id, log_entry)

    def add_log_to_specific_task(self, target_task_id: str, log_entry: Dict) -> None:
        app_key = self._app_key(target_task_id)
        with self._lock:
            if app_key not in self._histories:
                self._histories[app_key] = []
            self._histories[app_key].append(log_entry)
            from major_tom.constants import HISTORY_MAX_SIZE
            if len(self._histories[app_key]) > HISTORY_MAX_SIZE:
                self._histories[app_key].pop(0)
            if app_key == self._app_key(self.current_task_id):
                self.active_history = self._histories[app_key]

    def switch_task(self, new_task_id: str, reason: str = "Context Switch") -> None:
        app_key = self._app_key(new_task_id)
        with self._lock:
            self.current_task_id = new_task_id
            if app_key not in self._histories:
                self._histories[app_key] = []
            self.active_history = self._histories[app_key]

    def get_context_summary(self) -> str:
        app_key = self._app_key(self.current_task_id)
        history = self._histories.get(app_key, [])
        if not history:
            return "> (No recent actions)"
        from major_tom.constants import CONTEXT_HISTORY_LINES
        lines = [f"# App Context: {app_key}"]
        for h in history[-CONTEXT_HISTORY_LINES:]:
            content = h.get("content", "")[:500]
            lines.append(f"- {content}")
        return "\n".join(lines)

    def _persist_task(self, task_id: str) -> None:
        app_key = self._app_key(task_id)
        history = self._histories.get(app_key, [])
        file_path = self.storage_path / f"{app_key}.md"
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# App Memory: {app_key}\n\n")
                for entry in history:
                    content = entry.get("content", "")
                    f.write(f"- {content}\n")
        except OSError:
            pass


class AblationManager:
    """Applies experiment config to selectively enable/disable components."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def apply_to_config(self) -> None:
        """Mutate the global Config based on experiment settings."""
        Config.BRAIN_MODEL = self.config.brain_model
        Config.EYE_MODEL = self.config.eye_model
        Config.EMBEDDING_MODEL = self.config.embedding_model
        Config.SEMANTIC_ENABLED = self.config.semantic_gating_enabled
        Config.SEMANTIC_THRESHOLD = self.config.semantic_gating_threshold
        Config.CONTEXT_ROUTING_ENABLED = self.config.context_routing_enabled
        Config.CONTEXT_ROUTING_METHOD = self.config.context_routing_method
        Config.VLM_COOLDOWN = self.config.vlm_cooldown
        Config.VISUAL_DIFF_THRESHOLD = self.config.vlm_diff_threshold

    def get_memory_manager(self, md_logger: MarkdownStreamLogger):
        """Return the appropriate memory manager for the experiment's dimension."""
        dimension = self.config.dimension
        name = self.config.name.lower()

        if "m0" in name or "global" in name:
            return GlobalMemoryManager(md_logger)
        elif "m1" in name or "app_only" in name:
            return AppOnlyMemoryManager(md_logger)
        else:
            from major_tom.memory.task_block_manager import TaskBlockManager
            return TaskBlockManager(md_logger)

    def should_use_vlm(self) -> bool:
        return self.config.vlm_enabled

    def should_use_cache(self) -> bool:
        return self.config.decision_cache_enabled
