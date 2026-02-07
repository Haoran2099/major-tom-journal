"""Stateful Brain with Self-Evolution & Adaptive Heartbeat."""

import concurrent.futures
import json
import logging
from typing import Any, Callable, Dict, Optional

from major_tom.config import Config
from major_tom.constants import MAX_PATTERN_LENGTH, MIN_PATTERN_LENGTH
from major_tom.llm.base import LLMBackend
from major_tom.memory.audit_logger import AuditLogger
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.memory.task_block_manager import TaskBlockManager
from major_tom.brain.semantic_gating import SemanticGatingLayer

logger = logging.getLogger(__name__)


class IntelligentContextRouter:
    """Stateful Brain combining semantic gating, decision cache, and LLM brain."""

    def __init__(
        self,
        md_logger: MarkdownStreamLogger,
        block_manager: TaskBlockManager,
        llm_backend: LLMBackend,
        audit_logger: Optional[AuditLogger] = None,
    ):
        self.logger = md_logger
        self.memory = block_manager
        self._llm = llm_backend
        self.audit = audit_logger
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._current_future = None
        self.semantic_router = SemanticGatingLayer(md_logger, llm_backend)
        self.cache_path = Config.LOG_ROOT / "decision_cache.json"
        self.cache = self._load_cache()

        self.working_state: Dict[str, object] = {
            "summary": "Session started.",
            "current_app": None,
        }

    def reset_working_state(self, task_id: str) -> None:
        """Reset working_state when switching tasks."""
        self.working_state["summary"] = f"Switched to {task_id}. Analyzing new context..."
        self.working_state["current_app"] = task_id
        logger.info("Reset state for new task: %s", task_id)

    def _load_cache(self) -> Dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _normalize_key(self, app: str, title: str) -> str:
        return f"{app} :: {title.split(' - ')[0][:50]}"

    def _make_heavy_decision(
        self,
        app: str,
        title: str,
        io_stats: Dict[str, float],
        cache_key: str,
    ) -> Dict:
        """Make decision using semantic router or LLM brain."""
        semantic_decision = self.semantic_router.match(app, title)
        if semantic_decision:
            self.cache[cache_key] = semantic_decision
            logger.info(
                "Semantic Hit: %s (%s)",
                semantic_decision["action"],
                semantic_decision["reason"],
            )
            if self.audit:
                self.audit.log(
                    component="Brain",
                    event_type="SEMANTIC_HIT",
                    data={
                        "app": app,
                        "title": title,
                        "matched_phrase": semantic_decision.get("reason", ""),
                        "action": semantic_decision.get("action", "SKIP"),
                        "total_tokens": semantic_decision.get("total_tokens", 0),
                    },
                )
            return semantic_decision

        try:
            context_summary = self.memory.get_context_summary()
            prompt = Config.BRAIN_SYSTEM_PROMPT.format(
                summary=self.working_state["summary"],
                context=context_summary,
                app=app,
                title=title,
                stats=io_stats,
            )

            res = self._llm.generate(
                model=Config.BRAIN_MODEL,
                prompt=prompt,
                format="json",
                stream=False,
                options={"num_ctx": 4096, "temperature": 0.1, "num_predict": 500},
            )

            decision = json.loads(res.text)
            decision["total_tokens"] = res.total_tokens
            decision.setdefault("action", "SKIP")
            decision["source"] = "LLM_BRAIN"

            if self.audit:
                self.audit.log(
                    component="Brain",
                    event_type="LLM_DECISION",
                    data={
                        "app": app,
                        "title": title,
                        "model": Config.BRAIN_MODEL,
                        "prompt": prompt,
                        "raw_response": res.text,
                        "parsed_decision": json.dumps(decision, indent=2, ensure_ascii=False),
                        "input_tokens": res.prompt_tokens,
                        "output_tokens": res.completion_tokens,
                        "total_tokens": res.total_tokens,
                    },
                )

            if decision.get("updated_summary"):
                self.working_state["summary"] = decision["updated_summary"]
                self.working_state["current_app"] = app

            if decision.get("learn_pattern") and decision.get("new_pattern_phrase"):
                phrase = decision["new_pattern_phrase"]
                is_valid = True
                bad_keywords = ["Block", "seconds", "context object", "json file"]
                if len(phrase) > MAX_PATTERN_LENGTH or len(phrase) < MIN_PATTERN_LENGTH:
                    is_valid = False
                if any(bad in phrase for bad in bad_keywords):
                    is_valid = False

                if is_valid:
                    self.semantic_router.learn_pattern(decision["action"], phrase)
                else:
                    logger.warning("Rejected bad pattern: '%s'", phrase)

            if decision["action"] == "SKIP":
                self.cache[cache_key] = decision
                self._save_cache()

            logger.info(
                "Brain: %s | Delay: %ss | State: %s...",
                decision["action"],
                decision.get("next_check_delay", 5),
                str(self.working_state["summary"])[:40],
            )
            return decision

        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Router error: %s", e)
            if self.audit:
                self.audit.log(
                    component="Brain",
                    event_type="LLM_ERROR",
                    data={"app": app, "title": title, "error": str(e), "model": Config.BRAIN_MODEL},
                )
            return {"action": "SKIP", "source": "ERROR", "next_check_delay": 5, "total_tokens": 0}

    def decide_async(
        self,
        app: str,
        title: str,
        io_stats: Dict[str, float],
        callback_func: Callable[[Dict], None],
    ) -> None:
        """Make asynchronous decision with caching support."""
        cache_key = self._normalize_key(app, title)

        if cache_key in self.cache:
            decision = self.cache[cache_key]
            decision["source"] = "CACHE"
            decision["next_check_delay"] = Config.SAMPLE_INTERVAL
            callback_func(decision)
            return

        if self._current_future and not self._current_future.done():
            return

        self._current_future = self.executor.submit(
            self._make_heavy_decision, app, title, io_stats, cache_key
        )
        self._current_future.add_done_callback(lambda f: callback_func(f.result()))

    def shutdown(self) -> None:
        """Shut down the thread pool executor."""
        self.executor.shutdown(wait=False)
