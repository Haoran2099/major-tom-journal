"""Global configuration for Major Tom Journal."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from major_tom.constants import (
    FILE_READ_LIMIT,
    HISTORY_MAX_SIZE,
    VLM_MAX_DIMENSION,
    DIFF_RESIZE_SIZE,
)

logger = logging.getLogger(__name__)


class Config:
    """Global configuration loaded from config.json or internal defaults."""

    USER_HOME = Path.home()
    MONITOR_PATH = Path(os.getenv("JOURNAL_MONITOR_PATH", USER_HOME / "Documents"))
    LOG_ROOT = USER_HOME / "Downloads" / "LLM_Journal" / "Record"
    MEMORY_ROOT = USER_HOME / "Downloads" / "LLM_Journal" / "Memory"
    CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.json"

    SAMPLE_INTERVAL = 5
    IDLE_THRESHOLD = 180
    MEDIA_IDLE_THRESHOLD = 300
    VLM_COOLDOWN = 60
    VISUAL_DIFF_THRESHOLD = 0.9

    BRAIN_MODEL = "qwen3:8b"
    EYE_MODEL = "qwen3-vl:8b"
    EMBEDDING_MODEL = "qwen3-embedding:8b"

    SEMANTIC_ENABLED = True
    SEMANTIC_THRESHOLD = 0.30
    SEMANTIC_ROUTES: Dict[str, List[str]] = {
        "SKIP": [
            "Browsing file explorer or Finder window",
            "System settings and control panel configuration",
            "Login screen or password entry manager",
            "Desktop background or empty screen",
        ],
        "SNAPSHOT": [
            "Writing code in Python, JavaScript, C++ or IDE",
            "Reading academic papers or technical documentation",
            "Conversations in chat applications",
        ],
    }

    CONTEXT_ROUTING_ENABLED = True
    CONTEXT_ROUTING_METHOD = "keyword"  # "keyword" or "semantic"
    CONTEXT_ROUTING_APPS: Dict[str, Dict[str, List[str]]] = {}
    CONTEXT_ROUTING_DEFAULT_SUFFIX = "General"

    VLM_SYSTEM_PROMPT = (
        "You are a Knowledge Extractor. Ignore UI layout. Focus strictly on semantic content. "
        "Analyze text, code, or diagrams. Output ONLY 1-2 dense, factual sentences summarizing "
        "WHAT is being learned or worked on."
    )

    BRAIN_SYSTEM_PROMPT = """[SYSTEM]
You are the Brain of an OS Agent. Manage the user's attention and memory.

[STATE]
Current Task Summary: "{summary}"
Context History:
{context}

[INPUT]
App: "{app}"
Title: "{title}"
Stats: {stats}

[GOALS]
1. ACTION: 'SNAPSHOT' (Work/Study/Chat) or 'SKIP' (Music/System/Idle).
2. PATTERN: Identify generic repeatable activities (e.g., "Watching tech videos", "Coding in Python").
   - DO NOT include verbs like "Block" or specific filenames in the pattern.
3. PACING:
   - HIGH FOCUS (Coding, Writing, Debugging): Delay = 10 seconds.
   - LOW FOCUS (Reading, Browsing, Video, Meeting): Delay = 30 to 60 seconds.
   - IDLE/MEDIA (Bilibili, YouTube, Music): Delay = 60 to 120 seconds.
4. REGION: If SNAPSHOT, choose 'ACTIVE_WINDOW' (Detail focus) or 'FULL_SCREEN' (Context focus).
5. STATE: Update the 'task_summary' to reflect what the user is doing NOW.

Output JSON ONLY:
{{
    "action": "SNAPSHOT" | "SKIP",
    "reason": "Brief reason",
    "prompt": "VLM instruction (e.g. 'Summarize code')",
    "learn_pattern": bool,
    "new_pattern_phrase": "Optional generic phrase",
    "next_check_delay": int,
    "region_mode": "ACTIVE_WINDOW" | "FULL_SCREEN",
    "updated_summary": "e.g. User is debugging python..."
}}
"""

    @classmethod
    def load_config(cls, config_path: Path = None) -> None:
        """Load configuration from external config file."""
        if "NO_PROXY" not in os.environ:
            os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

        path = config_path or cls.CONFIG_PATH
        if not path.exists():
            logger.info("Using internal defaults (no config file found).")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info("Loading config from %s ...", path)
            if "paths" in data:
                p = data["paths"]
                if "monitor_path" in p:
                    cls.MONITOR_PATH = Path(p["monitor_path"]).expanduser()
                if "log_root" in p:
                    cls.LOG_ROOT = Path(p["log_root"]).expanduser()
                if "memory_root" in p:
                    cls.MEMORY_ROOT = Path(p["memory_root"]).expanduser()
            if "parameters" in data:
                p = data["parameters"]
                cls.SAMPLE_INTERVAL = p.get("sample_interval", cls.SAMPLE_INTERVAL)
                cls.IDLE_THRESHOLD = p.get("idle_threshold", cls.IDLE_THRESHOLD)
                cls.VLM_COOLDOWN = p.get("vlm_cooldown", cls.VLM_COOLDOWN)
            if "models" in data:
                m = data["models"]
                cls.BRAIN_MODEL = m.get("brain_model", cls.BRAIN_MODEL)
                cls.EYE_MODEL = m.get("eye_model", cls.EYE_MODEL)
                cls.EMBEDDING_MODEL = m.get("embedding_model", cls.EMBEDDING_MODEL)
            if "semantic_router" in data:
                sr = data["semantic_router"]
                cls.SEMANTIC_ENABLED = sr.get("enabled", cls.SEMANTIC_ENABLED)
                cls.SEMANTIC_THRESHOLD = sr.get("similarity_threshold", cls.SEMANTIC_THRESHOLD)
                if "routes" in sr:
                    cls.SEMANTIC_ROUTES = sr["routes"]
            if "context_routing" in data:
                cr = data["context_routing"]
                cls.CONTEXT_ROUTING_ENABLED = cr.get("enabled", cls.CONTEXT_ROUTING_ENABLED)
                cls.CONTEXT_ROUTING_METHOD = cr.get("method", cls.CONTEXT_ROUTING_METHOD)
                cls.CONTEXT_ROUTING_DEFAULT_SUFFIX = cr.get(
                    "default_suffix", cls.CONTEXT_ROUTING_DEFAULT_SUFFIX
                )
                if "apps" in cr:
                    cls.CONTEXT_ROUTING_APPS = cr["apps"]

            cls.LOG_ROOT.mkdir(parents=True, exist_ok=True)
            cls.MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

        except (json.JSONDecodeError, OSError) as e:
            logger.error("Error reading config: %s. Reverting to defaults.", e)

    @classmethod
    def save_route(cls, action: str, phrase: str) -> None:
        """[Self-Evolution] Write learned rules to config.json."""
        if phrase in cls.SEMANTIC_ROUTES.get(action, []):
            return

        if action not in cls.SEMANTIC_ROUTES:
            cls.SEMANTIC_ROUTES[action] = []
        cls.SEMANTIC_ROUTES[action].append(phrase)

        try:
            data: Dict = {}
            if cls.CONFIG_PATH.exists():
                with open(cls.CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)

            if "semantic_router" not in data:
                data["semantic_router"] = {}
            if "routes" not in data["semantic_router"]:
                data["semantic_router"]["routes"] = {}

            current_routes = data["semantic_router"]["routes"].get(action, [])
            if phrase not in current_routes:
                current_routes.append(phrase)
                data["semantic_router"]["routes"][action] = current_routes

                with open(cls.CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info("[Evolution] Learned & Saved pattern: [%s] '%s'", action, phrase)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to save route: %s", e)

    @classmethod
    def reload(cls) -> None:
        """Reload configuration from disk (hot reload support)."""
        logger.info("Reloading configuration...")
        cls.load_config()
        logger.info("Configuration reloaded successfully.")
