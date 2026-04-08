import concurrent.futures
import ctypes
import datetime
import json
import os
if "NO_PROXY" not in os.environ:
            os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
import platform
import queue
import re
import subprocess
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyautogui
from PIL import Image, ImageChops
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

try:
    import numpy as np
    HAS_SEMANTIC_LIB = True
except ImportError:
    print(">> [Warning] 'numpy' not found. Semantic Router disabled.")
    HAS_SEMANTIC_LIB = False

import ollama
from pynput import keyboard, mouse

FILE_READ_LIMIT = 3000
HISTORY_MAX_SIZE = 50
CONTEXT_HISTORY_LINES = 10
VLM_MAX_DIMENSION = 960
DIFF_RESIZE_SIZE = 64
MIN_WINDOW_WIDTH = 10
MIN_PATTERN_LENGTH = 5
MAX_PATTERN_LENGTH = 50


class Config:
    """
    Global configuration
    """
    USER_HOME = Path.home()
    MONITOR_PATH = Path(os.getenv("JOURNAL_MONITOR_PATH", USER_HOME / "Documents"))
    LOG_ROOT = USER_HOME / "Downloads" / "LLM_Journal" / "Record"
    MEMORY_ROOT = USER_HOME / "Downloads" / "LLM_Journal" / "Memory"
    CONFIG_PATH = Path(__file__).parent / "config.json"

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
    SEMANTIC_ROUTES = {
        "SKIP": [
            "Browsing file explorer or Finder window",
            "System settings and control panel configuration",
            "Login screen or password entry manager",
            "Desktop background or empty screen"
        ],
        "SNAPSHOT": [
            "Writing code in Python, JavaScript, C++ or IDE",
            "Reading academic papers or technical documentation",
            "Conversations in chat applications"
        ]
    }

    # Context Routing for sub-task classification
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
    def load_config(cls) -> None:
        """Load configuration from external config file."""
        if "NO_PROXY" not in os.environ:
            os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"

        if not cls.CONFIG_PATH.exists():
            print(f">> [Config] Using internal defaults.")
            return

        try:
            with open(cls.CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f">> [Config] Loading from {cls.CONFIG_PATH}...")
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

            # Load context routing configuration
            if "context_routing" in data:
                cr = data["context_routing"]
                cls.CONTEXT_ROUTING_ENABLED = cr.get("enabled", cls.CONTEXT_ROUTING_ENABLED)
                cls.CONTEXT_ROUTING_METHOD = cr.get("method", cls.CONTEXT_ROUTING_METHOD)
                cls.CONTEXT_ROUTING_DEFAULT_SUFFIX = cr.get("default_suffix", cls.CONTEXT_ROUTING_DEFAULT_SUFFIX)
                if "apps" in cr:
                    cls.CONTEXT_ROUTING_APPS = cr["apps"]

            cls.LOG_ROOT.mkdir(parents=True, exist_ok=True)
            cls.MEMORY_ROOT.mkdir(parents=True, exist_ok=True)

        except (json.JSONDecodeError, OSError) as e:
            print(f">> [Config] Error reading config: {e}. Reverting to defaults.")

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
                with open(cls.CONFIG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            if "semantic_router" not in data:
                data["semantic_router"] = {}
            if "routes" not in data["semantic_router"]:
                data["semantic_router"]["routes"] = {}

            current_routes = data["semantic_router"]["routes"].get(action, [])
            if phrase not in current_routes:
                current_routes.append(phrase)
                data["semantic_router"]["routes"][action] = current_routes

                with open(cls.CONFIG_PATH, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f">> [💭 Evolution] Learned & Saved pattern: [{action}] '{phrase}'")
        except (OSError, json.JSONDecodeError) as e:
            print(f">> [Config Error] Failed to save route: {e}")


class ContextClassifier:
    """
    Dynamic sub-task classifier for tab-based and project-based applications.
    Routes context to different memory files based on window title/content.
    """

    @staticmethod
    def classify_task_id(app: str, title: str) -> str:
        """
        Classify the task ID based on app name and window title.
        Returns a composite ID like "Safari_Research" or just "Safari" if no match.
        """
        if not Config.CONTEXT_ROUTING_ENABLED:
            return app

        # Check if this app has routing rules
        app_rules = Config.CONTEXT_ROUTING_APPS.get(app)
        if not app_rules:
            return app

        # Keyword-based classification (fast, no LLM overhead)
        if Config.CONTEXT_ROUTING_METHOD == "keyword":
            title_lower = title.lower()
            for category, keywords in app_rules.items():
                if any(kw.lower() in title_lower for kw in keywords):
                    return f"{app}_{category}"

        # Semantic classification (if enabled and embedding available)
        elif Config.CONTEXT_ROUTING_METHOD == "semantic" and HAS_SEMANTIC_LIB:
            return ContextClassifier._semantic_classify(app, title, app_rules)

        # Default: return app name with default suffix
        return f"{app}_{Config.CONTEXT_ROUTING_DEFAULT_SUFFIX}"

    @staticmethod
    def _semantic_classify(app: str, title: str, app_rules: Dict[str, List[str]]) -> str:
        """
        Use embeddings to find the best matching category.
        Higher accuracy but requires embedding model.
        """
        try:
            import numpy as np

            # Get embedding for window title
            title_vec = ContextClassifier._get_embedding(title)
            if title_vec is None:
                return f"{app}_{Config.CONTEXT_ROUTING_DEFAULT_SUFFIX}"

            best_score = -1.0
            best_category = Config.CONTEXT_ROUTING_DEFAULT_SUFFIX

            # Compare with category keyword embeddings
            for category, keywords in app_rules.items():
                for kw in keywords:
                    kw_vec = ContextClassifier._get_embedding(kw)
                    if kw_vec is not None:
                        score = np.dot(title_vec, kw_vec) / (np.linalg.norm(title_vec) * np.linalg.norm(kw_vec))
                        if score > best_score:
                            best_score = score
                            best_category = category

            # Only use classification if confidence is high enough
            if best_score >= 0.6:
                return f"{app}_{best_category}"

        except Exception as e:
            print(f"   └── [Classifier Error] {e}")

        return f"{app}_{Config.CONTEXT_ROUTING_DEFAULT_SUFFIX}"

    @staticmethod
    def _get_embedding(text: str) -> Optional[np.ndarray]:
        """Get embedding vector from Ollama."""
        try:
            resp = ollama.embeddings(model=Config.EMBEDDING_MODEL, prompt=text)
            vec = resp.get('embedding')
            if vec:
                return np.array(vec, dtype=np.float32)
        except Exception:
            pass
        return None


class ContextTools:
    """
    Read-only tools to fetch context cheaply (CPU-only).
    Improved search paths and title parsing.
    """
    FILE_EXTENSIONS = "py|md|txt|json|js|ts|c|cpp|h|java|rs|go|tex|log"

    @staticmethod
    def read_active_file(window_title: str, monitor_path: Path) -> Optional[str]:
        """
        Attempts to find and read the file currently open in the active window.

        Args:
            window_title: The title of the active window.
            monitor_path: The configured path to monitor for file changes.

        Returns:
            File content if found and readable, None otherwise.
        """
        match = re.search(
            rf"([^\\/:\*\?\"<>\|]+\.({ContextTools.FILE_EXTENSIONS}))",
            window_title,
            re.IGNORECASE
        )

        if not match:
            return None

        target_filename = match.group(1)

        search_roots = [
            Path.cwd(),
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            monitor_path,
        ]

        found_file = None

        for root in search_roots:
            if not root.exists():
                continue

            direct_path = root / target_filename
            if direct_path.exists():
                found_file = direct_path
                break

            if root == monitor_path:
                excluded_dirs = {'.git', 'node_modules', 'build', 'dist', '__pycache__', '.venv', 'venv'}
                max_depth = 3
                for current_dir, dirs, files in os.walk(root, followlinks=True):
                    dirs[:] = [d for d in dirs if d not in excluded_dirs]
                    current_depth = current_dir.replace(str(root), '').count(os.sep)
                    if current_depth >= max_depth:
                        dirs[:] = []
                        continue
                    if target_filename in files:
                        found_file = Path(current_dir) / target_filename
                        break
                if found_file:
                    break

        if not found_file:
            print(f">> [Tool] Filename '{target_filename}' detected, but file not found on disk.")
            return None

        try:
            with open(found_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(FILE_READ_LIMIT)

            print(f">> [Tool] 📖 Successfully read: {found_file.name}")
            return f"## [FILE_READ] Path: {found_file}\n```\n{content}\n...\n```"
        except (OSError, UnicodeDecodeError) as e:
            print(f">> [Tool Error] Read failed: {e}")
            return None

    @staticmethod
    def read_clipboard() -> Optional[str]:
        """Get text content from clipboard safely."""
        try:
            if platform.system() == "Darwin":
                p = subprocess.check_output(['pbpaste'], stderr=subprocess.STDOUT)
                content = p.decode('utf-8').strip()
                if len(content) > 5 and len(content) < 3000:
                    return f"## [CLIPBOARD]\n> {content}"
        except (subprocess.SubprocessError, OSError):
            pass
        return None


class AuditLogger:
    """
    Decision audit logger for debugging LLM/VLM decision processes.
    Records prompts, outputs, and routing decisions to a dedicated log file.
    """
    def __init__(self):
        self.log_dir = Config.LOG_ROOT
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "decision_debug.log"
        self._lock = threading.Lock()

    def log(self, component: str, event_type: str, data: Dict[str, Any]) -> None:
        """
        Log a decision event with timestamp.

        Args:
            component: Component name (e.g., 'Brain', 'Eye', 'SemanticRouter')
            event_type: Type of event (e.g., 'LLM_CALL', 'VLM_CALL', 'STATIC_SKIP')
            data: Dictionary containing relevant data (prompt, output, scores, etc.)
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        log_entry = f"\n{'='*80}\n"
        log_entry += f"[{timestamp}] [{component}] [{event_type}]\n"
        log_entry += f"{'-'*80}\n"

        for key, value in data.items():
            if value is not None:
                log_entry += f"{key}:\n{value}\n"

        log_entry += f"{'='*80}\n"

        with self._lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
            except OSError as e:
                print(f"[AuditLogger Error] Failed to write: {e}")


class MarkdownStreamLogger:
    """
    Stream-based Markdown logger for real-time memory tracking.
    Writes directly to Markdown files for human-readable records.
    """
    def __init__(self):
        self.log_dir = Config.LOG_ROOT
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_day = datetime.datetime.now().strftime('%Y-%m-%d')
        self._ensure_header()

    def _ensure_header(self) -> None:
        """Ensure each day's file has a proper Markdown header."""
        file_path = self.log_dir / f"{self.current_day}.md"
        if not file_path.exists():
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# 📅 Journal: {self.current_day}\n")
                f.write(f"> Auto-generated by Major_Tom_Journal\n\n---\n")

    def log(
        self,
        entry_type: str,
        content: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Log an entry to the daily Markdown file.

        Args:
            entry_type: Type of the log entry (e.g., VLM_ANALYSIS, TEXT_SNAPSHOT).
            content: The main content to log.
            context: Additional context information.

        Returns:
            Structured data dictionary for memory management.
        """
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        if today != self.current_day:
            self.current_day = today
            self._ensure_header()

        timestamp = datetime.datetime.now().strftime("%H:%M")
        timestamp_with_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        file_path = self.log_dir / f"{self.current_day}.md"

        app = context.get('app', 'Unknown') if context else 'Unknown'

        if entry_type == "VLM_ANALYSIS":
            md_block = f"\n### 👁️ {timestamp} | Visual: {app}\n> {content}\n"
        elif entry_type == "TEXT_SNAPSHOT":
            md_block = f"\n### 📄 {timestamp} | Text: {app}\n{content}\n"
        elif entry_type == "IDLE_START":
            md_block = f"\n> 💤 **Away** ({timestamp}): {content}\n"
        elif entry_type == "FILE_MODIFIED":
            md_block = f"- *{timestamp}* 📝 File Edited: `{content}`\n"
        elif entry_type == "TASK_SWITCH":
            md_block = f"\n---\n**🔄 {timestamp} {content}**\n\n"
        else:
            md_block = f"- *{timestamp}* [{entry_type}] {content}\n"

        print(f"{timestamp} | {entry_type[:4]} | {content[:60]}...")

        try:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(md_block)
        except OSError:
            pass

        return {
            "timestamp": timestamp_with_date,
            "type": entry_type,
            "content": content,
            "context": context or {}
        }


class InputActivitySensor:
    def __init__(self):
        self._keystrokes = 0
        self._clicks = 0
        self._lock = threading.Lock()
        self.kb_listener = keyboard.Listener(on_press=self._on_press)
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.kb_listener.start()
        self.mouse_listener.start()

    def _on_press(self, _) -> None:
        with self._lock:
            self._keystrokes += 1

    def _on_click(self, *args) -> None:
        if len(args) >= 4 and args[3]:
            with self._lock:
                self._clicks += 1

    def get_and_reset_stats(self, duration_seconds: float) -> Dict[str, float]:
        with self._lock:
            k, c = self._keystrokes, self._clicks
            self._keystrokes = 0
            self._clicks = 0
        factor = 60.0 / max(duration_seconds, 1.0)
        return {"kpm": int(k * factor), "cpm": int(c * factor)}


class IdleSensor:
    def __init__(self):
        self.os_type = platform.system()
        self._last_idle = 0.0
        self._last_check_time = 0.0
        self._cache_ttl = 5.0

    def get_idle_duration(self) -> float:
        now = time.time()
        if now - self._last_check_time < self._cache_ttl:
            return self._last_idle

        self._last_check_time = now
        try:
            if self.os_type == "Darwin":
                cmd = "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF; exit}'"
                output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
                self._last_idle = int(output) / 1_000_000_000 if output else 0
            elif self.os_type == "Windows":
                class LastInputInfo(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

                lii = LastInputInfo()
                lii.cbSize = ctypes.sizeof(LastInputInfo)
                if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                    self._last_idle = (ctypes.windll.kernel32.GetTickCount() - lii.dwTime) / 1000.0
                else:
                    self._last_idle = 0.0
            else:
                self._last_idle = 0.0
        except (subprocess.SubprocessError, OSError, AttributeError):
            self._last_idle = 0.0

        return self._last_idle


class PlatformSensor:
    def __init__(self):
        self._last_result = ("Unknown", "Unknown", None)
        self._last_check_time = 0
        self._cache_ttl = 1.0

    def _is_valid_title(self, title: str) -> bool:
        """Check if window title is meaningful (not placeholder)."""
        if not title or not title.strip():
            return False
        invalid_patterns = [
            "未命名", "Untitled", "无标题",
            "new tab", "New Tab", "_blank"
        ]
        return not any(pattern in title for pattern in invalid_patterns)

    def get_active_window(self) -> Tuple[str, str, Optional[Tuple[int, int, int, int]]]:
        if time.time() - self._last_check_time < self._cache_ttl:
            return self._last_result

        system = platform.system()
        current_app, current_title, current_region = self._last_result

        try:
            if system == "Darwin":
                script = '''global frontApp, windowTitle, winPos, winSize
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    try
                        tell process frontApp
                            set windowTitle to value of attribute "AXTitle" of window 1
                            set winPos to value of attribute "AXPosition" of window 1
                            set winSize to value of attribute "AXSize" of window 1
                        end tell
                    on error
                        set windowTitle to ""
                        set winPos to {0, 0}
                        set winSize to {0, 0}
                    end try
                end tell
                return frontApp & " ||| " & windowTitle & " ||| " & (item 1 of winPos) & ", " & (item 2 of winPos) & " ||| " & (item 1 of winSize) & ", " & (item 2 of winSize)'''
                res = subprocess.check_output(
                    ['osascript', '-e', script],
                    stderr=subprocess.STDOUT
                ).decode().strip()

                if "|||" in res:
                    parts = res.split(" ||| ")
                    fetched_app = parts[0]
                    fetched_title = parts[1].strip()

                    try:
                        x, y = map(int, parts[2].split(", "))
                        w, h = map(int, parts[3].split(", "))
                        region = (x, y, w, h) if w > MIN_WINDOW_WIDTH else None
                    except ValueError:
                        region = None

                    if fetched_app != current_app:
                        current_app = fetched_app
                        if self._is_valid_title(fetched_title):
                            current_title = fetched_title
                        else:
                            current_title = ""
                        current_region = region
                    elif self._is_valid_title(fetched_title):
                        current_title = fetched_title
                        current_region = region

            elif system == "Windows":
                import pygetwindow as gw
                win = gw.getActiveWindow()
                if win:
                    current_app = win.title.split(" - ")[-1] if " - " in win.title else "Windows App"
                    current_title = win.title
                    current_region = (win.left, win.top, win.width, win.height)

        except (subprocess.SubprocessError, OSError, ImportError):
            pass

        self._last_result = (current_app, current_title, current_region)
        self._last_check_time = time.time()
        return self._last_result


class TaskBlockManager:
    """
    Human-in-the-Loop Memory manager using Markdown for task context.
    Allows users to edit memory files directly to correct agent behavior.
    """
    def __init__(self, logger: MarkdownStreamLogger):
        self.logger = logger
        self.storage_path = Config.MEMORY_ROOT
        self.active_history: List[Dict] = []
        self.current_task_id = "startup"
        self._lock = threading.Lock()  # Thread-safe lock to prevent race conditions between main and worker threads

        self.storage_path.mkdir(parents=True, exist_ok=True)

    def _entry_to_md(self, entry: Dict) -> str:
        """Convert structured data to Markdown list item."""
        content = entry.get('content', '').replace('\n', ' ')

        # Fix: Remove all duplicate app tags if content already contains them
        import re
        content = re.sub(r'➣ \[[^\]]+\]\s*', '', content)

        timestamp = entry.get('timestamp', '00:00')
        etype = entry.get('type', 'INFO')

        # Extract window title from entry context for better traceability
        context = entry.get('context', {})
        title = context.get('title', '')
        title_str = f" | {title}" if title else ""

        # Fix: Always use current task ID to ensure correct app tagging
        return f"- **[{etype}]** ({timestamp}): ➣ [{self.current_task_id}]{title_str} {content}"

    def _md_to_entry(self, line: str) -> Optional[Dict]:
        """Parse Markdown line back to dictionary."""
        line = line.strip()
        if not line.startswith("-"):
            return None

        match = re.match(r"- \*\*\[(.*?)\]\*\* \((.*?)\): (.*)", line)

        if match:
            return {
                "type": match.group(1),
                "timestamp": match.group(2),
                "content": match.group(3)
            }

        return {
            "type": "USER_NOTE",
            "timestamp": "Manual",
            "content": line.replace("- ", "").replace("**", "")
        }

    def _persist_task(self, task_id: str) -> None:
        """Save current task state to .md file."""
        file_path = self.storage_path / f"{task_id}.md"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Context Memory: {task_id}\n")
                f.write(f"> Last Active: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write("> **Tip**: You can edit this file to guide the Agent's context.\n\n")

                for entry in self.active_history:
                    f.write(self._entry_to_md(entry) + "\n")

            print(f"   └── [Memory] Saved '{task_id}.md'")
        except OSError as e:
            print(f"   └── [Memory Error] Save failed: {e}")

    def _load_task(self, task_id: str) -> List[Dict]:
        """Load task state from .md file."""
        file_path = self.storage_path / f"{task_id}.md"
        loaded_history: List[Dict] = []

        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    print(f"   └── [Memory] Loading context from '{task_id}.md'...")
                    for line in f:
                        entry = self._md_to_entry(line)
                        if entry:
                            loaded_history.append(entry)

                if len(loaded_history) > HISTORY_MAX_SIZE:
                    loaded_history = loaded_history[-HISTORY_MAX_SIZE:]

            except OSError as e:
                print(f"   └── [Memory Error] Load failed: {e}")

        return loaded_history

    def add_log_to_specific_task(self, target_task_id: str, log_entry: Dict) -> None:
        """
        Write log entry to a specific task's memory, regardless of currently active task.
        This prevents context contamination during asynchronous VLM processing.
        """
        safe_id = "".join(c for c in target_task_id if c.isalnum() or c in (' ', '_', '-')).strip()
        if not safe_id:
            safe_id = "General_Task"

        # Acquire lock to ensure atomic check-and-write operations
        with self._lock:
            # Case 1: Target is current active task - update in-memory buffer
            if safe_id == self.current_task_id:
                self.active_history.append(log_entry)
                if len(self.active_history) > HISTORY_MAX_SIZE:
                    self.active_history.pop(0)
                print(f"   └── [Memory] Updated active context: {safe_id}")

            # Case 2: Target is a background task - append directly to file
            else:
                file_path = self.storage_path / f"{safe_id}.md"
                try:
                    # Generate Markdown line from entry
                    timestamp = log_entry.get('timestamp', '00:00')
                    etype = log_entry.get('type', 'INFO')
                    content = log_entry.get('content', '')
                    # Remove any existing app tags to prevent duplication
                    content = re.sub(r'➣ \[[^\]]+\]\s*', '', content)

                    # Extract window title for better traceability
                    context = log_entry.get('context', {})
                    title = context.get('title', '')
                    title_str = f" | {title}" if title else ""

                    line = f"- **[{etype}]** ({timestamp}): ➣ [{safe_id}]{title_str} {content}\n"

                    with open(file_path, 'a', encoding='utf-8') as f:
                        f.write(line)
                    print(f"   └── [Memory] Append to background task: {safe_id}.md")
                except OSError as e:
                    print(f"   └── [Memory Error] Background write failed: {e}")

    def update(self, log_entry: Dict) -> None:
        """Add new entry to the hot zone (兼容旧接口)."""
        self.add_log_to_specific_task(self.current_task_id, log_entry)

    def switch_task(self, new_task_id: str, reason: str = "Context Switch") -> None:
        """Switch to a new task context."""
        safe_id = "".join(
            c for c in new_task_id if c.isalnum() or c in (' ', '_', '-')
        ).strip()
        if not safe_id:
            safe_id = "General_Task"

        if safe_id == self.current_task_id:
            return

        self.logger.log("TASK_SWITCH", f"Switch: {self.current_task_id} -> {safe_id} ({reason})")

        # Acquire lock to prevent VLM writes during task switch
        with self._lock:
            # Persist current task state before switching
            if self.current_task_id:
                self._persist_task(self.current_task_id)

            # Update current_task_id BEFORE loading new history to prevent context pollution
            old_task_id = self.current_task_id
            self.current_task_id = safe_id
            self.active_history = self._load_task(safe_id)

        print(f"   └── [Memory] Switched '{old_task_id}' -> '{safe_id}'")

    def get_context_summary(self) -> str:
        """
        Generate Markdown-formatted context summary for LLM injection.
        """
        if not self.active_history:
            return "> (No recent actions in this context)"

        lines = [f"# Current Task Context: {self.current_task_id}"]
        for h in self.active_history[-CONTEXT_HISTORY_LINES:]:
            line = self._entry_to_md(h)
            if len(line) > 500:
                line = line[:500] + "..."
            lines.append(line)

        return "\n".join(lines)

    def get_context_summary_for_task(self, task_id: str) -> str:
        """Get recent context summary for a specific task ID."""
        safe_id = "".join(c for c in task_id if c.isalnum() or c in (' ', '_', '-')).strip()
        if not safe_id:
            safe_id = "General_Task"

        with self._lock:
            if safe_id == self.current_task_id:
                history = list(self.active_history)
            else:
                history = self._load_task(safe_id)

        if not history:
            return f"> (No recent actions in task {safe_id})"

        lines = [f"# Current Task Context: {safe_id}"]
        for h in history[-CONTEXT_HISTORY_LINES:]:
            content = h.get('content', '').replace('\n', ' ')
            content = re.sub(r'➣ \[[^\]]+\]\s*', '', content)
            timestamp = h.get('timestamp', '00:00')
            etype = h.get('type', 'INFO')
            line = f"- **[{etype}]** ({timestamp}): ➣ [{safe_id}] {content}"
            if len(line) > 500:
                line = line[:500] + "..."
            lines.append(line)

        return "\n".join(lines)


class SemanticGatingLayer:
    """
    Vector Router with Dynamic Learning capabilities.
    """
    def __init__(self, logger: MarkdownStreamLogger):
        self.logger = logger
        self.enabled = Config.SEMANTIC_ENABLED and HAS_SEMANTIC_LIB
        self.route_embeddings: Dict[str, List[Tuple[str, np.ndarray]]] = {}

        if self.enabled:
            try:
                print(f">> [Router] Initializing embeddings model: {Config.EMBEDDING_MODEL}")
                ollama.embeddings(model=Config.EMBEDDING_MODEL, prompt="test")
                self._reindex_vectors()
                print(">> [Router] Semantic Vectors Ready.")
            except Exception as e:
                print(f">> [Router Error] Failed to init embeddings: {e}")
                self.enabled = False

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get numpy embedding vector from Ollama."""
        try:
            resp = ollama.embeddings(model=Config.EMBEDDING_MODEL, prompt=text)
            vec = resp.get('embedding')
            if vec:
                return np.array(vec, dtype=np.float32)
        except Exception as e:
            print(f"   └── [Embed Error] {e}")
        return None

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def _reindex_vectors(self) -> None:
        """Re-compute embeddings for all routes using Ollama."""
        self.route_embeddings = {}
        for action, sentences in Config.SEMANTIC_ROUTES.items():
            self.route_embeddings[action] = []
            if not sentences:
                continue

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
                print(f"   └── [🔍 Brain] Semantic Firewall updated with: '{phrase}'")
        except Exception as e:
            print(f"   └── [Error] Vector update failed: {e}")

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate tokens for System 1 (Embedding).
        Rule of thumb: 1 token ≈ 4 chars (English) or 0.7 chars (Chinese).
        Minimum cost is 1 token.
        """
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
                    "total_tokens": dynamic_cost
                }
        except Exception as e:
            print(f"   └── [Match Error] {e}")
        return None


class IntelligentContextRouter:
    """
    Stateful Brain with Self-Evolution & Adaptive Heartbeat.
    """
    def __init__(
        self,
        logger: MarkdownStreamLogger,
        block_manager: TaskBlockManager,
        audit_logger: Optional[AuditLogger] = None
    ):
        self.logger = logger
        self.memory = block_manager
        self.audit = audit_logger
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._current_future = None
        self.semantic_router = SemanticGatingLayer(logger)
        self.cache_path = Config.LOG_ROOT / "decision_cache.json"
        self.cache = self._load_cache()
        self._state_lock = threading.Lock()

        self.task_states: Dict[str, Dict[str, object]] = {}

    def _default_task_state(self, task_id: str) -> Dict[str, object]:
        return {
            "summary": f"Switched to {task_id}. Analyzing new context...",
            "current_app": task_id
        }

    def _ensure_task_state(self, task_id: str) -> Dict[str, object]:
        with self._state_lock:
            if task_id not in self.task_states:
                self.task_states[task_id] = self._default_task_state(task_id)
            return dict(self.task_states[task_id])

    def get_working_summary(self, task_id: str) -> str:
        """Get task-local working summary for downstream VLM conditioning."""
        state = self._ensure_task_state(task_id)
        return str(state.get("summary", "Session started.")).strip()

    def reset_working_state(self, task_id: str) -> None:
        """
        Reset working_state when switching tasks.
        Prevents Brain's summary from carrying over content from previous task.
        """
        with self._state_lock:
            self.task_states[task_id] = self._default_task_state(task_id)
        print(f"   └── [Brain] Reset state for new task: {task_id}")

    def _load_cache(self) -> Dict:
        """Load decision cache from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    def _save_cache(self) -> None:
        """Save decision cache to disk."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    def _normalize_key(self, task_id: str, app: str, title: str) -> str:
        """Normalize cache key for window identification."""
        return f"{task_id} :: {app} :: {title.split(' - ')[0][:50]}"

    def _make_heavy_decision(
        self,
        task_id: str,
        app: str,
        title: str,
        io_stats: Dict[str, float],
        cache_key: str
    ) -> Dict:
        """Make decision using semantic router or LLM brain."""
        semantic_decision = self.semantic_router.match(app, title)
        if semantic_decision:
            self.cache[cache_key] = semantic_decision
            print(f"[🔍 Router] Semantic Hit: {semantic_decision['action']} ({semantic_decision['reason']})")

            # Audit log semantic router hit
            if self.audit:
                self.audit.log(
                    component="Brain",
                    event_type="SEMANTIC_HIT",
                    data={
                        "app": app,
                        "title": title,
                        "matched_phrase": semantic_decision.get("reason", ""),
                        "action": semantic_decision.get("action", "SKIP"),
                        "similarity_score": semantic_decision.get("reason", "").split("(")[-1].rstrip(")") if "(" in semantic_decision.get("reason", "") else "N/A",
                        "total_tokens": semantic_decision.get("total_tokens", 0)
                    }
                )

            return semantic_decision

        try:
            task_state = self._ensure_task_state(task_id)
            context_summary = self.memory.get_context_summary_for_task(task_id)

            prompt = Config.BRAIN_SYSTEM_PROMPT.format(
                summary=task_state['summary'],
                context=context_summary,
                app=app,
                title=title,
                stats=io_stats
            )

            res = ollama.generate(
                model=Config.BRAIN_MODEL,
                prompt=prompt,
                format="json",
                stream=False,
                options={"num_ctx": 4096, "temperature": 0.1, "num_predict": 500}
            )

            raw_response = res['response']
            decision = json.loads(raw_response)
            input_tokens = res.get('prompt_eval_count', 0)
            output_tokens = res.get('eval_count', 0)
            decision['total_tokens'] = input_tokens + output_tokens

            decision.setdefault("action", "SKIP")
            decision["source"] = "LLM_BRAIN"

            # Audit log LLM decision
            if self.audit:
                self.audit.log(
                    component="Brain",
                    event_type="LLM_DECISION",
                    data={
                        "app": app,
                        "title": title,
                        "model": Config.BRAIN_MODEL,
                        "prompt": prompt,
                        "raw_response": raw_response,
                        "parsed_decision": json.dumps(decision, indent=2, ensure_ascii=False),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                )

            if decision.get("updated_summary"):
                with self._state_lock:
                    state = self.task_states.get(task_id, self._default_task_state(task_id))
                    state["summary"] = decision["updated_summary"]
                    state["current_app"] = app
                    self.task_states[task_id] = state

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
                    print(f"   └── [🚨 Guardrail] Rejected bad pattern: '{phrase}'")

            if decision["action"] == "SKIP":
                self.cache[cache_key] = decision
                self._save_cache()

            current_summary = self.get_working_summary(task_id)
            print(f"[🧠 Brain] [{task_id}] {decision['action']} | Delay: {decision.get('next_check_delay', 5)}s | State: {current_summary[:40]}...")
            return decision

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[❌ Router Error] {e}")

            # Audit log error
            if self.audit:
                self.audit.log(
                    component="Brain",
                    event_type="LLM_ERROR",
                    data={
                        "app": app,
                        "title": title,
                        "error": str(e),
                        "model": Config.BRAIN_MODEL
                    }
                )

            return {"action": "SKIP", "source": "ERROR", "next_check_delay": 5, "total_tokens": 0}

    def decide_async(
        self,
        task_id: str,
        app: str,
        title: str,
        io_stats: Dict[str, float],
        callback_func: Callable[[Dict], None]
    ) -> None:
        """Make asynchronous decision with caching support."""
        cache_key = self._normalize_key(task_id, app, title)

        if cache_key in self.cache:
            decision = self.cache[cache_key]
            decision["source"] = "CACHE"
            decision["next_check_delay"] = Config.SAMPLE_INTERVAL
            callback_func(decision)
            return

        if self._current_future and not self._current_future.done():
            return

        self._current_future = self.executor.submit(
            self._make_heavy_decision, task_id, app, title, io_stats, cache_key
        )
        self._current_future.add_done_callback(lambda f: callback_func(f.result()))


class VisualHarvester:
    """Visual Language Model interface for screenshot analysis."""
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.last_thumb: Optional[Image.Image] = None
        self.diff_threshold = Config.VISUAL_DIFF_THRESHOLD  # e.g. 0.90
        self.audit = audit_logger

    def _compose_prompt(self, task_prompt: str) -> str:
        """Build final VLM prompt. Accept pre-formatted focus capsules."""
        cleaned = (task_prompt or "Analyze current visual activity").strip()
        if "[Focus]" in cleaned:
            return f"{Config.VLM_SYSTEM_PROMPT}\n{cleaned}"
        return f"{Config.VLM_SYSTEM_PROMPT}\n[Focus]: {cleaned}"

    def harvest(self, task_prompt: str, screenshot_image: Image.Image) -> str:
        """Analyze screenshot using VLM."""
        if screenshot_image is None:
            return "Error: No image."

        try:
            current_thumb = screenshot_image.resize(
                (DIFF_RESIZE_SIZE, DIFF_RESIZE_SIZE)
            ).convert("L")

            if self.last_thumb:
                diff = ImageChops.difference(current_thumb, self.last_thumb)
                diff_hist = diff.histogram()
                total_pixels = DIFF_RESIZE_SIZE * DIFF_RESIZE_SIZE
                unchanged_pixels = diff_hist[0]
                similarity_ratio = unchanged_pixels / total_pixels

                if similarity_ratio > self.diff_threshold:
                    # Audit log static skip
                    if self.audit:
                        self.audit.log(
                            component="Eye",
                            event_type="STATIC_SKIP",
                            data={
                                "task_prompt": task_prompt,
                                "similarity_ratio": f"{similarity_ratio:.4f}",
                                "threshold": f"{self.diff_threshold:.4f}",
                                "reason": "Screen unchanged - skipping VLM call",
                                "image_size": f"{screenshot_image.size}"
                            }
                        )
                    return "[STATIC] Screen unchanged."

            self.last_thumb = current_thumb

            img_to_send = screenshot_image.copy()
            if max(img_to_send.size) > VLM_MAX_DIMENSION:
                img_to_send.thumbnail((VLM_MAX_DIMENSION, VLM_MAX_DIMENSION))

            img_byte_arr = BytesIO()
            img_to_send.convert('RGB').save(img_byte_arr, format='JPEG', quality=85)

            full_prompt = self._compose_prompt(task_prompt)

            # Audit log VLM call start
            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_CALL_START",
                    data={
                        "model": Config.EYE_MODEL,
                        "task_prompt": task_prompt,
                        "full_prompt": full_prompt,
                        "image_size": f"{img_to_send.size}",
                        "image_bytes": len(img_byte_arr.getvalue())
                    }
                )

            res = ollama.generate(
                model=Config.EYE_MODEL,
                prompt=full_prompt,
                images=[img_byte_arr.getvalue()],
                stream=False,
                keep_alive="5m"
            )

            raw_response = res['response'].strip().replace("\n", " ")
            input_tokens = res.get('prompt_eval_count', 0)
            output_tokens = res.get('eval_count', 0)

            # Audit log VLM call complete
            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_CALL_COMPLETE",
                    data={
                        "model": Config.EYE_MODEL,
                        "task_prompt": task_prompt,
                        "raw_response": raw_response,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens
                    }
                )

            return raw_response

        except (ollama.ResponseError, IOError) as e:
            error_msg = f"Visual Error (Network/IO): {e}"
            # Audit log VLM error
            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_ERROR",
                    data={
                        "task_prompt": task_prompt,
                        "error_type": "Network/IO",
                        "error": str(e),
                        "model": Config.EYE_MODEL
                    }
                )
            return error_msg
        except Exception as e:
            error_msg = f"Visual Error (Unknown): {e}"
            # Audit log VLM unknown error
            if self.audit:
                self.audit.log(
                    component="Eye",
                    event_type="VLM_ERROR",
                    data={
                        "task_prompt": task_prompt,
                        "error_type": "Unknown",
                        "error": str(e),
                        "model": Config.EYE_MODEL
                    }
                )
            return error_msg

class Major_Tom_Recorder:
    """Main agent combining sensors, router, and harvester."""
    def __init__(self):
        Config.load_config()
        self.logger = MarkdownStreamLogger()
        self.audit_logger = AuditLogger()

        self.sensor = PlatformSensor()
        self.idle_sensor = IdleSensor()
        self.memory_manager = TaskBlockManager(self.logger)

        self.router = IntelligentContextRouter(self.logger, self.memory_manager, self.audit_logger)
        self.harvester = VisualHarvester(self.audit_logger)
        self.io_sensor = InputActivitySensor()
        self.observer = Observer()

        self.vlm_task_queue = queue.Queue(maxsize=1)
        self.pending_snapshot = None
        self.pending_lock = threading.Lock()
        self.last_app = ""
        self.last_title = ""
        self.last_vlm_time = 0.0
        self.is_away = False
        self.current_interval = Config.SAMPLE_INTERVAL
        self.current_task_id = "startup"
        self.last_file_path = ""
        self.last_file_time = 0.0
        self.task_streak_start_time: Optional[float] = None
        self.task_idle_accumulated: float = 0.0
        self.idle_started_at: Optional[float] = None

        self._init_file_monitor()
        threading.Thread(target=self._vlm_worker_loop, daemon=True).start()

    def _reset_task_duration_clock(self, now: float) -> None:
        """Reset duration clock when switching to a new task."""
        self.task_streak_start_time = now
        self.task_idle_accumulated = 0.0

    def _on_idle_start(self, now: float) -> None:
        """Mark idle segment start so it can be excluded from task duration."""
        if self.idle_started_at is None:
            self.idle_started_at = now

    def _on_idle_end(self, now: float) -> None:
        """Accumulate idle gap for the current task streak."""
        if self.idle_started_at is None:
            return
        if self.task_streak_start_time is not None:
            self.task_idle_accumulated += max(0.0, now - self.idle_started_at)
        self.idle_started_at = None

    def _get_task_duration_seconds(self, now: float) -> int:
        """Return active duration of unchanged current task, excluding idle time."""
        if self.task_streak_start_time is None:
            return 0
        effective = now - self.task_streak_start_time - self.task_idle_accumulated
        if self.idle_started_at is not None:
            effective -= max(0.0, now - self.idle_started_at)
        return max(0, int(effective))

    def _init_file_monitor(self) -> None:
        """Initialize file system watcher."""
        if Config.MONITOR_PATH.exists():
            class FileChangeHandler(FileSystemEventHandler):
                """Handler for file modification events."""
                def __init__(self, logger, manager):
                    self.logger = logger
                    self.manager = manager
                    self.last_mod = 0

                def on_modified(self, event) -> None:
                    """Handle file modification events."""
                    if event.is_directory or time.time() - self.last_mod < 1.0:
                        return
                    self.last_mod = time.time()

                    temp_extensions = ['.tmp', '.log', '.json', '.DS_Store', '.md']
                    if not any(x in event.src_path for x in temp_extensions):
                        entry = self.logger.log(
                            "FILE_MODIFIED",
                            f"Edited: {os.path.basename(event.src_path)}"
                        )
                        self.manager.update(entry)

            self.observer.schedule(
                FileChangeHandler(self.logger, self.memory_manager),
                str(Config.MONITOR_PATH),
                recursive=True
            )
            self.observer.start()

    def _vlm_worker_loop(self) -> None:
        """VLM consumer thread for visual analysis tasks."""
        while True:
            task = self.vlm_task_queue.get()
            # Extract source_task_id to ensure logs go to correct task file even if context switched
            prompt, app, title, screenshot, source_task_id = task
            try:
                if screenshot:
                    result = self.harvester.harvest(prompt, screenshot_image=screenshot)
                    if "[STATIC]" not in result:
                        entry = self.logger.log(
                            "VLM_ANALYSIS",
                            result,
                            context={"app": app, "title": title}
                        )
                        # Use targeted write to ensure log goes to correct task regardless of current context
                        self.memory_manager.add_log_to_specific_task(source_task_id, entry)
            except Exception as e:
                print(f">> [Worker Error] {e}")
            finally:
                self.vlm_task_queue.task_done()

    def _bucket_io_stats(self, io_stats: Optional[Dict[str, float]]) -> str:
        """Convert raw interaction stats into compact behavior hints."""
        if not io_stats:
            return "unknown"
        kpm = int(io_stats.get("kpm", 0))
        cpm = int(io_stats.get("cpm", 0))
        if kpm >= 80:
            typing = "very_high_typing"
        elif kpm >= 30:
            typing = "active_typing"
        elif kpm >= 5:
            typing = "light_typing"
        else:
            typing = "low_typing"

        if cpm >= 20:
            clicking = "high_clicking"
        elif cpm >= 8:
            clicking = "active_clicking"
        elif cpm >= 2:
            clicking = "light_clicking"
        else:
            clicking = "low_clicking"

        return f"{typing}, {clicking}, kpm={kpm}, cpm={cpm}"

    def _compact_recent_context(self, task_id: str) -> str:
        """Keep only the last few lines from the specific task memory."""
        summary = self.memory_manager.get_context_summary_for_task(task_id)
        if not summary:
            return "none"
        lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
        if not lines:
            return "none"
        tail = lines[-3:]
        compact = " | ".join(tail)
        return compact[:320]

    def _build_vlm_focus_capsule(
        self,
        decision: Dict,
        task_id: str,
        app: str,
        title: str,
        io_stats: Optional[Dict[str, float]]
    ) -> str:
        """Build a compact, evidence-aware focus prompt for VLM."""
        focus = str(decision.get("prompt", f"Analyze {app}")).strip()
        source = str(decision.get("source", "UNKNOWN"))
        reason = str(decision.get("reason", "")).strip() or "N/A"
        task_summary = self.router.get_working_summary(task_id)
        task_summary = task_summary[:180]
        recent_context = self._compact_recent_context(task_id)
        io_hint = self._bucket_io_stats(io_stats)

        # Surface explicit routing evidence when Layer 1 semantic router triggers SNAPSHOT.
        if source == "SEMANTIC":
            evidence = f"semantic_router_hit ({reason})"
        elif source == "CACHE":
            evidence = "cache_reuse_of_prior_decision"
        else:
            evidence = f"brain_decision ({reason})"

        return (
            f"[Focus]: {focus}\n"
            f"[TaskContext] summary={task_summary}\n"
            f"[RecentContext] {recent_context}\n"
            f"[Window] app={app}; title={title[:120]}\n"
            f"[IOHint] {io_hint}\n"
            f"[RoutingEvidence] {evidence}\n"
            "[Instruction] Ground the output in visible evidence and context. "
            "Do not describe generic UI layout. Output only 1-2 factual sentences about what the user is doing and why this snapshot matters."
        )

    def _on_router_decision(
        self,
        decision: Dict,
        task_id: str,
        title: str,
        region: Optional[Tuple],
        io_stats: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Handle router decision callback.

        Args:
            decision: The routing decision from the brain.
            task_id: Dynamic classified task ID (e.g., "Safari_Research").
            title: Current window title.
            region: Screen region coordinates.
        """
        if "next_check_delay" in decision:
            self.current_interval = max(1, min(int(decision["next_check_delay"]), 300))

        # Memory switching is handled in main loop; only process decision-specific logic here

        if decision.get("action") == "SNAPSHOT":
            captured_text = ContextTools.read_active_file(
                title, Config.MONITOR_PATH
            )

            if captured_text:
                if captured_text.startswith("## [FILE_READ]"):
                    match = re.search(r"Path: (.+?)\n", captured_text)
                    if match:
                        file_path = match.group(1)
                        now = time.time()
                        if file_path == self.last_file_path and now - self.last_file_time < 60:
                            return
                        self.last_file_path = file_path
                        self.last_file_time = now

                entry = self.logger.log(
                    "TEXT_SNAPSHOT",
                    captured_text,
                    {"app": task_id, "title": title}
                )
                self.memory_manager.update(entry)
                print(">> [Tool] Text context captured via API. Skipping Visual Analysis.")
                return

            # Store task_id for dynamic context routing
            with self.pending_lock:
                self.pending_snapshot = {
                    "decision": decision,
                    "app": task_id.split("_")[0] if "_" in task_id else task_id,  # Original app name
                    "task_id": task_id,  # Full task ID with scene classification
                    "title": title,
                    "region": region,
                    "io_stats": io_stats
                }

    def run(self) -> None:
        """Main execution loop."""
        print(f"SYSTEM ONLINE | Brain: {Config.BRAIN_MODEL} | Eye: {Config.EYE_MODEL}")
        print(f"Memory Mode: Markdown Stream | Storage: {Config.LOG_ROOT}")
        print("Press Ctrl+C to stop.\n")

        try:
            while True:
                try:
                    idle = self.idle_sensor.get_idle_duration()
                    if idle > Config.IDLE_THRESHOLD:
                        if not self.is_away:
                            self.logger.log("IDLE_START", f"Inactive > {int(idle)}s")
                            self.is_away = True
                            self._on_idle_start(time.time())
                        time.sleep(Config.SAMPLE_INTERVAL)
                        continue
                    if self.is_away:
                        self.logger.log("IDLE_END", "Resumed")
                        self._on_idle_end(time.time())
                        self.is_away = False
                        self.current_interval = Config.SAMPLE_INTERVAL

                    app, title, win_region = self.sensor.get_active_window()
                    # Always use short interval for activity detection; VLM cooldown is handled separately
                    io_stats = self.io_sensor.get_and_reset_stats(Config.SAMPLE_INTERVAL)
                    now = time.time()

                    if app and app != "Unknown":
                        # Dynamic context routing: classify sub-task based on window title
                        task_id = ContextClassifier.classify_task_id(app, title)

                        switched = (task_id != self.current_task_id)
                        # VLM cooldown only affects screenshot frequency, not window switch detection
                        vlm_cooldown = (now - self.last_vlm_time > Config.VLM_COOLDOWN)

                        # Always detect window switches with short interval
                        if switched:
                            self.logger.log("FOCUS_SWITCH", f"[{task_id}] {title}")
                            self.last_app = app
                            self.last_title = title

                            # Update memory context switch
                            if task_id != self.current_task_id:
                                self.memory_manager.switch_task(task_id)
                                self.current_task_id = task_id
                                # Reset Brain's working_state to prevent summary contamination from previous task
                                self.router.reset_working_state(task_id)
                                self._reset_task_duration_clock(now)

                            time.sleep(0.5)

                        if self.task_streak_start_time is None:
                            self._reset_task_duration_clock(now)

                        # Duration is task-local elapsed active time since this task became stable.
                        io_stats["duration"] = self._get_task_duration_seconds(now)

                        # VLM screenshot only executes after cooldown period
                        if switched or vlm_cooldown:
                            self.router.decide_async(
                                task_id,
                                app, title, io_stats,
                                callback_func=lambda d, tid=task_id, s=io_stats: self._on_router_decision(
                                    d, tid, title, win_region, s
                                )
                            )

                    task_to_run = None
                    with self.pending_lock:
                        if self.pending_snapshot:
                            if time.time() - self.last_vlm_time > Config.VLM_COOLDOWN:
                                task_to_run = self.pending_snapshot
                                self.pending_snapshot = None
                            elif not self.vlm_task_queue.empty():
                                self.pending_snapshot = None

                    if task_to_run and not self.vlm_task_queue.full():
                        try:
                            d = task_to_run["decision"]
                            capture_region = task_to_run.get("region")
                            if d.get("region_mode") == "FULL_SCREEN":
                                capture_region = None

                            screenshot = pyautogui.screenshot(region=capture_region)

                            # Use dynamically classified task_id as source for targeted logging
                            source_task_id = task_to_run['task_id']
                            vlm_prompt = self._build_vlm_focus_capsule(
                                d,
                                source_task_id,
                                task_to_run['app'],
                                task_to_run['title'],
                                task_to_run.get('io_stats')
                            )
                            self.vlm_task_queue.put_nowait((
                                vlm_prompt,
                                task_to_run['app'],  # Original app name for logging
                                task_to_run['title'],
                                screenshot,
                                source_task_id  # Full task ID with scene classification
                            ))
                            self.last_vlm_time = time.time()
                        except queue.Full:
                            print(">> [Info] VLM Queue full, dropping old frame.")
                        except OSError:
                            pass

                    # Always use short sampling interval to ensure timely window switch detection
                    time.sleep(Config.SAMPLE_INTERVAL)

                except Exception as inner_e:
                    print(f"!! [Runtime Error] {inner_e}")
                    time.sleep(5)

        except KeyboardInterrupt:
            pass
        except Exception as fatal_e:
            print(f"!! [Fatal Crash] {fatal_e}")
        finally:
            self.observer.stop()
            print("\n>> Saving Memories & Shutting down.")
            self.memory_manager._persist_task(self.memory_manager.current_task_id)


if __name__ == "__main__":
    Major_Tom_Recorder().run()
