"""Main agent combining sensors, router, and harvester."""

import hashlib
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime
from typing import Dict, Optional, Tuple

import pyautogui
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from major_tom.brain.context_classifier import ContextClassifier
from major_tom.brain.context_router import IntelligentContextRouter
from major_tom.config import Config
from major_tom.llm.base import LLMBackend
from major_tom.llm.ollama_backend import OllamaBackend
from major_tom.memory.audit_logger import AuditLogger
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.memory.task_block_manager import TaskBlockManager
from major_tom.metrics.collector import MetricsCollector
from major_tom.metrics.exporters import MetricsCollectingBackend
from major_tom.metrics.types import MetricCategory, MetricEvent
from major_tom.sensors.idle_sensor import IdleSensor
from major_tom.sensors.input_sensor import InputActivitySensor
from major_tom.sensors.platform_sensor import PlatformSensor
from major_tom.tools.context_tools import ContextTools
from major_tom.vision.visual_harvester import VisualHarvester
from major_tom.web.event_bus import EventBus

logger = logging.getLogger(__name__)


class FileChangeHandler(FileSystemEventHandler):
    """Handler for file modification events."""

    def __init__(self, md_logger: MarkdownStreamLogger, manager: TaskBlockManager):
        self.logger = md_logger
        self.manager = manager
        self.last_mod = 0.0

    def on_modified(self, event) -> None:
        if event.is_directory or time.time() - self.last_mod < 1.0:
            return
        self.last_mod = time.time()

        temp_extensions = [".tmp", ".log", ".json", ".DS_Store", ".md"]
        if not any(x in event.src_path for x in temp_extensions):
            entry = self.logger.log(
                "FILE_MODIFIED",
                f"Edited: {os.path.basename(event.src_path)}",
            )
            self.manager.update(entry)


class Major_Tom_Recorder:
    """Main agent combining sensors, router, and harvester."""

    def __init__(self, llm_backend: Optional[LLMBackend] = None):
        Config.load_config()

        self.metrics_collector = MetricsCollector()
        raw_llm = llm_backend or OllamaBackend()
        self._llm = MetricsCollectingBackend(raw_llm, self.metrics_collector)

        self.md_logger = MarkdownStreamLogger()
        self.audit_logger = AuditLogger()

        self.sensor = PlatformSensor()
        self.idle_sensor = IdleSensor()
        self.memory_manager = TaskBlockManager(self.md_logger)
        self.classifier = ContextClassifier(self._llm)

        self.router = IntelligentContextRouter(
            self.md_logger, self.memory_manager, self._llm, self.audit_logger
        )
        self.harvester = VisualHarvester(self._llm, self.audit_logger)
        self.io_sensor = InputActivitySensor()
        self.observer = Observer()

        self.vlm_task_queue: queue.Queue = queue.Queue(maxsize=1)
        self._shutdown_event = threading.Event()
        self.pending_snapshot: Optional[Dict] = None
        self.pending_lock = threading.Lock()
        self.last_app = ""
        self.last_title = ""
        self.last_vlm_time = 0.0
        self.is_away = False
        self.current_interval = Config.SAMPLE_INTERVAL
        self.current_task_id = "startup"
        self._file_content_hashes: Dict[str, str] = {}  # file_path -> content_hash
        self.task_streak_start_time: Optional[float] = None
        self.task_idle_accumulated: float = 0.0
        self.idle_started_at: Optional[float] = None

        self._bus = EventBus()
        self._start_time = time.time()

        self._init_file_monitor()
        self._vlm_thread = threading.Thread(target=self._vlm_worker_loop, daemon=True)
        self._vlm_thread.start()

    def _reset_task_duration_clock(self, now: float) -> None:
        self.task_streak_start_time = now
        self.task_idle_accumulated = 0.0

    def _on_idle_start(self, now: float) -> None:
        if self.idle_started_at is None:
            self.idle_started_at = now

    def _on_idle_end(self, now: float) -> None:
        if self.idle_started_at is None:
            return
        if self.task_streak_start_time is not None:
            self.task_idle_accumulated += max(0.0, now - self.idle_started_at)
        self.idle_started_at = None

    def _get_task_duration_seconds(self, now: float) -> int:
        if self.task_streak_start_time is None:
            return 0
        effective = now - self.task_streak_start_time - self.task_idle_accumulated
        if self.idle_started_at is not None:
            effective -= max(0.0, now - self.idle_started_at)
        return max(0, int(effective))

    def _bucket_io_stats(self, io_stats: Optional[Dict[str, float]]) -> str:
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

        duration = int(io_stats.get("duration", 0))
        return f"{typing}, {clicking}, kpm={kpm}, cpm={cpm}, duration={duration}s"

    def _compact_recent_context(self, task_id: str) -> str:
        summary = self.memory_manager.get_context_summary_for_task(task_id)
        if not summary:
            return "none"
        lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
        if not lines:
            return "none"
        return " | ".join(lines[-3:])[:320]

    def _build_vlm_focus_capsule(
        self,
        decision: Dict,
        task_id: str,
        app: str,
        title: str,
        io_stats: Optional[Dict[str, float]],
    ) -> str:
        focus = str(decision.get("prompt", f"Analyze {app}")).strip()
        source = str(decision.get("source", "UNKNOWN"))
        reason = str(decision.get("reason", "")).strip() or "N/A"
        task_summary = self.router.get_working_summary(task_id)[:180]
        recent_context = self._compact_recent_context(task_id)
        io_hint = self._bucket_io_stats(io_stats)

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

    def _publish(self, event_type: str, data: Dict) -> None:
        """Publish an event to the EventBus (non-blocking, errors swallowed)."""
        try:
            self._bus.publish(event_type, data)
        except Exception:
            pass

    def _init_file_monitor(self) -> None:
        """Initialize file system watcher."""
        if Config.MONITOR_PATH.exists():
            self.observer.schedule(
                FileChangeHandler(self.md_logger, self.memory_manager),
                str(Config.MONITOR_PATH),
                recursive=True,
            )
            self.observer.start()

    def _vlm_worker_loop(self) -> None:
        """VLM consumer thread for visual analysis tasks."""
        while not self._shutdown_event.is_set():
            try:
                task = self.vlm_task_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            prompt, app, title, screenshot, source_task_id = task
            try:
                if screenshot:
                    result = self.harvester.harvest(prompt, screenshot_image=screenshot)
                    if "[STATIC]" not in result:
                        entry = self.md_logger.log(
                            "VLM_ANALYSIS",
                            result,
                            context={"app": app, "title": title},
                        )
                        self.memory_manager.add_log_to_specific_task(source_task_id, entry)
                        self.metrics_collector.record(MetricEvent(
                            timestamp=datetime.now(),
                            category=MetricCategory.JOURNAL_ENTRY,
                            component="eye",
                            event_type="VLM_ANALYSIS",
                            task_id=source_task_id,
                        ))
                        self._publish("journal_entry", {
                            "type": "VLM_ANALYSIS",
                            "task_id": source_task_id,
                            "app": app,
                            "title": title,
                            "preview": result[:200],
                        })
            except Exception as e:
                logger.error("VLM worker error: %s", e)
            finally:
                self.vlm_task_queue.task_done()

    def _on_router_decision(
        self,
        decision: Dict,
        task_id: str,
        title: str,
        region: Optional[Tuple],
        io_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        """Handle router decision callback."""
        self.metrics_collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.DECISION,
            component="brain",
            event_type=decision.get("action", "SKIP"),
            action=decision.get("action", "SKIP"),
            decision_source=decision.get("source", ""),
            total_tokens=decision.get("total_tokens", 0),
            task_id=task_id,
        ))
        self._publish("decision", {
            "action": decision.get("action"),
            "task_id": task_id,
            "title": title,
            "decision_source": decision.get("source", ""),
        })
        self._publish("status", {
            "last_decision": decision,
            "last_decision_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

        if "next_check_delay" in decision:
            self.current_interval = max(1, min(int(decision["next_check_delay"]), 300))

        if decision.get("action") == "SNAPSHOT":
            captured_text = ContextTools.read_active_file(title, Config.MONITOR_PATH)

            if captured_text:
                if captured_text.startswith("## [FILE_READ]"):
                    match = re.search(r"Path: (.+?)\n", captured_text)
                    if match:
                        file_path = match.group(1)
                        content_hash = hashlib.sha256(
                            captured_text.encode("utf-8")
                        ).hexdigest()[:16]
                        if self._file_content_hashes.get(file_path) == content_hash:
                            logger.debug("Skipping duplicate file: %s", file_path)
                            return
                        self._file_content_hashes[file_path] = content_hash

                entry = self.md_logger.log(
                    "TEXT_SNAPSHOT",
                    captured_text,
                    {"app": task_id, "title": title},
                )
                self.memory_manager.add_log_to_specific_task(task_id, entry)
                self.metrics_collector.record(MetricEvent(
                    timestamp=datetime.now(),
                    category=MetricCategory.JOURNAL_ENTRY,
                    component="brain",
                    event_type="TEXT_SNAPSHOT",
                    task_id=task_id,
                ))
                self._publish("journal_entry", {
                    "type": "TEXT_SNAPSHOT",
                    "task_id": task_id,
                    "title": title,
                    "preview": captured_text[:200],
                })
                logger.info("Text context captured via API. Skipping Visual Analysis.")
                return

            with self.pending_lock:
                self.pending_snapshot = {
                    "decision": decision,
                    "app": task_id.split("_")[0] if "_" in task_id else task_id,
                    "task_id": task_id,
                    "title": title,
                    "region": region,
                    "io_stats": io_stats,
                }

    def run(self) -> None:
        """Main execution loop."""
        logger.info(
            "SYSTEM ONLINE | Brain: %s | Eye: %s", Config.BRAIN_MODEL, Config.EYE_MODEL
        )
        logger.info("Memory Mode: Markdown Stream | Storage: %s", Config.LOG_ROOT)
        logger.info("Press Ctrl+C to stop.")

        self._publish("status", {
            "running": True,
            "start_time": self._start_time,
            "active_app": "",
            "active_title": "",
            "current_task_id": self.current_task_id,
            "is_away": False,
        })

        try:
            while True:
                try:
                    idle = self.idle_sensor.get_idle_duration()
                    if idle > Config.IDLE_THRESHOLD:
                        if not self.is_away:
                            self.md_logger.log("IDLE_START", f"Inactive > {int(idle)}s")
                            self.is_away = True
                            self._on_idle_start(time.time())
                            self._publish("status", {"is_away": True})
                        time.sleep(Config.SAMPLE_INTERVAL)
                        continue
                    if self.is_away:
                        self.md_logger.log("IDLE_END", "Resumed")
                        self._on_idle_end(time.time())
                        self.is_away = False
                        self.current_interval = Config.SAMPLE_INTERVAL
                        self._publish("status", {"is_away": False})

                    app, title, win_region = self.sensor.get_active_window()
                    io_stats = self.io_sensor.get_and_reset_stats(Config.SAMPLE_INTERVAL)
                    now = time.time()

                    if app and app != "Unknown":
                        task_id = self.classifier.classify_task_id(app, title)

                        switched = task_id != self.current_task_id
                        vlm_cooldown = now - self.last_vlm_time > Config.VLM_COOLDOWN

                        if switched:
                            self.md_logger.log("FOCUS_SWITCH", f"[{task_id}] {title}")
                            self.last_app = app
                            self.last_title = title

                            if task_id != self.current_task_id:
                                self.memory_manager.switch_task(task_id)
                                old_task = self.current_task_id
                                self.current_task_id = task_id
                                self.router.reset_working_state(task_id)
                                self._reset_task_duration_clock(now)
                                self.metrics_collector.record(MetricEvent(
                                    timestamp=datetime.now(),
                                    category=MetricCategory.MEMORY_OP,
                                    component="memory",
                                    event_type="TASK_SWITCH",
                                    task_id=task_id,
                                    is_task_switch=True,
                                ))
                                self._publish("task_switch", {
                                    "from_task": old_task,
                                    "to_task": task_id,
                                    "app": app,
                                    "title": title,
                                })

                            self._publish("status", {
                                "active_app": app,
                                "active_title": title,
                                "current_task_id": task_id,
                                "kpm": io_stats.get("kpm", 0),
                                "cpm": io_stats.get("cpm", 0),
                            })

                            time.sleep(0.5)

                        if self.task_streak_start_time is None:
                            self._reset_task_duration_clock(now)

                        io_stats["duration"] = self._get_task_duration_seconds(now)

                        if switched or vlm_cooldown:
                            self.router.decide_async(
                                task_id,
                                app,
                                title,
                                io_stats,
                                callback_func=lambda d, tid=task_id, s=io_stats: self._on_router_decision(
                                    d, tid, title, win_region, s
                                ),
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
                            source_task_id = task_to_run["task_id"]
                            vlm_prompt = self._build_vlm_focus_capsule(
                                d,
                                source_task_id,
                                task_to_run["app"],
                                task_to_run["title"],
                                task_to_run.get("io_stats"),
                            )
                            self.vlm_task_queue.put_nowait(
                                (
                                    vlm_prompt,
                                    task_to_run["app"],
                                    task_to_run["title"],
                                    screenshot,
                                    source_task_id,
                                )
                            )
                            self.last_vlm_time = time.time()
                        except queue.Full:
                            logger.info("VLM Queue full, dropping old frame.")
                        except OSError:
                            pass

                    time.sleep(Config.SAMPLE_INTERVAL)

                except Exception as inner_e:
                    logger.error("Runtime error: %s", inner_e)
                    time.sleep(5)

        except KeyboardInterrupt:
            pass
        except Exception as fatal_e:
            logger.critical("Fatal crash: %s", fatal_e)
        finally:
            self._shutdown_event.set()
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.io_sensor.stop()
            self.router.shutdown()
            self._vlm_thread.join(timeout=5.0)
            self._publish("status", {"running": False})
            logger.info("Saving Memories & Shutting down.")
            self.memory_manager._persist_task(self.memory_manager.current_task_id)
