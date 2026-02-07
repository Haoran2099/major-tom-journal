"""Activity trace recording and deterministic replay for experiment reproducibility."""

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """A single recorded sensor snapshot."""

    timestamp: float               # monotonic clock
    elapsed_ms: float              # ms since trace start
    app: str = ""                  # active window app
    title: str = ""                # window title
    region: Optional[Tuple[int, int, int, int]] = None
    kpm: float = 0.0
    cpm: float = 0.0
    idle_seconds: float = 0.0
    file_events: List[str] = field(default_factory=list)
    screenshot_path: Optional[str] = None
    ground_truth_description: str = ""


class TraceRecorder:
    """Records live sensor data to a JSONL trace file with screenshots."""

    def __init__(self, trace_dir: str):
        self.trace_dir = Path(trace_dir)
        self.screenshot_dir = self.trace_dir / "screenshots"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._trace_file = self.trace_dir / "trace.jsonl"
        self._start_time: Optional[float] = None
        self._event_count = 0

    def record_event(
        self,
        app: str,
        title: str,
        region: Optional[Tuple[int, int, int, int]],
        kpm: float,
        cpm: float,
        idle_seconds: float,
        file_events: Optional[List[str]] = None,
        screenshot: Optional[Image.Image] = None,
    ) -> TraceEvent:
        """Record a single sensor snapshot."""
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now

        elapsed = (now - self._start_time) * 1000

        screenshot_path = None
        if screenshot is not None:
            fname = f"{self._event_count:04d}_{int(time.time())}.png"
            spath = self.screenshot_dir / fname
            screenshot.save(spath)
            screenshot_path = str(spath.relative_to(self.trace_dir))

        event = TraceEvent(
            timestamp=now,
            elapsed_ms=elapsed,
            app=app,
            title=title,
            region=region,
            kpm=kpm,
            cpm=cpm,
            idle_seconds=idle_seconds,
            file_events=file_events or [],
            screenshot_path=screenshot_path,
        )

        with open(self._trace_file, "a", encoding="utf-8") as f:
            d = asdict(event)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

        self._event_count += 1
        return event

    def save_metadata(self, extra: Optional[Dict] = None) -> None:
        """Save session metadata."""
        meta = {
            "event_count": self._event_count,
            "trace_file": "trace.jsonl",
            "screenshots_dir": "screenshots/",
        }
        if self._start_time is not None:
            meta["duration_seconds"] = round(time.monotonic() - self._start_time, 1)
        if extra:
            meta.update(extra)

        with open(self.trace_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def run(self) -> None:
        """Record a trace from live sensors (entry point for --record-trace)."""
        import pyautogui
        from major_tom.config import Config
        from major_tom.sensors import IdleSensor, InputActivitySensor, PlatformSensor

        Config.load_config()
        sensor = PlatformSensor()
        idle_sensor = IdleSensor()
        io_sensor = InputActivitySensor()

        logger.info("Recording trace to %s (Ctrl+C to stop)", self.trace_dir)
        try:
            while True:
                app, title, region = sensor.get_active_window()
                idle = idle_sensor.get_idle_duration()
                stats = io_sensor.get_and_reset_stats(Config.SAMPLE_INTERVAL)

                try:
                    screenshot = pyautogui.screenshot()
                except OSError:
                    screenshot = None

                self.record_event(
                    app=app,
                    title=title,
                    region=region,
                    kpm=stats["kpm"],
                    cpm=stats["cpm"],
                    idle_seconds=idle,
                    screenshot=screenshot,
                )
                time.sleep(Config.SAMPLE_INTERVAL)
        except KeyboardInterrupt:
            pass
        finally:
            self.save_metadata()
            logger.info("Trace saved: %d events", self._event_count)


class TraceReplayer:
    """Replaces live sensors with recorded trace data for deterministic replay."""

    def __init__(self, trace_dir: str | Path):
        self.trace_dir = Path(trace_dir)
        self.events: List[TraceEvent] = []
        self.cursor = 0
        self._load()

    def _load(self) -> None:
        trace_file = self.trace_dir / "trace.jsonl"
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")

        with open(trace_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    region = d.get("region")
                    if region and isinstance(region, list):
                        region = tuple(region)
                    self.events.append(
                        TraceEvent(
                            timestamp=d["timestamp"],
                            elapsed_ms=d["elapsed_ms"],
                            app=d.get("app", ""),
                            title=d.get("title", ""),
                            region=region,
                            kpm=d.get("kpm", 0),
                            cpm=d.get("cpm", 0),
                            idle_seconds=d.get("idle_seconds", 0),
                            file_events=d.get("file_events", []),
                            screenshot_path=d.get("screenshot_path"),
                            ground_truth_description=d.get("ground_truth_description", ""),
                        )
                    )
        logger.info("Loaded trace: %d events from %s", len(self.events), self.trace_dir)

    def _current(self) -> TraceEvent:
        if self.cursor >= len(self.events):
            return self.events[-1] if self.events else TraceEvent(timestamp=0, elapsed_ms=0)
        return self.events[self.cursor]

    def _advance(self) -> TraceEvent:
        event = self._current()
        if self.cursor < len(self.events):
            self.cursor += 1
        return event

    def get_active_window(self) -> Tuple[str, str, Optional[Tuple[int, int, int, int]]]:
        """Return next recorded window state."""
        event = self._current()
        return event.app, event.title, event.region

    def get_idle_duration(self) -> float:
        """Return next recorded idle duration."""
        return self._current().idle_seconds

    def get_and_reset_stats(self, interval: float) -> Dict[str, float]:
        """Return next recorded IO stats and advance cursor."""
        event = self._advance()
        return {"kpm": event.kpm, "cpm": event.cpm}

    def get_screenshot(self) -> Optional[Image.Image]:
        """Return next recorded screenshot."""
        event = self._current()
        if event.screenshot_path:
            full_path = self.trace_dir / event.screenshot_path
            if full_path.exists():
                return Image.open(full_path)
        return None

    def has_next(self) -> bool:
        """Check if trace has more events."""
        return self.cursor < len(self.events)

    def reset(self) -> None:
        """Reset cursor to beginning."""
        self.cursor = 0
