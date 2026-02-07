"""Generate synthetic or dataset-based traces for reproducible experiments.

Supports:
1. Synthetic generation with configurable activity patterns
2. BEHACOM dataset conversion (user behavior on computers)
3. Custom CSV/JSON import
"""

import csv
import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from major_tom.experiments.trace import TraceEvent

logger = logging.getLogger(__name__)


# Common application profiles for synthetic generation
APP_PROFILES = {
    "coding": {
        "apps": [
            ("VS Code", "main.py - project", (100, 100, 1400, 900)),
            ("VS Code", "utils.py - project", (100, 100, 1400, 900)),
            ("VS Code", "test_main.py - project", (100, 100, 1400, 900)),
            ("Terminal", "bash - ~/project", (100, 500, 800, 400)),
            ("Terminal", "python main.py", (100, 500, 800, 400)),
        ],
        "kpm_range": (80, 200),
        "cpm_range": (5, 30),
        "switch_prob": 0.05,
        "idle_prob": 0.02,
    },
    "research": {
        "apps": [
            ("Safari", "arxiv.org - Attention Is All You Need", (50, 50, 1500, 1000)),
            ("Safari", "arxiv.org - BERT: Pre-training", (50, 50, 1500, 1000)),
            ("Safari", "GitHub - transformers", (50, 50, 1500, 1000)),
            ("Preview", "paper.pdf", (200, 100, 1200, 900)),
            ("Zotero", "My Library", (100, 100, 1000, 800)),
            ("Notes", "Research Notes", (300, 200, 800, 600)),
        ],
        "kpm_range": (20, 80),
        "cpm_range": (10, 50),
        "switch_prob": 0.15,
        "idle_prob": 0.05,
    },
    "writing": {
        "apps": [
            ("Obsidian", "Daily Note.md", (100, 50, 1400, 950)),
            ("Obsidian", "Project Ideas.md", (100, 50, 1400, 950)),
            ("Word", "Document1.docx", (50, 50, 1500, 1000)),
            ("Safari", "Google Docs - Report", (50, 50, 1500, 1000)),
        ],
        "kpm_range": (60, 150),
        "cpm_range": (3, 15),
        "switch_prob": 0.03,
        "idle_prob": 0.08,
    },
    "entertainment": {
        "apps": [
            ("Safari", "YouTube - Video Title", (0, 0, 1600, 1000)),
            ("Safari", "bilibili - 视频标题", (0, 0, 1600, 1000)),
            ("Safari", "Netflix", (0, 0, 1600, 1000)),
            ("Spotify", "Now Playing", (100, 100, 400, 600)),
            ("Discord", "Server - Channel", (100, 100, 1200, 800)),
        ],
        "kpm_range": (5, 30),
        "cpm_range": (5, 40),
        "switch_prob": 0.08,
        "idle_prob": 0.1,
    },
    "mixed": {
        "apps": [
            ("VS Code", "app.py - backend", (100, 100, 1400, 900)),
            ("Safari", "Stack Overflow - Python error", (50, 50, 1500, 1000)),
            ("Safari", "Documentation - FastAPI", (50, 50, 1500, 1000)),
            ("Slack", "team-channel", (100, 100, 1000, 700)),
            ("Terminal", "git status", (100, 500, 800, 400)),
            ("Finder", "Downloads", (200, 200, 900, 600)),
        ],
        "kpm_range": (40, 150),
        "cpm_range": (10, 40),
        "switch_prob": 0.12,
        "idle_prob": 0.05,
    },
}


class SyntheticTraceGenerator:
    """Generate synthetic traces with configurable activity patterns."""

    def __init__(
        self,
        output_dir: str,
        duration_minutes: int = 60,
        sample_interval: float = 10.0,
        pattern: str = "mixed",
        seed: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.duration_minutes = duration_minutes
        self.sample_interval = sample_interval
        self.pattern = pattern
        self.seed = seed

        if seed is not None:
            random.seed(seed)

        self._trace_file = self.output_dir / "trace.jsonl"
        self._events: List[TraceEvent] = []

    def generate(self) -> int:
        """Generate a synthetic trace and save it."""
        profile = APP_PROFILES.get(self.pattern, APP_PROFILES["mixed"])
        apps = profile["apps"]
        kpm_min, kpm_max = profile["kpm_range"]
        cpm_min, cpm_max = profile["cpm_range"]
        switch_prob = profile["switch_prob"]
        idle_prob = profile["idle_prob"]

        total_samples = int((self.duration_minutes * 60) / self.sample_interval)
        current_app_idx = random.randint(0, len(apps) - 1)
        start_time = time.monotonic()

        logger.info(
            "Generating synthetic trace: %d minutes, pattern=%s, %d samples",
            self.duration_minutes, self.pattern, total_samples,
        )

        for i in range(total_samples):
            elapsed_ms = i * self.sample_interval * 1000

            # Decide if we switch apps
            if random.random() < switch_prob:
                current_app_idx = random.randint(0, len(apps) - 1)

            app_name, title, region = apps[current_app_idx]

            # Determine idle state
            idle_seconds = 0.0
            if random.random() < idle_prob:
                idle_seconds = random.uniform(30, 300)  # 30s to 5min idle

            # Generate activity stats (lower when idle)
            if idle_seconds > 0:
                kpm = 0.0
                cpm = random.uniform(0, 2)
            else:
                kpm = random.uniform(kpm_min, kpm_max)
                cpm = random.uniform(cpm_min, cpm_max)

            event = TraceEvent(
                timestamp=start_time + (elapsed_ms / 1000),
                elapsed_ms=elapsed_ms,
                app=app_name,
                title=title,
                region=region,
                kpm=round(kpm, 1),
                cpm=round(cpm, 1),
                idle_seconds=round(idle_seconds, 1),
                file_events=[],
                screenshot_path=None,
                ground_truth_description=f"{self.pattern}: {app_name}",
            )
            self._events.append(event)

        self._save()
        return len(self._events)

    def generate_scenario(self, scenario: List[Dict[str, Any]]) -> int:
        """Generate trace from a detailed scenario specification.

        Example scenario:
        [
            {"duration_min": 20, "pattern": "coding", "description": "Working on backend"},
            {"duration_min": 10, "pattern": "research", "description": "Reading papers"},
            {"duration_min": 5, "pattern": "entertainment", "description": "Break"},
            {"duration_min": 25, "pattern": "coding", "description": "Continue coding"},
        ]
        """
        start_time = time.monotonic()
        elapsed_ms = 0.0

        for segment in scenario:
            seg_duration_min = segment.get("duration_min", 10)
            seg_pattern = segment.get("pattern", "mixed")
            seg_description = segment.get("description", "")
            seg_samples = int((seg_duration_min * 60) / self.sample_interval)

            profile = APP_PROFILES.get(seg_pattern, APP_PROFILES["mixed"])
            apps = profile["apps"]
            current_app_idx = random.randint(0, len(apps) - 1)

            for _ in range(seg_samples):
                if random.random() < profile["switch_prob"]:
                    current_app_idx = random.randint(0, len(apps) - 1)

                app_name, title, region = apps[current_app_idx]

                idle_seconds = 0.0
                if random.random() < profile["idle_prob"]:
                    idle_seconds = random.uniform(30, 180)

                kpm_min, kpm_max = profile["kpm_range"]
                cpm_min, cpm_max = profile["cpm_range"]
                kpm = 0.0 if idle_seconds > 0 else random.uniform(kpm_min, kpm_max)
                cpm = random.uniform(0, 2) if idle_seconds > 0 else random.uniform(cpm_min, cpm_max)

                event = TraceEvent(
                    timestamp=start_time + (elapsed_ms / 1000),
                    elapsed_ms=elapsed_ms,
                    app=app_name,
                    title=title,
                    region=region,
                    kpm=round(kpm, 1),
                    cpm=round(cpm, 1),
                    idle_seconds=round(idle_seconds, 1),
                    file_events=[],
                    screenshot_path=None,
                    ground_truth_description=seg_description or f"{seg_pattern}: {app_name}",
                )
                self._events.append(event)
                elapsed_ms += self.sample_interval * 1000

        self._save()
        return len(self._events)

    def _save(self) -> None:
        """Save trace to file."""
        with open(self._trace_file, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        # Save metadata
        meta = {
            "event_count": len(self._events),
            "duration_seconds": len(self._events) * self.sample_interval,
            "pattern": self.pattern,
            "seed": self.seed,
            "generator": "synthetic",
            "generated_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Saved synthetic trace: %d events to %s", len(self._events), self._trace_file)


class BEHACOMConverter:
    """Convert BEHACOM dataset to Major Tom trace format.

    BEHACOM dataset: https://data.mendeley.com/datasets/cg4br62535/2

    Expected BEHACOM CSV columns (per-minute aggregated):
    - timestamp: Unix timestamp
    - app_name: Current foreground application
    - key_count: Number of keystrokes
    - mouse_clicks: Number of mouse clicks
    - cpu_percent: CPU usage
    - memory_percent: Memory usage
    """

    # Map common Windows app names to Mac equivalents for consistency
    APP_NAME_MAP = {
        "chrome.exe": "Chrome",
        "firefox.exe": "Firefox",
        "code.exe": "VS Code",
        "explorer.exe": "Finder",
        "notepad.exe": "Notes",
        "word.exe": "Word",
        "excel.exe": "Excel",
        "outlook.exe": "Mail",
        "slack.exe": "Slack",
        "discord.exe": "Discord",
        "spotify.exe": "Spotify",
        "powershell.exe": "Terminal",
        "cmd.exe": "Terminal",
        "windowsterminal.exe": "Terminal",
        "pycharm64.exe": "PyCharm",
        "idea64.exe": "IntelliJ IDEA",
    }

    def __init__(self, behacom_dir: str, output_dir: str, user_id: Optional[str] = None):
        self.behacom_dir = Path(behacom_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.user_id = user_id
        self._events: List[TraceEvent] = []

    def convert(self, max_events: Optional[int] = None) -> int:
        """Convert BEHACOM data to trace format."""
        # Find user data files
        if self.user_id:
            user_dir = self.behacom_dir / self.user_id
        else:
            # Use first available user
            user_dirs = [d for d in self.behacom_dir.iterdir() if d.is_dir()]
            if not user_dirs:
                raise FileNotFoundError(f"No user directories found in {self.behacom_dir}")
            user_dir = user_dirs[0]
            self.user_id = user_dir.name

        logger.info("Converting BEHACOM data for user: %s", self.user_id)

        # Find CSV files
        csv_files = sorted(user_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {user_dir}")

        start_time = time.monotonic()
        event_count = 0

        for csv_file in csv_files:
            try:
                events = self._parse_csv(csv_file, start_time, event_count)
                self._events.extend(events)
                event_count += len(events)

                if max_events and event_count >= max_events:
                    self._events = self._events[:max_events]
                    break
            except Exception as e:
                logger.warning("Failed to parse %s: %s", csv_file, e)
                continue

        self._save()
        return len(self._events)

    def _parse_csv(
        self, csv_file: Path, start_time: float, offset: int
    ) -> List[TraceEvent]:
        """Parse a single BEHACOM CSV file."""
        events = []

        with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
            # Try to detect format
            first_line = f.readline()
            f.seek(0)

            # BEHACOM may have different formats
            if "," in first_line:
                reader = csv.DictReader(f)
            else:
                reader = csv.DictReader(f, delimiter=";")

            for row in reader:
                try:
                    # Extract fields (handle different column name conventions)
                    timestamp = float(row.get("timestamp", row.get("time", 0)))
                    app = row.get("app_name", row.get("application", row.get("app", "Unknown")))
                    key_count = float(row.get("key_count", row.get("keystrokes", row.get("keys", 0))))
                    mouse_clicks = float(row.get("mouse_clicks", row.get("clicks", row.get("mouse", 0))))

                    # Normalize app name
                    app_lower = app.lower() if app else ""
                    app = self.APP_NAME_MAP.get(app_lower, app.split(".")[0] if "." in app else app)

                    # Convert to per-minute rates (BEHACOM uses 1-minute windows)
                    kpm = key_count
                    cpm = mouse_clicks

                    elapsed_ms = (offset + len(events)) * 60 * 1000  # 1-minute intervals

                    event = TraceEvent(
                        timestamp=start_time + (elapsed_ms / 1000),
                        elapsed_ms=elapsed_ms,
                        app=app or "Unknown",
                        title=f"{app} - Activity",
                        region=None,
                        kpm=round(kpm, 1),
                        cpm=round(cpm, 1),
                        idle_seconds=0.0 if kpm > 0 or cpm > 0 else 60.0,
                        file_events=[],
                        screenshot_path=None,
                        ground_truth_description=f"BEHACOM: {app}",
                    )
                    events.append(event)

                except (ValueError, KeyError) as e:
                    continue

        return events

    def _save(self) -> None:
        """Save converted trace."""
        trace_file = self.output_dir / "trace.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        meta = {
            "event_count": len(self._events),
            "duration_seconds": len(self._events) * 60,  # 1-minute intervals
            "source": "BEHACOM",
            "user_id": self.user_id,
            "generator": "behacom_converter",
            "converted_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Saved BEHACOM trace: %d events to %s", len(self._events), trace_file)


class GenericCSVConverter:
    """Convert generic CSV activity logs to trace format.

    Supports flexible column mapping for custom datasets.
    """

    def __init__(
        self,
        csv_path: str,
        output_dir: str,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default column mapping
        self.column_mapping = column_mapping or {
            "app": "app",
            "title": "title",
            "timestamp": "timestamp",
            "kpm": "kpm",
            "cpm": "cpm",
            "idle": "idle_seconds",
        }
        self._events: List[TraceEvent] = []

    def convert(self, max_events: Optional[int] = None) -> int:
        """Convert CSV to trace format."""
        start_time = time.monotonic()

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if max_events and i >= max_events:
                    break

                app = row.get(self.column_mapping.get("app", ""), "Unknown")
                title = row.get(self.column_mapping.get("title", ""), "")
                kpm = float(row.get(self.column_mapping.get("kpm", ""), 0))
                cpm = float(row.get(self.column_mapping.get("cpm", ""), 0))
                idle = float(row.get(self.column_mapping.get("idle", ""), 0))

                elapsed_ms = i * 10 * 1000  # Assume 10-second intervals

                event = TraceEvent(
                    timestamp=start_time + (elapsed_ms / 1000),
                    elapsed_ms=elapsed_ms,
                    app=app,
                    title=title or f"{app} - Activity",
                    region=None,
                    kpm=round(kpm, 1),
                    cpm=round(cpm, 1),
                    idle_seconds=idle,
                    file_events=[],
                    screenshot_path=None,
                    ground_truth_description="",
                )
                self._events.append(event)

        self._save()
        return len(self._events)

    def _save(self) -> None:
        """Save converted trace."""
        trace_file = self.output_dir / "trace.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")

        meta = {
            "event_count": len(self._events),
            "source": str(self.csv_path),
            "generator": "csv_converter",
            "converted_at": datetime.now().isoformat(),
        }
        with open(self.output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    """CLI entry point for trace generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate or convert traces for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 60-minute synthetic coding trace
  python -m major_tom.experiments.trace_generator \\
      --mode synthetic --output traces/coding_60m \\
      --duration 60 --pattern coding --seed 42

  # Generate trace from scenario
  python -m major_tom.experiments.trace_generator \\
      --mode scenario --output traces/mixed_session \\
      --scenario '[{"duration_min":30,"pattern":"coding"},{"duration_min":15,"pattern":"research"}]'

  # Convert BEHACOM dataset
  python -m major_tom.experiments.trace_generator \\
      --mode behacom --input ~/Downloads/BEHACOM \\
      --output traces/behacom_user1 --user user1

  # Convert generic CSV
  python -m major_tom.experiments.trace_generator \\
      --mode csv --input activity_log.csv \\
      --output traces/custom
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "scenario", "behacom", "csv"],
        required=True,
        help="Generation mode",
    )
    parser.add_argument("--output", required=True, help="Output directory for trace")
    parser.add_argument("--input", help="Input path (for behacom/csv modes)")
    parser.add_argument(
        "--duration", type=int, default=60, help="Duration in minutes (synthetic mode)"
    )
    parser.add_argument(
        "--pattern",
        choices=list(APP_PROFILES.keys()),
        default="mixed",
        help="Activity pattern (synthetic mode)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--scenario", help="JSON scenario specification (scenario mode)")
    parser.add_argument("--user", help="User ID (behacom mode)")
    parser.add_argument("--max-events", type=int, help="Maximum events to generate/convert")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.mode == "synthetic":
        gen = SyntheticTraceGenerator(
            output_dir=args.output,
            duration_minutes=args.duration,
            pattern=args.pattern,
            seed=args.seed,
        )
        count = gen.generate()

    elif args.mode == "scenario":
        if not args.scenario:
            parser.error("--scenario required for scenario mode")
        scenario = json.loads(args.scenario)
        gen = SyntheticTraceGenerator(output_dir=args.output, seed=args.seed)
        count = gen.generate_scenario(scenario)

    elif args.mode == "behacom":
        if not args.input:
            parser.error("--input required for behacom mode")
        conv = BEHACOMConverter(
            behacom_dir=args.input,
            output_dir=args.output,
            user_id=args.user,
        )
        count = conv.convert(max_events=args.max_events)

    elif args.mode == "csv":
        if not args.input:
            parser.error("--input required for csv mode")
        conv = GenericCSVConverter(csv_path=args.input, output_dir=args.output)
        count = conv.convert(max_events=args.max_events)

    logger.info("Generated trace with %d events", count)


if __name__ == "__main__":
    main()
