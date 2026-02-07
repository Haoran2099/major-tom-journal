"""Ground truth annotation CLI tool for activity traces."""

import json
import logging
import sys
from pathlib import Path

from major_tom.experiments.trace import TraceReplayer

logger = logging.getLogger(__name__)


def annotate(trace_dir: str) -> None:
    """Interactive annotation of trace events with ground truth descriptions."""
    trace_path = Path(trace_dir)
    replayer = TraceReplayer(trace_path)
    output_file = trace_path / "ground_truth.jsonl"

    # Load existing annotations to allow resuming
    existing: dict[int, str] = {}
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line.strip())
                existing[d["event_idx"]] = d["description"]

    print(f"\nAnnotating trace: {trace_path}")
    print(f"Total events: {len(replayer.events)}")
    print(f"Already annotated: {len(existing)}")
    print("Enter a short description of what the user is doing.")
    print("Press Enter to skip, 'q' to quit.\n")

    annotations = []
    for idx, event in enumerate(replayer.events):
        if idx in existing:
            continue

        print(f"--- Event {idx + 1}/{len(replayer.events)} ---")
        print(f"  App:   {event.app}")
        print(f"  Title: {event.title}")
        print(f"  KPM:   {event.kpm}, CPM: {event.cpm}, Idle: {event.idle_seconds:.0f}s")
        if event.screenshot_path:
            print(f"  Screenshot: {event.screenshot_path}")

        try:
            desc = input("  Description> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping annotation.")
            break

        if desc.lower() == "q":
            break

        if desc:
            entry = {
                "event_idx": idx,
                "timestamp": event.timestamp,
                "app": event.app,
                "title": event.title,
                "description": desc,
            }
            annotations.append(entry)

    # Append new annotations
    if annotations:
        with open(output_file, "a", encoding="utf-8") as f:
            for entry in annotations:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nSaved {len(annotations)} annotations to {output_file}")
    else:
        print("\nNo new annotations.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m major_tom.experiments.annotate <trace_dir>")
        sys.exit(1)
    annotate(sys.argv[1])


if __name__ == "__main__":
    main()
