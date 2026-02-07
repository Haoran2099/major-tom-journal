"""Experiment runner orchestrating trace replay, metrics collection, and evaluation."""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from major_tom.config import Config
from major_tom.experiments.ablation import AblationManager
from major_tom.experiments.config import ExperimentConfig
from major_tom.experiments.trace import TraceReplayer
from major_tom.llm.base import LLMBackend
from major_tom.llm.ollama_backend import OllamaBackend
from major_tom.memory.audit_logger import AuditLogger
from major_tom.memory.markdown_logger import MarkdownStreamLogger
from major_tom.metrics.collector import MetricsCollector
from major_tom.metrics.exporters import MetricsCollectingBackend, export_csv, export_json

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """Results from a single experiment run."""

    config_name: str
    dimension: str
    run_id: int
    trace_dir: str
    started_at: str = ""
    ended_at: str = ""
    duration_seconds: float = 0.0
    config_hash: str = ""
    models: Dict[str, str] = field(default_factory=dict)
    components: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    journal_entries: List[str] = field(default_factory=list)
    memory_files: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": {
                "name": self.config_name,
                "dimension": self.dimension,
                "run_id": self.run_id,
                "trace": self.trace_dir,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "duration_seconds": self.duration_seconds,
                "config_hash": self.config_hash,
            },
            "models": self.models,
            "components": self.components,
            **self.metrics,
        }


class ExperimentRunner:
    """Orchestrates experiment execution with trace replay and metrics collection."""

    def __init__(
        self,
        config_path: str,
        trace_dir: str,
        llm_backend: Optional[LLMBackend] = None,
        output_dir: Optional[str] = None,
    ):
        self.config = ExperimentConfig.load(config_path)
        self.trace_dir = trace_dir
        self._base_llm = llm_backend or OllamaBackend()
        self.output_dir = Path(output_dir or f"experiments/results/{self.config.name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Compute config hash for reproducibility
        with open(config_path, "rb") as f:
            self._config_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    def run_single(self, run_id: int) -> ExperimentResults:
        """Execute a single experiment run."""
        collector = MetricsCollector(experiment_id=f"{self.config.name}_run{run_id}")
        instrumented = MetricsCollectingBackend(self._base_llm, collector)

        # Apply ablation settings
        ablation = AblationManager(self.config)
        ablation.apply_to_config()

        # Build components
        md_logger = MarkdownStreamLogger()
        audit_logger = AuditLogger()
        memory_manager = ablation.get_memory_manager(md_logger)

        from major_tom.brain.context_classifier import ContextClassifier
        from major_tom.brain.context_router import IntelligentContextRouter
        from major_tom.vision.visual_harvester import VisualHarvester

        classifier = ContextClassifier(instrumented)
        router = IntelligentContextRouter(
            md_logger, memory_manager, instrumented, audit_logger
        )
        harvester = VisualHarvester(instrumented, audit_logger)

        # If semantic gating is disabled, replace with no-op
        if not self.config.semantic_gating_enabled:
            from major_tom.experiments.ablation import NoOpSemanticGating
            router.semantic_router = NoOpSemanticGating()

        # If cache is disabled, always return empty cache
        if not self.config.decision_cache_enabled:
            router.cache = {}
            router._save_cache = lambda: None

        # Load trace
        replayer = TraceReplayer(self.trace_dir)

        started = datetime.now()
        logger.info(
            "Starting run %d of %s (trace: %s)",
            run_id, self.config.name, self.trace_dir,
        )

        # Replay loop
        current_task_id = "startup"
        while replayer.has_next():
            app, title, region = replayer.get_active_window()
            idle = replayer.get_idle_duration()
            stats = replayer.get_and_reset_stats(Config.SAMPLE_INTERVAL)

            if idle > Config.IDLE_THRESHOLD:
                continue

            if app and app != "Unknown":
                task_id = classifier.classify_task_id(app, title)

                if task_id != current_task_id:
                    memory_manager.switch_task(task_id)
                    current_task_id = task_id
                    router.reset_working_state(task_id)

                # Synchronous decision (no async needed in replay)
                cache_key = router._normalize_key(app, title)
                if cache_key in router.cache and self.config.decision_cache_enabled:
                    decision = router.cache[cache_key]
                    decision["source"] = "CACHE"
                else:
                    decision = router._make_heavy_decision(app, title, stats, cache_key)

                from major_tom.metrics.types import MetricCategory, MetricEvent
                collector.record(MetricEvent(
                    timestamp=datetime.now(),
                    category=MetricCategory.DECISION,
                    component="brain",
                    event_type="DECISION",
                    action=decision.get("action"),
                    decision_source=decision.get("source"),
                    task_id=task_id,
                    is_task_switch=(task_id != current_task_id),
                    latency_ms=0,
                ))

                # VLM processing
                if (
                    decision.get("action") == "SNAPSHOT"
                    and ablation.should_use_vlm()
                ):
                    screenshot = replayer.get_screenshot()
                    if screenshot is not None:
                        result = harvester.harvest(
                            decision.get("prompt", "Analyze"), screenshot
                        )
                        if "[STATIC]" not in result:
                            entry = md_logger.log(
                                "VLM_ANALYSIS", result,
                                context={"app": app, "title": title},
                            )
                            memory_manager.add_log_to_specific_task(task_id, entry)
                            collector.record(MetricEvent(
                                timestamp=datetime.now(),
                                category=MetricCategory.JOURNAL_ENTRY,
                                component="eye",
                                event_type="VLM_ANALYSIS",
                                task_id=task_id,
                            ))

        ended = datetime.now()
        duration = (ended - started).total_seconds()

        # Persist final state
        memory_manager._persist_task(current_task_id)

        # Read journal entries
        journal_entries = self._read_journal_entries()
        memory_files = self._read_memory_files()

        results = ExperimentResults(
            config_name=self.config.name,
            dimension=self.config.dimension,
            run_id=run_id,
            trace_dir=str(self.trace_dir),
            started_at=started.isoformat(),
            ended_at=ended.isoformat(),
            duration_seconds=round(duration, 1),
            config_hash=f"sha256:{self._config_hash}",
            models={
                "brain": self.config.brain_model,
                "eye": self.config.eye_model,
                "embedding": self.config.embedding_model,
            },
            components={
                "semantic_gating": self.config.semantic_gating_enabled,
                "decision_cache": self.config.decision_cache_enabled,
                "vlm": self.config.vlm_enabled,
                "context_routing": self.config.context_routing_method,
                "adaptive_sampling": self.config.adaptive_sampling_enabled,
            },
            metrics=collector.get_summary(),
            journal_entries=journal_entries,
            memory_files=memory_files,
        )

        # Export
        if self.config.export_on_complete:
            run_dir = self.output_dir / f"run_{run_id}"
            run_dir.mkdir(parents=True, exist_ok=True)

            with open(run_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)

            events = collector.get_events()
            fmt = self.config.export_format
            if fmt in ("json", "both"):
                export_json(events, run_dir / "events.json")
            if fmt in ("csv", "both"):
                export_csv(events, run_dir / "events.csv")

        logger.info("Run %d complete: %d events, %.1fs", run_id, len(collector.get_events()), duration)
        return results

    def run_all(self) -> List[ExperimentResults]:
        """Execute all repetitions."""
        results = []
        for i in range(self.config.repeat):
            results.append(self.run_single(i))
        return results

    def _read_journal_entries(self) -> List[str]:
        """Read generated journal entries from log files."""
        entries = []
        for md_file in sorted(Config.LOG_ROOT.glob("*.md")):
            try:
                content = md_file.read_text(encoding="utf-8")
                entries.append(content)
            except OSError:
                pass
        return entries

    def _read_memory_files(self) -> Dict[str, str]:
        """Read memory files."""
        memories = {}
        for md_file in sorted(Config.MEMORY_ROOT.glob("*.md")):
            try:
                memories[md_file.stem] = md_file.read_text(encoding="utf-8")
            except OSError:
                pass
        return memories


def main():
    """CLI entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Major Tom experiments")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--trace", required=True, help="Path to trace directory")
    parser.add_argument("--output", help="Output directory for results")
    args = parser.parse_args()

    runner = ExperimentRunner(args.config, args.trace, output_dir=args.output)
    results = runner.run_all()
    logger.info("All runs complete: %d results", len(results))


if __name__ == "__main__":
    main()
