"""Metrics export utilities and LLM backend proxy for automatic instrumentation."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from major_tom.llm.base import EmbeddingResponse, LLMBackend, LLMResponse
from major_tom.metrics.collector import MetricsCollector
from major_tom.metrics.types import MetricCategory, MetricEvent

logger = logging.getLogger(__name__)


class MetricsCollectingBackend(LLMBackend):
    """Transparent proxy that records MetricEvents for every generate/embed call."""

    def __init__(self, wrapped: LLMBackend, collector: MetricsCollector):
        self._wrapped = wrapped
        self._collector = collector

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        images: Optional[List[bytes]] = None,
        format: Optional[str] = None,
        stream: bool = False,
        options: Optional[Dict[str, Any]] = None,
        keep_alive: Optional[str] = None,
    ) -> LLMResponse:
        res = self._wrapped.generate(
            model=model,
            prompt=prompt,
            images=images,
            format=format,
            stream=stream,
            options=options,
            keep_alive=keep_alive,
        )

        category = MetricCategory.VLM_CALL if images else MetricCategory.LLM_CALL
        component = "eye" if images else "brain"

        self._collector.record(
            MetricEvent(
                timestamp=datetime.now(),
                category=category,
                component=component,
                event_type="GENERATE",
                model=model,
                prompt_tokens=res.prompt_tokens,
                completion_tokens=res.completion_tokens,
                total_tokens=res.total_tokens,
                latency_ms=res.latency_ms,
            )
        )
        return res

    def embed(self, model: str, prompt: str) -> EmbeddingResponse:
        res = self._wrapped.embed(model=model, prompt=prompt)

        self._collector.record(
            MetricEvent(
                timestamp=datetime.now(),
                category=MetricCategory.EMBEDDING_CALL,
                component="semantic_router",
                event_type="EMBED",
                model=model,
                prompt_tokens=res.prompt_tokens,
                total_tokens=res.prompt_tokens,
                latency_ms=res.latency_ms,
            )
        )
        return res


def export_json(events: List[MetricEvent], path: Path) -> None:
    """Export metric events to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [e.to_dict() for e in events]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Exported %d events to %s", len(data), path)


def export_csv(events: List[MetricEvent], path: Path) -> None:
    """Export metric events to CSV."""
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp", "category", "component", "event_type", "model",
        "prompt_tokens", "completion_tokens", "total_tokens", "latency_ms",
        "action", "decision_source", "task_id", "is_task_switch",
        "experiment_id",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in events:
            row = e.to_dict()
            writer.writerow(row)
    logger.info("Exported %d events to %s", len(events), path)
