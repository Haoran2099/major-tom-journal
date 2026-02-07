"""Thread-safe metrics collector with aggregation and export."""

import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from major_tom.metrics.types import MetricCategory, MetricEvent


class MetricsCollector:
    """Thread-safe metrics recording, aggregation, and export."""

    MAX_EVENTS = 50_000  # Prevent unbounded memory growth

    def __init__(self, experiment_id: Optional[str] = None):
        self.experiment_id = experiment_id
        self._events: List[MetricEvent] = []
        self._lock = threading.Lock()

        # Running counters
        self._total_tokens = 0
        self._brain_tokens = 0
        self._eye_tokens = 0
        self._embedding_tokens = 0
        self._total_decisions = 0
        self._snapshot_count = 0
        self._skip_count = 0
        self._semantic_hits = 0
        self._cache_hits = 0
        self._llm_brain_calls = 0
        self._vlm_calls = 0
        self._vlm_static_skips = 0
        self._task_switches = 0
        self._journal_entries = 0
        self._latencies: Dict[str, List[float]] = {
            "decision": [],
            "semantic": [],
            "brain": [],
            "vlm": [],
        }

    def record(self, event: MetricEvent) -> None:
        """Record a metric event and update running counters."""
        if self.experiment_id:
            event.experiment_id = self.experiment_id

        with self._lock:
            self._events.append(event)
            if len(self._events) > self.MAX_EVENTS:
                self._events = self._events[-self.MAX_EVENTS:]
            self._update_counters(event)

    def _update_counters(self, event: MetricEvent) -> None:
        self._total_tokens += event.total_tokens

        if event.category == MetricCategory.LLM_CALL:
            self._brain_tokens += event.total_tokens
            self._llm_brain_calls += 1
            self._latencies["brain"].append(event.latency_ms)
        elif event.category == MetricCategory.VLM_CALL:
            self._eye_tokens += event.total_tokens
            self._vlm_calls += 1
            self._latencies["vlm"].append(event.latency_ms)
            if event.event_type == "STATIC_SKIP":
                self._vlm_static_skips += 1
        elif event.category == MetricCategory.EMBEDDING_CALL:
            self._embedding_tokens += event.total_tokens

        if event.category == MetricCategory.DECISION:
            self._total_decisions += 1
            self._latencies["decision"].append(event.latency_ms)
            if event.action == "SNAPSHOT":
                self._snapshot_count += 1
            elif event.action == "SKIP":
                self._skip_count += 1
            if event.decision_source == "SEMANTIC":
                self._semantic_hits += 1
                self._latencies["semantic"].append(event.latency_ms)
            elif event.decision_source == "CACHE":
                self._cache_hits += 1

        if event.is_task_switch:
            self._task_switches += 1

        if event.category == MetricCategory.JOURNAL_ENTRY:
            self._journal_entries += 1

    def get_summary(self) -> Dict[str, Any]:
        """Return aggregated metrics summary."""
        with self._lock:
            total_decisions = max(self._total_decisions, 1)
            vlm_calls = max(self._vlm_calls, 1)
            journal_entries = max(self._journal_entries, 1)

            unique_tasks = set()
            for e in self._events:
                if e.task_id:
                    unique_tasks.add(e.task_id)

            return {
                "token_metrics": {
                    "total_tokens": self._total_tokens,
                    "brain_tokens_total": self._brain_tokens,
                    "eye_tokens_total": self._eye_tokens,
                    "embedding_tokens_total": self._embedding_tokens,
                    "tokens_per_decision": round(self._total_tokens / total_decisions, 1),
                    "tokens_per_journal_entry": round(self._total_tokens / journal_entries, 1),
                },
                "routing_metrics": {
                    "total_decisions": self._total_decisions,
                    "snapshot_count": self._snapshot_count,
                    "skip_count": self._skip_count,
                    "semantic_hit_count": self._semantic_hits,
                    "semantic_hit_rate": round(self._semantic_hits / total_decisions, 3),
                    "cache_hit_count": self._cache_hits,
                    "cache_hit_rate": round(self._cache_hits / total_decisions, 3),
                    "llm_brain_call_count": self._llm_brain_calls,
                    "llm_brain_call_rate": round(self._llm_brain_calls / total_decisions, 3),
                    "vlm_call_count": self._vlm_calls,
                    "vlm_static_skip_count": self._vlm_static_skips,
                    "vlm_effective_rate": round(
                        (self._vlm_calls - self._vlm_static_skips) / vlm_calls, 3
                    ),
                },
                "latency_metrics": {
                    "decision_avg_ms": self._avg(self._latencies["decision"]),
                    "decision_p50_ms": self._percentile(self._latencies["decision"], 50),
                    "decision_p95_ms": self._percentile(self._latencies["decision"], 95),
                    "semantic_avg_ms": self._avg(self._latencies["semantic"]),
                    "brain_avg_ms": self._avg(self._latencies["brain"]),
                    "vlm_avg_ms": self._avg(self._latencies["vlm"]),
                },
                "memory_metrics": {
                    "task_switch_count": self._task_switches,
                    "unique_task_ids": len(unique_tasks),
                },
            }

    def get_events(
        self,
        category: Optional[MetricCategory] = None,
        component: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[MetricEvent]:
        """Filter and return events."""
        with self._lock:
            result = list(self._events)

        if category:
            result = [e for e in result if e.category == category]
        if component:
            result = [e for e in result if e.component == component]
        if since:
            result = [e for e in result if e.timestamp >= since]
        if until:
            result = [e for e in result if e.timestamp <= until]
        return result

    def get_time_series(
        self, metric: str, interval_seconds: int = 60
    ) -> List[Dict[str, Any]]:
        """Aggregate a metric into time-bucketed series."""
        with self._lock:
            events = list(self._events)

        if not events:
            return []

        start = events[0].timestamp
        buckets: Dict[int, List[MetricEvent]] = {}
        for e in events:
            bucket_idx = int((e.timestamp - start).total_seconds()) // interval_seconds
            buckets.setdefault(bucket_idx, []).append(e)

        series = []
        for idx in sorted(buckets):
            bucket_events = buckets[idx]
            value = sum(getattr(e, metric, 0) for e in bucket_events)
            series.append({
                "bucket": idx,
                "offset_seconds": idx * interval_seconds,
                "value": value,
                "count": len(bucket_events),
            })
        return series

    def reset(self) -> None:
        """Clear all collected metrics."""
        with self._lock:
            self._events.clear()
            self._total_tokens = 0
            self._brain_tokens = 0
            self._eye_tokens = 0
            self._embedding_tokens = 0
            self._total_decisions = 0
            self._snapshot_count = 0
            self._skip_count = 0
            self._semantic_hits = 0
            self._cache_hits = 0
            self._llm_brain_calls = 0
            self._vlm_calls = 0
            self._vlm_static_skips = 0
            self._task_switches = 0
            self._journal_entries = 0
            for v in self._latencies.values():
                v.clear()

    @staticmethod
    def _avg(values: List[float]) -> float:
        return round(sum(values) / len(values), 1) if values else 0.0

    @staticmethod
    def _percentile(values: List[float], pct: int) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = int(len(s) * pct / 100)
        idx = min(idx, len(s) - 1)
        return round(s[idx], 1)
