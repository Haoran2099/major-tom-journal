"""Tests for MetricsCollector."""

import threading
from datetime import datetime, timedelta

import pytest

from major_tom.metrics.collector import MetricsCollector
from major_tom.metrics.types import MetricCategory, MetricEvent


class TestMetricsCollector:
    def test_record_and_summary(self):
        collector = MetricsCollector(experiment_id="test")

        collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.LLM_CALL,
            component="brain",
            event_type="GENERATE",
            model="test",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=200.0,
        ))

        collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.DECISION,
            component="brain",
            event_type="DECISION",
            action="SNAPSHOT",
            decision_source="LLM_BRAIN",
            latency_ms=250.0,
        ))

        summary = collector.get_summary()
        assert summary["token_metrics"]["total_tokens"] == 150
        assert summary["token_metrics"]["brain_tokens_total"] == 150
        assert summary["routing_metrics"]["total_decisions"] == 1
        assert summary["routing_metrics"]["snapshot_count"] == 1

    def test_semantic_hit_tracking(self):
        collector = MetricsCollector()
        collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.DECISION,
            component="brain",
            event_type="DECISION",
            action="SKIP",
            decision_source="SEMANTIC",
            latency_ms=5.0,
        ))

        summary = collector.get_summary()
        assert summary["routing_metrics"]["semantic_hit_count"] == 1
        assert summary["routing_metrics"]["semantic_hit_rate"] == 1.0

    def test_thread_safety(self):
        collector = MetricsCollector()
        errors = []

        def record_many(thread_id):
            try:
                for i in range(50):
                    collector.record(MetricEvent(
                        timestamp=datetime.now(),
                        category=MetricCategory.EMBEDDING_CALL,
                        component="router",
                        event_type="EMBED",
                        total_tokens=10,
                        latency_ms=1.0,
                    ))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        summary = collector.get_summary()
        assert summary["token_metrics"]["embedding_tokens_total"] == 250 * 10

    def test_get_events_filter(self):
        collector = MetricsCollector()
        t1 = datetime.now()
        t2 = t1 + timedelta(minutes=5)

        collector.record(MetricEvent(
            timestamp=t1, category=MetricCategory.LLM_CALL,
            component="brain", event_type="GENERATE",
        ))
        collector.record(MetricEvent(
            timestamp=t2, category=MetricCategory.VLM_CALL,
            component="eye", event_type="GENERATE",
        ))

        llm_events = collector.get_events(category=MetricCategory.LLM_CALL)
        assert len(llm_events) == 1
        assert llm_events[0].component == "brain"

        eye_events = collector.get_events(component="eye")
        assert len(eye_events) == 1

    def test_time_series(self):
        collector = MetricsCollector()
        base = datetime.now()

        for i in range(5):
            collector.record(MetricEvent(
                timestamp=base + timedelta(seconds=i * 30),
                category=MetricCategory.EMBEDDING_CALL,
                component="router",
                event_type="EMBED",
                total_tokens=10,
            ))

        series = collector.get_time_series("total_tokens", interval_seconds=60)
        assert len(series) >= 1
        assert all("value" in s for s in series)

    def test_reset(self):
        collector = MetricsCollector()
        collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.LLM_CALL,
            component="brain", event_type="GENERATE",
            total_tokens=100,
        ))
        collector.reset()
        summary = collector.get_summary()
        assert summary["token_metrics"]["total_tokens"] == 0

    def test_empty_summary(self):
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["token_metrics"]["total_tokens"] == 0
        assert summary["routing_metrics"]["total_decisions"] == 0

    def test_percentile(self):
        assert MetricsCollector._percentile([1, 2, 3, 4, 5], 50) == 3.0
        assert MetricsCollector._percentile([], 50) == 0.0
        assert MetricsCollector._percentile([1], 95) == 1.0

    def test_vlm_tracking(self):
        collector = MetricsCollector()
        collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.VLM_CALL,
            component="eye",
            event_type="GENERATE",
            total_tokens=500,
            latency_ms=3000.0,
        ))
        collector.record(MetricEvent(
            timestamp=datetime.now(),
            category=MetricCategory.VLM_CALL,
            component="eye",
            event_type="STATIC_SKIP",
        ))

        summary = collector.get_summary()
        assert summary["routing_metrics"]["vlm_call_count"] == 2
        assert summary["routing_metrics"]["vlm_static_skip_count"] == 1
        assert summary["token_metrics"]["eye_tokens_total"] == 500
