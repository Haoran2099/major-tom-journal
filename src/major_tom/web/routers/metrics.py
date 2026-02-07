"""Token usage, latency, and decision analytics endpoints."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from major_tom.metrics.collector import MetricsCollector
from major_tom.web.models import MetricsSummary, TimeSeriesPoint

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/metrics", tags=["metrics"])

# Shared collector instance - set by app startup
_collector: Optional[MetricsCollector] = None


def set_collector(collector: MetricsCollector) -> None:
    global _collector
    _collector = collector


def _get_collector() -> MetricsCollector:
    if _collector is None:
        return MetricsCollector()
    return _collector


@router.get("/summary", response_model=MetricsSummary)
def get_summary():
    """Get current session metrics summary."""
    collector = _get_collector()
    return MetricsSummary(**collector.get_summary())


@router.get("/tokens", response_model=List[TimeSeriesPoint])
def get_token_series(interval: int = Query(60, description="Bucket interval in seconds")):
    """Get token usage time series."""
    collector = _get_collector()
    series = collector.get_time_series("total_tokens", interval)
    return [TimeSeriesPoint(**s) for s in series]


@router.get("/decisions", response_model=List[Dict[str, Any]])
def get_decisions(limit: int = Query(100)):
    """Get recent decision events."""
    from major_tom.metrics.types import MetricCategory
    collector = _get_collector()
    events = collector.get_events(category=MetricCategory.DECISION)
    return [e.to_dict() for e in events[-limit:]]


@router.get("/latency", response_model=List[TimeSeriesPoint])
def get_latency_series(interval: int = Query(60)):
    """Get latency time series."""
    collector = _get_collector()
    series = collector.get_time_series("latency_ms", interval)
    return [TimeSeriesPoint(**s) for s in series]
