"""Structured metric event types for experiment tracking."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MetricCategory(str, Enum):
    LLM_CALL = "LLM_CALL"
    VLM_CALL = "VLM_CALL"
    EMBEDDING_CALL = "EMBEDDING_CALL"
    DECISION = "DECISION"
    MEMORY_OP = "MEMORY_OP"
    JOURNAL_ENTRY = "JOURNAL_ENTRY"


@dataclass
class MetricEvent:
    """A single metric event with full context for experiment analysis."""

    timestamp: datetime
    category: MetricCategory
    component: str          # "brain" | "eye" | "semantic_router" | "cache" | "memory"
    event_type: str         # "GENERATE" | "EMBED" | "SEMANTIC_HIT" | "CACHE_HIT" | etc.

    # Token accounting
    model: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0

    # Decision tracking
    action: Optional[str] = None
    decision_source: Optional[str] = None

    # Memory tracking
    task_id: Optional[str] = None
    is_task_switch: bool = False
    source_task_id: Optional[str] = None

    # Trace correlation
    trace_event_idx: Optional[int] = None
    experiment_id: Optional[str] = None

    # Flexible payload
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "component": self.component,
            "event_type": self.event_type,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "action": self.action,
            "decision_source": self.decision_source,
            "task_id": self.task_id,
            "is_task_switch": self.is_task_switch,
            "source_task_id": self.source_task_id,
            "trace_event_idx": self.trace_event_idx,
            "experiment_id": self.experiment_id,
            "data": self.data,
        }
