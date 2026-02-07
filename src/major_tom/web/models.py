"""Pydantic response models for the Web API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StatusResponse(BaseModel):
    running: bool = True
    brain_model: str = ""
    eye_model: str = ""
    embedding_model: str = ""
    current_task_id: str = ""
    is_away: bool = False
    uptime_seconds: float = 0.0


class CurrentStateResponse(BaseModel):
    active_app: str = ""
    active_title: str = ""
    current_task_id: str = ""
    last_decision: Optional[Dict[str, Any]] = None
    last_decision_time: Optional[str] = None
    kpm: float = 0.0
    cpm: float = 0.0


class JournalDay(BaseModel):
    date: str
    file_path: str
    entry_count: int = 0


class JournalEntry(BaseModel):
    date: str
    content: str


class MemoryFile(BaseModel):
    task_id: str
    file_path: str
    last_modified: Optional[str] = None
    size_bytes: int = 0


class MemoryContent(BaseModel):
    task_id: str
    content: str


class SearchResult(BaseModel):
    source: str  # "journal" or "memory"
    file: str
    line_number: int
    content: str
    match: str


class ExperimentConfigInfo(BaseModel):
    name: str
    dimension: str
    description: str
    path: str


class ExperimentResultInfo(BaseModel):
    name: str
    run_id: int
    trace: str
    started_at: str
    duration_seconds: float


class MetricsSummary(BaseModel):
    token_metrics: Dict[str, Any] = {}
    routing_metrics: Dict[str, Any] = {}
    latency_metrics: Dict[str, Any] = {}
    memory_metrics: Dict[str, Any] = {}


class TimeSeriesPoint(BaseModel):
    bucket: int
    offset_seconds: int
    value: float
    count: int
