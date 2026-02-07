"""Real-time status and WebSocket monitoring endpoints."""

import asyncio
import json
import logging
from typing import Any, Dict

from typing import List

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from major_tom.web.event_bus import EventBus
from major_tom.web.models import CurrentStateResponse, StatusResponse
from major_tom.config import Config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])

# Shared state updated by the recorder via EventBus
_state: Dict[str, Any] = {
    "running": False,
    "current_task_id": "",
    "active_app": "",
    "active_title": "",
    "is_away": False,
    "last_decision": None,
    "last_decision_time": None,
    "kpm": 0.0,
    "cpm": 0.0,
    "start_time": None,
}


def update_state(event: Dict[str, Any]) -> None:
    """EventBus callback to update monitoring state."""
    data = event.get("data", {})
    for key in data:
        if key in _state:
            _state[key] = data[key]


@router.get("/status", response_model=StatusResponse)
def get_status():
    """Get system status."""
    import time
    uptime = 0.0
    if _state.get("start_time"):
        uptime = time.time() - _state["start_time"]

    return StatusResponse(
        running=_state.get("running", False),
        brain_model=Config.BRAIN_MODEL,
        eye_model=Config.EYE_MODEL,
        embedding_model=Config.EMBEDDING_MODEL,
        current_task_id=_state.get("current_task_id", ""),
        is_away=_state.get("is_away", False),
        uptime_seconds=uptime,
    )


@router.get("/current", response_model=CurrentStateResponse)
def get_current_state():
    """Get current active window and task state."""
    return CurrentStateResponse(
        active_app=_state.get("active_app", ""),
        active_title=_state.get("active_title", ""),
        current_task_id=_state.get("current_task_id", ""),
        last_decision=_state.get("last_decision"),
        last_decision_time=_state.get("last_decision_time"),
        kpm=_state.get("kpm", 0.0),
        cpm=_state.get("cpm", 0.0),
    )


@router.get("/recent")
def get_recent_events(limit: int = Query(50, le=100)):
    """Get recent events from the EventBus for the activity timeline."""
    bus = EventBus()
    events = bus.get_recent(limit=limit)
    # Filter out status/heartbeat events, keep only interesting ones
    visible = [e for e in events if e.get("type") not in ("status", "heartbeat")]
    return visible[-limit:]


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time event stream via WebSocket."""
    await websocket.accept()
    bus = EventBus()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def on_event(event):
        await event_queue.put(event)

    bus.subscribe_async("*", on_event)

    try:
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=30.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "data": {}})
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe("*", on_event)
