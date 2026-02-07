"""EventBus for decoupling recorder events from UI consumers."""

import asyncio
import logging
import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe publish/subscribe event bus bridging sync recorder and async web UI."""

    _instance: Optional["EventBus"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                inst._subscribers: Dict[str, List[Callable]] = defaultdict(list)
                inst._async_subscribers: Dict[str, List[tuple]] = defaultdict(list)
                inst._recent_events: List[Dict[str, Any]] = []
                inst._max_recent = 100
                inst._bus_lock = threading.Lock()
                inst._initialized = True
                cls._instance = inst
            return cls._instance

    def __init__(self):
        pass  # All initialization done in __new__ under lock

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe a sync callback to an event type."""
        with self._bus_lock:
            self._subscribers[event_type].append(callback)

    def subscribe_async(self, event_type: str, callback: Callable) -> None:
        """Subscribe an async callback to an event type.

        Captures the current running event loop so publish() can safely
        schedule coroutines from any thread.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._bus_lock:
            self._async_subscribers[event_type].append((callback, loop))

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Remove a subscriber."""
        with self._bus_lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
            self._async_subscribers[event_type] = [
                (cb, loop) for cb, loop in self._async_subscribers[event_type]
                if cb is not callback
            ]

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event (called from sync recorder thread)."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data,
        }

        with self._bus_lock:
            self._recent_events.append(event)
            if len(self._recent_events) > self._max_recent:
                self._recent_events.pop(0)

            sync_subs = list(self._subscribers.get(event_type, []))
            async_subs = list(self._async_subscribers.get(event_type, []))
            # Also include wildcard subscribers (avoid duplicates if event_type == "*")
            if event_type != "*":
                sync_subs += list(self._subscribers.get("*", []))
                async_subs += list(self._async_subscribers.get("*", []))

        for cb in sync_subs:
            try:
                cb(event)
            except Exception as e:
                logger.error("Sync subscriber error: %s", e)

        for cb, loop in async_subs:
            try:
                if loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(cb(event), loop)
                else:
                    # Fallback: try current thread's loop
                    try:
                        current_loop = asyncio.get_running_loop()
                        current_loop.create_task(cb(event))
                    except RuntimeError:
                        pass
            except Exception as e:
                logger.error("Async subscriber error: %s", e)

    def get_recent(self, event_type: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent events, optionally filtered by type."""
        with self._bus_lock:
            events = list(self._recent_events)
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]

    def clear(self) -> None:
        """Clear all events and subscribers."""
        with self._bus_lock:
            self._recent_events.clear()
            self._subscribers.clear()
            self._async_subscribers.clear()
