"""Keyboard and mouse activity sensor."""

import threading
from typing import Dict

from pynput import keyboard, mouse


class InputActivitySensor:
    """Tracks keyboard and mouse activity rates."""

    def __init__(self):
        self._keystrokes = 0
        self._clicks = 0
        self._lock = threading.Lock()
        self.kb_listener = keyboard.Listener(on_press=self._on_press)
        self.mouse_listener = mouse.Listener(on_click=self._on_click)
        self.kb_listener.start()
        self.mouse_listener.start()

    def _on_press(self, _) -> None:
        with self._lock:
            self._keystrokes += 1

    def _on_click(self, *args) -> None:
        if len(args) >= 4 and args[3]:
            with self._lock:
                self._clicks += 1

    def get_and_reset_stats(self, duration_seconds: float) -> Dict[str, float]:
        """Return KPM/CPM stats and reset counters."""
        with self._lock:
            k, c = self._keystrokes, self._clicks
            self._keystrokes = 0
            self._clicks = 0
        factor = 60.0 / max(duration_seconds, 1.0)
        return {"kpm": int(k * factor), "cpm": int(c * factor)}

    def stop(self) -> None:
        """Stop keyboard and mouse listeners."""
        try:
            self.kb_listener.stop()
        except Exception:
            pass
        try:
            self.mouse_listener.stop()
        except Exception:
            pass
