"""Active window detection across platforms."""

import logging
import platform
import subprocess
import time
from typing import Optional, Tuple

from major_tom.constants import MIN_WINDOW_WIDTH

logger = logging.getLogger(__name__)


class PlatformSensor:
    """Detects the currently active window application and title."""

    def __init__(self):
        self._last_result: Tuple[str, str, Optional[Tuple[int, int, int, int]]] = (
            "Unknown",
            "Unknown",
            None,
        )
        self._last_check_time = 0.0
        self._cache_ttl = 1.0

    @staticmethod
    def _is_valid_title(title: str) -> bool:
        """Check if window title is meaningful (not placeholder)."""
        if not title or not title.strip():
            return False
        invalid_patterns = [
            "\u672a\u547d\u540d", "Untitled", "\u65e0\u6807\u9898",
            "new tab", "New Tab", "_blank",
        ]
        return not any(pattern in title for pattern in invalid_patterns)

    def get_active_window(
        self,
    ) -> Tuple[str, str, Optional[Tuple[int, int, int, int]]]:
        """Return (app_name, window_title, region_or_none)."""
        if time.time() - self._last_check_time < self._cache_ttl:
            return self._last_result

        system = platform.system()
        current_app, current_title, current_region = self._last_result

        try:
            if system == "Darwin":
                script = '''global frontApp, windowTitle, winPos, winSize
                tell application "System Events"
                    set frontApp to name of first application process whose frontmost is true
                    try
                        tell process frontApp
                            set windowTitle to value of attribute "AXTitle" of window 1
                            set winPos to value of attribute "AXPosition" of window 1
                            set winSize to value of attribute "AXSize" of window 1
                        end tell
                    on error
                        set windowTitle to ""
                        set winPos to {0, 0}
                        set winSize to {0, 0}
                    end try
                end tell
                return frontApp & " ||| " & windowTitle & " ||| " & (item 1 of winPos) & ", " & (item 2 of winPos) & " ||| " & (item 1 of winSize) & ", " & (item 2 of winSize)'''
                res = subprocess.check_output(
                    ["osascript", "-e", script], stderr=subprocess.STDOUT
                ).decode().strip()

                if "|||" in res:
                    parts = res.split(" ||| ")
                    fetched_app = parts[0]
                    fetched_title = parts[1].strip()

                    try:
                        x, y = map(int, parts[2].split(", "))
                        w, h = map(int, parts[3].split(", "))
                        region = (x, y, w, h) if w > MIN_WINDOW_WIDTH else None
                    except ValueError:
                        region = None

                    if fetched_app != current_app:
                        current_app = fetched_app
                        if self._is_valid_title(fetched_title):
                            current_title = fetched_title
                        else:
                            current_title = ""
                        current_region = region
                    elif self._is_valid_title(fetched_title):
                        current_title = fetched_title
                        current_region = region

            elif system == "Windows":
                import pygetwindow as gw

                win = gw.getActiveWindow()
                if win:
                    current_app = (
                        win.title.split(" - ")[-1]
                        if " - " in win.title
                        else "Windows App"
                    )
                    current_title = win.title
                    current_region = (win.left, win.top, win.width, win.height)

        except (subprocess.SubprocessError, OSError, ImportError):
            pass

        self._last_result = (current_app, current_title, current_region)
        self._last_check_time = time.time()
        return self._last_result
