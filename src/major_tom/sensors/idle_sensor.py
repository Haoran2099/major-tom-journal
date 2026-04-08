"""System idle time detection."""

import ctypes
import logging
import platform
import subprocess
import time

logger = logging.getLogger(__name__)


class IdleSensor:
    """Detects how long the user has been idle."""

    def __init__(self):
        self.os_type = platform.system()
        self._last_idle = 0.0
        self._last_check_time = 0.0
        self._cache_ttl = 5.0

    def get_idle_duration(self) -> float:
        """Return seconds since last user input."""
        now = time.time()
        if now - self._last_check_time < self._cache_ttl:
            return self._last_idle

        self._last_check_time = now
        try:
            if self.os_type == "Darwin":
                ioreg = subprocess.check_output(
                    ["ioreg", "-c", "IOHIDSystem"], stderr=subprocess.DEVNULL
                ).decode()
                output = ""
                for line in ioreg.splitlines():
                    if "HIDIdleTime" in line:
                        parts = line.strip().split()
                        if parts:
                            output = parts[-1]
                        break
                self._last_idle = int(output) / 1_000_000_000 if output else 0
            elif self.os_type == "Windows":
                class LastInputInfo(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

                lii = LastInputInfo()
                lii.cbSize = ctypes.sizeof(LastInputInfo)
                if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                    self._last_idle = (
                        ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                    ) / 1000.0
                else:
                    self._last_idle = 0.0
            else:
                self._last_idle = 0.0
        except (subprocess.SubprocessError, OSError, AttributeError):
            self._last_idle = 0.0

        return self._last_idle
