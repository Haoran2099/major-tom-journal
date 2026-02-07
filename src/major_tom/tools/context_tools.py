"""Read-only tools to fetch context cheaply (CPU-only)."""

import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Optional

from major_tom.constants import FILE_READ_LIMIT

logger = logging.getLogger(__name__)


class ContextTools:
    """Read-only tools to fetch context cheaply (CPU-only)."""

    FILE_EXTENSIONS = "py|md|txt|json|js|ts|c|cpp|h|java|rs|go|tex|log"

    @staticmethod
    def read_active_file(window_title: str, monitor_path: Path) -> Optional[str]:
        """Attempt to find and read the file currently open in the active window."""
        match = re.search(
            rf"([^\\/:\*\?\"<>\|]+\.({ContextTools.FILE_EXTENSIONS}))",
            window_title,
            re.IGNORECASE,
        )
        if not match:
            return None

        target_filename = match.group(1)
        search_roots = [
            Path.cwd(),
            Path.home() / "Downloads",
            Path.home() / "Desktop",
            monitor_path,
        ]

        found_file = None
        for root in search_roots:
            if not root.exists():
                continue
            direct_path = root / target_filename
            if direct_path.exists():
                found_file = direct_path
                break
            if root == monitor_path:
                excluded_dirs = {
                    ".git", "node_modules", "build", "dist", "__pycache__", ".venv", "venv",
                }
                max_depth = 3
                for current_dir, dirs, files in os.walk(root, followlinks=True):
                    dirs[:] = [d for d in dirs if d not in excluded_dirs]
                    current_depth = current_dir.replace(str(root), "").count(os.sep)
                    if current_depth >= max_depth:
                        dirs[:] = []
                        continue
                    if target_filename in files:
                        found_file = Path(current_dir) / target_filename
                        break
                if found_file:
                    break

        if not found_file:
            logger.info("Filename '%s' detected, but file not found on disk.", target_filename)
            return None

        try:
            with open(found_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(FILE_READ_LIMIT)
            logger.info("Successfully read: %s", found_file.name)
            return f"## [FILE_READ] Path: {found_file}\n```\n{content}\n...\n```"
        except (OSError, UnicodeDecodeError) as e:
            logger.error("Read failed: %s", e)
            return None

    @staticmethod
    def read_clipboard() -> Optional[str]:
        """Get text content from clipboard safely."""
        try:
            if platform.system() == "Darwin":
                p = subprocess.check_output(["pbpaste"], stderr=subprocess.STDOUT)
                content = p.decode("utf-8").strip()
                if 5 < len(content) < 3000:
                    return f"## [CLIPBOARD]\n> {content}"
        except (subprocess.SubprocessError, OSError):
            pass
        return None
