"""Journal and memory viewing endpoints."""

import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query

from major_tom.config import Config
from major_tom.web.models import (
    JournalDay, JournalEntry, MemoryContent, MemoryFile, SearchResult,
    TimetableEntry,
)

router = APIRouter(prefix="/api/journal", tags=["journal"])


@router.get("/days", response_model=List[JournalDay])
def list_journal_days():
    """List available journal dates."""
    days = []
    for md_file in sorted(Config.LOG_ROOT.glob("*.md"), reverse=True):
        if md_file.stem == "decision_debug":
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        entry_count = content.count("### ")
        days.append(JournalDay(
            date=md_file.stem,
            file_path=str(md_file),
            entry_count=entry_count,
        ))
    return days


@router.get("/day/{date}", response_model=JournalEntry)
def get_journal_day(date: str):
    """Get parsed daily journal content."""
    file_path = Config.LOG_ROOT / f"{date}.md"
    if not file_path.resolve().is_relative_to(Config.LOG_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid date parameter")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"No journal for {date}")
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    return JournalEntry(date=date, content=content)


@router.get("/memories", response_model=List[MemoryFile])
def list_memories():
    """List task memory files."""
    memories = []
    for md_file in sorted(Config.MEMORY_ROOT.glob("*.md")):
        stat = md_file.stat()
        memories.append(MemoryFile(
            task_id=md_file.stem,
            file_path=str(md_file),
            last_modified=str(stat.st_mtime),
            size_bytes=stat.st_size,
        ))
    return memories


@router.get("/memory/{task_id}", response_model=MemoryContent)
def get_memory(task_id: str):
    """Get specific task memory content."""
    file_path = Config.MEMORY_ROOT / f"{task_id}.md"
    if not file_path.resolve().is_relative_to(Config.MEMORY_ROOT.resolve()):
        raise HTTPException(status_code=400, detail="Invalid task_id parameter")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"No memory for {task_id}")
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    return MemoryContent(task_id=task_id, content=content)


@router.get("/search", response_model=List[SearchResult])
def search_journals(q: str = Query(..., min_length=1)):
    """Full-text search across journals and memories."""
    results = []
    query_lower = q.lower()

    # Search journals
    for md_file in Config.LOG_ROOT.glob("*.md"):
        try:
            lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    results.append(SearchResult(
                        source="journal",
                        file=md_file.stem,
                        line_number=i + 1,
                        content=line.strip()[:200],
                        match=q,
                    ))
        except OSError:
            pass

    # Search memories
    for md_file in Config.MEMORY_ROOT.glob("*.md"):
        try:
            lines = md_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    results.append(SearchResult(
                        source="memory",
                        file=md_file.stem,
                        line_number=i + 1,
                        content=line.strip()[:200],
                        match=q,
                    ))
        except OSError:
            pass

    return results[:100]


# Regex patterns for parsing journal markdown entries
_ENTRY_RE = re.compile(
    r"^### (\d{2}:\d{2}) \| (Visual|Text|[\w_]+): (.+?)(?:\s+—\s+(.*))?$"
)
_TASK_SWITCH_RE = re.compile(
    r"^\*\*(\d{2}:\d{2}) (.+)\*\*$"
)
_IDLE_RE = re.compile(
    r"^> \*\*Away\*\* \((\d{2}:\d{2})\): (.*)$"
)
_FILE_RE = re.compile(
    r"^- \*(\d{2}:\d{2})\* File Edited: `(.*)`$"
)
_FOCUS_RE = re.compile(
    r"^- \*(\d{2}:\d{2})\* \[FOCUS_SWITCH\] (.*)$"
)


def _parse_entries_from_md(date: str, content: str) -> List[TimetableEntry]:
    """Parse a journal markdown file into structured timetable entries."""
    entries: List[TimetableEntry] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # ### HH:MM | Visual: App — Title  or  ### HH:MM | Text: App — Title
        m = _ENTRY_RE.match(line)
        if m:
            time_str, raw_type = m.group(1), m.group(2)
            app = m.group(3).strip()
            title = (m.group(4) or "").strip()
            entry_type = "VLM_ANALYSIS" if raw_type == "Visual" else "TEXT_SNAPSHOT"
            # Collect preview from next lines
            preview_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("### ") and not lines[i].startswith("---") and not lines[i].startswith("**"):
                stripped = lines[i].strip().lstrip("> ").strip()
                if stripped:
                    preview_lines.append(stripped)
                i += 1
            entries.append(TimetableEntry(
                date=date, time=time_str, entry_type=entry_type,
                app=app, title=title,
                preview=" ".join(preview_lines)[:200],
            ))
            continue

        # **HH:MM Task Switch...**
        m = _TASK_SWITCH_RE.match(line)
        if m:
            entries.append(TimetableEntry(
                date=date, time=m.group(1), entry_type="TASK_SWITCH",
                app="", title=m.group(2), preview=m.group(2),
            ))
            i += 1
            continue

        # > **Away** (HH:MM): ...
        m = _IDLE_RE.match(line)
        if m:
            entries.append(TimetableEntry(
                date=date, time=m.group(1), entry_type="IDLE_START",
                app="", title="Away", preview=m.group(2),
            ))
            i += 1
            continue

        # - *HH:MM* File Edited: `path`
        m = _FILE_RE.match(line)
        if m:
            entries.append(TimetableEntry(
                date=date, time=m.group(1), entry_type="FILE_MODIFIED",
                app="", title=m.group(2), preview=m.group(2),
            ))
            i += 1
            continue

        # - *HH:MM* [FOCUS_SWITCH] ...
        m = _FOCUS_RE.match(line)
        if m:
            entries.append(TimetableEntry(
                date=date, time=m.group(1), entry_type="FOCUS_SWITCH",
                app="", title=m.group(2), preview=m.group(2),
            ))
            i += 1
            continue

        i += 1

    return entries


@router.get("/timetable", response_model=List[TimetableEntry])
def get_timetable(start: str = Query(...), end: str = Query(...)):
    """Get structured timetable entries for a date range (inclusive).

    Query params: ?start=YYYY-MM-DD&end=YYYY-MM-DD
    """
    entries: List[TimetableEntry] = []
    for md_file in sorted(Config.LOG_ROOT.glob("*.md")):
        if md_file.stem == "decision_debug":
            continue
        date_str = md_file.stem  # expected YYYY-MM-DD
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            continue
        if date_str < start or date_str > end:
            continue
        try:
            content = md_file.read_text(encoding="utf-8", errors="ignore")
            entries.extend(_parse_entries_from_md(date_str, content))
        except OSError:
            pass
    return entries
