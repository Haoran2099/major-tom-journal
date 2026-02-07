"""Journal and memory viewing endpoints."""

import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException, Query

from major_tom.config import Config
from major_tom.web.models import (
    JournalDay, JournalEntry, MemoryContent, MemoryFile, SearchResult,
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
