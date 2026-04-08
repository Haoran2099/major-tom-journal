"""File editing endpoints with hot reload support."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from major_tom.config import Config
from major_tom.web.event_bus import EventBus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/files", tags=["files"])

# Define allowed directories for security
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent


def _allowed_paths() -> Dict[str, Path]:
    """Resolve current editable roots from live configuration."""
    return {
        "configs": PROJECT_ROOT / "experiments" / "configs",
        "memory": Config.MEMORY_ROOT,
        "records": Config.LOG_ROOT,
        "config_json": PROJECT_ROOT / "config.json",
    }


def _display_path(path: Path) -> str:
    """Return project-relative path when possible, otherwise absolute path."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


class FileInfo(BaseModel):
    path: str
    name: str
    type: str  # "file" or "directory"
    size: int = 0
    modified: str = ""
    category: str = ""  # configs, memory, records


class FileContent(BaseModel):
    path: str
    content: str


class WriteRequest(BaseModel):
    path: str
    content: str


def _validate_path(path: str) -> Path:
    """Validate that path is within allowed directories."""
    allowed_paths = _allowed_paths()
    try:
        candidate = Path(path).expanduser()
        resolved = (candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")

    # Check if path is within allowed directories
    allowed = False
    for category, allowed_path in allowed_paths.items():
        if category == "config_json":
            if resolved == allowed_path.resolve():
                allowed = True
                break
        else:
            try:
                resolved.relative_to(allowed_path.resolve())
                allowed = True
                break
            except ValueError:
                continue

    if not allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Allowed paths: experiments/configs/, Memory/, Record/, config.json"
        )

    return resolved


def _get_category(path: Path) -> str:
    """Determine file category."""
    allowed_paths = _allowed_paths()
    try:
        path.resolve().relative_to(allowed_paths["configs"].resolve())
        return "configs"
    except ValueError:
        pass
    try:
        path.resolve().relative_to(allowed_paths["memory"].resolve())
        return "memory"
    except ValueError:
        pass
    try:
        path.resolve().relative_to(allowed_paths["records"].resolve())
        return "records"
    except ValueError:
        pass
    if path.resolve() == allowed_paths["config_json"].resolve():
        return "settings"
    return "unknown"


@router.get("/list", response_model=List[FileInfo])
def list_files(category: Optional[str] = None):
    """List editable files."""
    allowed_paths = _allowed_paths()
    files = []

    # Config files
    if category is None or category == "configs":
        configs_dir = allowed_paths["configs"]
        if configs_dir.exists():
            for f in sorted(configs_dir.glob("*.yaml")):
                stat = f.stat()
                files.append(FileInfo(
                    path=_display_path(f),
                    name=f.name,
                    type="file",
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    category="configs",
                ))

    # Memory files
    if category is None or category == "memory":
        memory_dir = allowed_paths["memory"]
        if memory_dir.exists():
            for f in sorted(memory_dir.glob("*.md")):
                stat = f.stat()
                files.append(FileInfo(
                    path=_display_path(f),
                    name=f.name,
                    type="file",
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    category="memory",
                ))

    # Record/log files
    if category is None or category == "records":
        records_dir = allowed_paths["records"]
        if records_dir.exists():
            for f in sorted(records_dir.glob("*.md"), reverse=True)[:30]:  # Last 30
                stat = f.stat()
                files.append(FileInfo(
                    path=_display_path(f),
                    name=f.name,
                    type="file",
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    category="records",
                ))

    # Main config.json
    if category is None or category == "settings":
        config_file = allowed_paths["config_json"]
        if config_file.exists():
            stat = config_file.stat()
            files.append(FileInfo(
                path="config.json",
                name="config.json",
                type="file",
                size=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                category="settings",
            ))

    return files


@router.get("/read", response_model=FileContent)
def read_file(path: str):
    """Read file content."""
    resolved = _validate_path(path)

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File is not a text file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    return FileContent(path=path, content=content)


@router.put("/write")
def write_file(request: WriteRequest):
    """Write file content with validation and hot reload trigger."""
    resolved = _validate_path(request.path)
    category = _get_category(resolved)

    # Validate content based on file type
    if resolved.suffix == ".yaml":
        try:
            import yaml
            yaml.safe_load(request.content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    elif resolved.suffix == ".json":
        try:
            json.loads(request.content)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Create backup
    if resolved.exists():
        backup_path = resolved.with_suffix(resolved.suffix + ".bak")
        try:
            backup_path.write_text(resolved.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass  # Backup failure shouldn't block write

    # Write file
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(request.content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing file: {e}")

    # Trigger hot reload for config files
    if category in ("configs", "settings"):
        _trigger_config_reload(resolved, category)

    # Publish event
    bus = EventBus()
    bus.publish("file_changed", {
        "path": request.path,
        "category": category,
        "timestamp": datetime.now().isoformat(),
    })

    return {
        "success": True,
        "path": request.path,
        "category": category,
        "message": f"File saved. {'Config reload triggered.' if category in ('configs', 'settings') else ''}",
    }


@router.delete("/delete")
def delete_file(path: str):
    """Delete a file (only memory and record files)."""
    resolved = _validate_path(path)
    category = _get_category(resolved)

    # Only allow deleting memory and record files
    if category not in ("memory", "records"):
        raise HTTPException(
            status_code=403,
            detail="Can only delete memory and record files"
        )

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    try:
        resolved.unlink()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")

    return {"success": True, "path": path, "deleted": True}


@router.post("/create")
def create_file(path: str = Body(...), content: str = Body("")):
    """Create a new file."""
    resolved = _validate_path(path)

    if resolved.exists():
        raise HTTPException(status_code=409, detail="File already exists")

    category = _get_category(resolved)

    # Validate content
    if resolved.suffix == ".yaml" and content:
        try:
            import yaml
            yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating file: {e}")

    return {
        "success": True,
        "path": path,
        "category": category,
    }


def _trigger_config_reload(path: Path, category: str):
    """Trigger configuration reload."""
    logger.info(f"Config reload triggered: {path}")

    # Reload main config if it's config.json
    if path.name == "config.json":
        try:
            from major_tom.config import Config
            Config.reload()
            logger.info("Main config reloaded")
        except Exception as e:
            logger.error(f"Failed to reload main config: {e}")

    # Publish reload event
    bus = EventBus()
    bus.publish("config_reload", {
        "path": str(path),
        "category": category,
        "timestamp": datetime.now().isoformat(),
    })


# Config hot reload endpoint
@router.post("/config/reload")
def force_config_reload():
    """Force reload all configurations."""
    try:
        from major_tom.config import Config
        Config.reload()

        bus = EventBus()
        bus.publish("config_reload", {
            "type": "full",
            "timestamp": datetime.now().isoformat(),
        })

        return {"success": True, "message": "Configuration reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")


@router.get("/config/current")
def get_current_config():
    """Get current runtime configuration."""
    try:
        config_path = _allowed_paths()["config_json"]
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
        return {}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading config: {e}")
