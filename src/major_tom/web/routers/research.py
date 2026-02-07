"""User research endpoints: ESM, participants, and data export."""

import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from major_tom.web.event_bus import EventBus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/research", tags=["research"])

# Research data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research_data"
RESEARCH_DIR.mkdir(exist_ok=True)

# ESM default questions (professional template based on UbiComp standards)
DEFAULT_ESM_QUESTIONS = [
    {
        "id": "current_activity",
        "type": "text",
        "question": "What are you currently working on?",
        "question_zh": "你现在在做什么？",
        "required": True,
        "placeholder": "Briefly describe your current activity...",
    },
    {
        "id": "focus_level",
        "type": "slider",
        "question": "How focused are you right now?",
        "question_zh": "你现在的专注程度如何？",
        "required": True,
        "min": 1,
        "max": 5,
        "labels": ["Very distracted", "Somewhat distracted", "Neutral", "Somewhat focused", "Very focused"],
        "labels_zh": ["非常分心", "有些分心", "一般", "比较专注", "非常专注"],
    },
    {
        "id": "system_accuracy",
        "type": "radio",
        "question": "Is the system's recent recording accurate?",
        "question_zh": "系统最近的记录准确吗？",
        "required": True,
        "options": [
            {"value": "accurate", "label": "Accurate", "label_zh": "准确"},
            {"value": "partial", "label": "Partially accurate", "label_zh": "部分准确"},
            {"value": "inaccurate", "label": "Inaccurate", "label_zh": "不准确"},
            {"value": "not_noticed", "label": "Didn't notice", "label_zh": "没注意"},
        ],
    },
    {
        "id": "mood",
        "type": "radio",
        "question": "How would you describe your current mood?",
        "question_zh": "你现在的心情如何？",
        "required": False,
        "options": [
            {"value": "positive", "label": "Positive", "label_zh": "积极"},
            {"value": "neutral", "label": "Neutral", "label_zh": "一般"},
            {"value": "negative", "label": "Negative", "label_zh": "消极"},
        ],
    },
    {
        "id": "notes",
        "type": "text",
        "question": "Any additional notes? (optional)",
        "question_zh": "其他备注（可选）",
        "required": False,
        "placeholder": "E.g., why the recording was inaccurate...",
    },
]


class Participant(BaseModel):
    id: str
    name: str  # Display name (can be pseudonym)
    status: str = "active"  # active, paused, completed, withdrawn
    created_at: str
    consent_signed: bool = False
    esm_interval_hours: int = 4
    esm_enabled: bool = True
    notes: str = ""


class ESMResponse(BaseModel):
    id: str
    participant_id: str
    timestamp: str
    responses: Dict[str, Any]
    context: Dict[str, Any] = {}  # Current app, window title, etc.
    duration_seconds: float = 0  # How long to complete


class ESMConfig(BaseModel):
    enabled: bool = True
    interval_hours: int = 4
    questions: List[Dict[str, Any]] = []
    next_trigger: Optional[str] = None


# In-memory state (persisted to files)
_esm_config: Optional[ESMConfig] = None
_participants: Dict[str, Participant] = {}
_esm_responses: List[ESMResponse] = []


def _load_research_data():
    """Load research data from disk."""
    global _esm_config, _participants, _esm_responses

    # Load ESM config
    config_file = RESEARCH_DIR / "esm_config.json"
    if config_file.exists():
        data = json.loads(config_file.read_text())
        _esm_config = ESMConfig(**data)
    else:
        _esm_config = ESMConfig(questions=DEFAULT_ESM_QUESTIONS)

    # Load participants
    participants_file = RESEARCH_DIR / "participants.json"
    if participants_file.exists():
        data = json.loads(participants_file.read_text())
        _participants = {p["id"]: Participant(**p) for p in data}

    # Load responses
    responses_file = RESEARCH_DIR / "esm_responses.json"
    if responses_file.exists():
        data = json.loads(responses_file.read_text())
        _esm_responses = [ESMResponse(**r) for r in data]


def _save_research_data():
    """Save research data to disk."""
    RESEARCH_DIR.mkdir(exist_ok=True)

    # Save ESM config
    if _esm_config:
        (RESEARCH_DIR / "esm_config.json").write_text(
            json.dumps(_esm_config.dict(), indent=2, default=str)
        )

    # Save participants
    (RESEARCH_DIR / "participants.json").write_text(
        json.dumps([p.dict() for p in _participants.values()], indent=2, default=str)
    )

    # Save responses
    (RESEARCH_DIR / "esm_responses.json").write_text(
        json.dumps([r.dict() for r in _esm_responses], indent=2, default=str)
    )


# Initialize on module load
_load_research_data()


# ============== ESM Endpoints ==============

@router.get("/esm/config")
def get_esm_config():
    """Get ESM configuration."""
    return _esm_config.dict() if _esm_config else {}


@router.put("/esm/config")
def update_esm_config(config: ESMConfig):
    """Update ESM configuration."""
    global _esm_config
    _esm_config = config
    _save_research_data()
    return {"success": True, "config": config.dict()}


@router.get("/esm/questions")
def get_esm_questions():
    """Get ESM questions."""
    if _esm_config and _esm_config.questions:
        return _esm_config.questions
    return DEFAULT_ESM_QUESTIONS


@router.post("/esm/trigger")
def trigger_esm(participant_id: str = Body(default="default")):
    """Trigger an ESM popup for a participant."""
    # Publish ESM trigger event
    bus = EventBus()
    bus.publish("esm_trigger", {
        "participant_id": participant_id,
        "questions": _esm_config.questions if _esm_config else DEFAULT_ESM_QUESTIONS,
        "timestamp": datetime.now().isoformat(),
    })

    return {
        "success": True,
        "participant_id": participant_id,
        "triggered_at": datetime.now().isoformat(),
    }


@router.post("/esm/response")
def submit_esm_response(
    participant_id: str = Body(...),
    responses: Dict[str, Any] = Body(...),
    context: Dict[str, Any] = Body(default={}),
    duration_seconds: float = Body(default=0),
):
    """Submit an ESM response."""
    response = ESMResponse(
        id=f"esm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        participant_id=participant_id,
        timestamp=datetime.now().isoformat(),
        responses=responses,
        context=context,
        duration_seconds=duration_seconds,
    )

    _esm_responses.append(response)
    _save_research_data()

    # Publish event
    bus = EventBus()
    bus.publish("esm_response", response.dict())

    return {"success": True, "response_id": response.id}


@router.get("/esm/responses")
def get_esm_responses(
    participant_id: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 100,
):
    """Get ESM responses."""
    responses = _esm_responses

    if participant_id:
        responses = [r for r in responses if r.participant_id == participant_id]

    if since:
        responses = [r for r in responses if r.timestamp >= since]

    return responses[-limit:]


# ============== Participant Endpoints ==============

@router.get("/participants")
def list_participants():
    """List all participants."""
    return list(_participants.values())


@router.post("/participants")
def create_participant(
    name: str = Body(...),
    esm_interval_hours: int = Body(default=4),
):
    """Create a new participant."""
    participant_id = f"P{len(_participants) + 1:02d}"

    participant = Participant(
        id=participant_id,
        name=name,
        created_at=datetime.now().isoformat(),
        esm_interval_hours=esm_interval_hours,
    )

    _participants[participant_id] = participant
    _save_research_data()

    return participant.dict()


@router.get("/participants/{participant_id}")
def get_participant(participant_id: str):
    """Get participant details."""
    if participant_id not in _participants:
        raise HTTPException(status_code=404, detail="Participant not found")
    return _participants[participant_id].dict()


@router.put("/participants/{participant_id}")
def update_participant(participant_id: str, updates: Dict[str, Any] = Body(...)):
    """Update participant."""
    if participant_id not in _participants:
        raise HTTPException(status_code=404, detail="Participant not found")

    participant = _participants[participant_id]

    for key, value in updates.items():
        if hasattr(participant, key):
            setattr(participant, key, value)

    _save_research_data()
    return participant.dict()


@router.delete("/participants/{participant_id}")
def delete_participant(participant_id: str):
    """Delete participant (soft delete - marks as withdrawn)."""
    if participant_id not in _participants:
        raise HTTPException(status_code=404, detail="Participant not found")

    _participants[participant_id].status = "withdrawn"
    _save_research_data()

    return {"success": True, "status": "withdrawn"}


# ============== Export Endpoints ==============

@router.post("/export")
def export_data(
    anonymize_ids: bool = Body(default=True),
    remove_paths: bool = Body(default=True),
    include_screenshots: bool = Body(default=False),
    include_raw_logs: bool = Body(default=False),
    format: str = Body(default="json"),
    participant_ids: Optional[List[str]] = Body(default=None),
):
    """Export research data with anonymization options."""

    # Collect data
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "anonymization": {
            "ids_anonymized": anonymize_ids,
            "paths_removed": remove_paths,
        },
        "participants": [],
        "esm_responses": [],
        "journal_entries": [],
    }

    # ID mapping for anonymization
    id_map = {}

    def anonymize_id(original_id: str) -> str:
        if not anonymize_ids:
            return original_id
        if original_id not in id_map:
            # Create deterministic hash
            hash_val = hashlib.sha256(original_id.encode()).hexdigest()[:8]
            id_map[original_id] = f"ANON_{hash_val}"
        return id_map[original_id]

    def clean_text(text: str) -> str:
        if not remove_paths:
            return text
        # Remove file paths
        text = re.sub(r'/Users/[^\s]+', '[PATH_REMOVED]', text)
        text = re.sub(r'C:\\Users\\[^\s]+', '[PATH_REMOVED]', text)
        # Remove potential names/emails
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REMOVED]', text)
        return text

    # Export participants
    for p in _participants.values():
        if participant_ids and p.id not in participant_ids:
            continue
        if p.status == "withdrawn":
            continue

        export_data["participants"].append({
            "id": anonymize_id(p.id),
            "status": p.status,
            "created_at": p.created_at,
            "esm_interval_hours": p.esm_interval_hours,
        })

    # Export ESM responses
    for r in _esm_responses:
        if participant_ids and r.participant_id not in participant_ids:
            continue

        response_data = {
            "id": anonymize_id(r.id),
            "participant_id": anonymize_id(r.participant_id),
            "timestamp": r.timestamp,
            "responses": {},
            "duration_seconds": r.duration_seconds,
        }

        # Clean response text
        for key, value in r.responses.items():
            if isinstance(value, str):
                response_data["responses"][key] = clean_text(value)
            else:
                response_data["responses"][key] = value

        # Clean context
        if r.context:
            response_data["context"] = {
                k: clean_text(str(v)) if isinstance(v, str) else v
                for k, v in r.context.items()
            }

        export_data["esm_responses"].append(response_data)

    # Export journal entries if requested
    if include_raw_logs:
        records_dir = PROJECT_ROOT / "Record"
        if records_dir.exists():
            for md_file in sorted(records_dir.glob("*.md"))[-30:]:
                content = md_file.read_text(encoding="utf-8")
                export_data["journal_entries"].append({
                    "date": md_file.stem,
                    "content": clean_text(content),
                })

    # Save export
    export_dir = RESEARCH_DIR / "exports"
    export_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"export_{timestamp}"

    if format == "json":
        export_path = export_dir / f"{export_filename}.json"
        export_path.write_text(json.dumps(export_data, indent=2, ensure_ascii=False))
    else:
        # CSV format for responses
        import csv
        export_path = export_dir / f"{export_filename}_responses.csv"
        with open(export_path, "w", newline="", encoding="utf-8") as f:
            if export_data["esm_responses"]:
                writer = csv.DictWriter(f, fieldnames=["id", "participant_id", "timestamp", "duration_seconds"] +
                                        list(export_data["esm_responses"][0].get("responses", {}).keys()))
                writer.writeheader()
                for r in export_data["esm_responses"]:
                    row = {
                        "id": r["id"],
                        "participant_id": r["participant_id"],
                        "timestamp": r["timestamp"],
                        "duration_seconds": r["duration_seconds"],
                        **r.get("responses", {}),
                    }
                    writer.writerow(row)

    return {
        "success": True,
        "export_path": str(export_path),
        "records_count": {
            "participants": len(export_data["participants"]),
            "esm_responses": len(export_data["esm_responses"]),
            "journal_entries": len(export_data["journal_entries"]),
        },
    }


@router.get("/exports")
def list_exports():
    """List available exports."""
    export_dir = RESEARCH_DIR / "exports"
    if not export_dir.exists():
        return []

    exports = []
    for f in sorted(export_dir.iterdir(), reverse=True):
        if f.is_file():
            stat = f.stat()
            exports.append({
                "name": f.name,
                "path": str(f),
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

    return exports[:20]  # Last 20 exports
