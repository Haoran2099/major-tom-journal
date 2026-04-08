"""FastAPI application for Major Tom Journal dashboard."""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from major_tom.config import Config
from major_tom.web.event_bus import EventBus
from major_tom.web.routers import journal, monitoring, metrics, files

logger = logging.getLogger(__name__)

# Load config so web endpoints see user's custom paths (LOG_ROOT, MEMORY_ROOT, models, etc.)
Config.load_config()

app = FastAPI(
    title="Major Tom Journal",
    description="AI-powered local activity journaling dashboard",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(journal.router)
app.include_router(monitoring.router)
app.include_router(metrics.router)
app.include_router(files.router)

# Wire EventBus to monitoring state
bus = EventBus()
bus.subscribe("status", monitoring.update_state)

# Serve static frontend files
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def serve_index():
    """Serve the dashboard index page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Major Tom Journal API", "docs": "/docs"}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
