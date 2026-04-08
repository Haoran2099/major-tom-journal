# Major Tom Journal

An AI-powered desktop activity journaling system that runs entirely on your local machine. Major Tom observes your screen, detects context switches, and automatically generates structured activity journals — all without sending data to external servers.

## Features

- **Intelligent Context Routing** — A two-tier decision system (semantic gating + LLM reasoning) determines when to capture, skip, or snapshot your activity, minimizing redundant observations.
- **Visual Harvesting** — Captures and analyzes screen content using a local Vision-Language Model (VLM) to extract meaningful semantic summaries.
- **Task Block Organization** — Automatically groups related activities into task blocks with context-aware boundaries.
- **Markdown Journal Output** — Generates clean, readable daily journals in Markdown format.
- **Web Dashboard** — A built-in FastAPI web interface for browsing journals, monitoring status, and viewing metrics in real time.
- **100% Local & Private** — All processing happens on-device via [Ollama](https://ollama.com). No cloud APIs required.

## Architecture

```
src/major_tom/
  brain/        # Context classification, intelligent routing, semantic gating
  sensors/      # Platform (window/app), idle detection, input (keyboard/mouse)
  vision/       # Screen capture & VLM analysis
  memory/       # Task block management, Markdown logging, audit trail
  llm/          # LLM backends (Ollama, OpenAI-compatible, mock)
  metrics/      # Token usage, latency, and decision analytics
  web/          # FastAPI dashboard with real-time monitoring
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- macOS (primary platform; Linux support is experimental)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Pull required models via Ollama
ollama pull qwen3:8b
ollama pull qwen3-vl:8b
ollama pull qwen3-embedding:8b

# Run Major Tom (recorder + web dashboard)
python -m major_tom

# Or run recorder only (no web UI)
python -m major_tom --no-web

# Or run web dashboard only
python -m major_tom --web-only --port 8000
```

## Configuration

Edit `config.json` to customize:

- **paths** — Where to monitor files, store journals, and save memory
- **parameters** — Sampling interval, idle thresholds, VLM cooldown
- **models** — Which Ollama models to use for brain, vision, and embedding
- **semantic_router** — Route definitions for SKIP/SNAPSHOT decisions
- **context_routing** — Per-app keyword-based context classification

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
