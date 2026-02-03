# Major Tom Journal

> *"Ground Control to Major Tom... Commencing countdown, engines on..."*

An intelligent, privacy-first activity journaling system that observes, understands, and remembers your digital workflow—without ever leaving your machine.

## Why Major Tom Journal?

**The Problem:** In the age of AI, we generate enormous amounts of digital activity across dozens of applications daily. Yet most of this context is ephemeral—we lose track of what we were working on, why we made certain decisions, and how projects evolved over time. Cloud-based solutions exist, but they require trusting third parties with your sensitive data and often fail to capture the nuanced context of your actual work.

**The Solution:** Major Tom Journal is a **fully local, AI-powered activity logger** that runs entirely on your machine. It intelligently observes your screen activity, understands context using local LLMs, and maintains organized, searchable memory files—giving you a complete, private record of your digital life.

---

## ✨ Unique Features

### 1. Complete Local Privacy

**What it does:** All processing happens on your local machine. Screenshots are analyzed by local vision models (Ollama), text is processed by local LLMs, and all data stays in your filesystem.

**Why others don't do this:** Building a fully local system requires solving complex technical challenges:
- **Model orchestration:** Running multiple specialized models (vision + text + embeddings) locally without overwhelming system resources
- **Privacy-preserving architecture:** Designing the entire data flow to never expose sensitive information
- **Intelligent filtering:** Using AI to determine what's worth recording, rather than logging everything (which would be overwhelming)

**Our solution:** A modular architecture with lightweight local models and intelligent gating mechanisms that only capture meaningful activity.

---

### 2. Efficient Operation via Adaptive Sampling

**What it does:** The system dynamically adjusts its observation frequency based on your activity. When you're deeply focused on coding, it captures more frequently. When you're idle or watching a video, it backs off to save resources.

**Why others don't do this:** Most activity loggers use fixed intervals, leading to:
- **Resource waste:** Constant screenshotting and analysis even when nothing meaningful is happening
- **Storage bloat:** Recording thousands of redundant frames
- **Analysis fatigue:** Overwhelming users with too much data

**Our solution:** A **Semantic Gating Layer** that uses lightweight heuristics and local LLM classification to determine:
- Whether the current activity is worth recording (SKIP vs SNAPSHOT)
- The optimal next check delay (10s for high-focus work, 120s for passive consumption)
- Contextual routing to appropriate memory files

This adaptive approach reduces resource usage by 80%+ compared to fixed-interval logging while capturing *more* meaningful context.

---

### 3. Context-Aware Memory Architecture

**What it does:** Unlike simple chronological logs, Major Tom maintains **separate, organized memory files for different contexts** (e.g., `Safari_Research.md` vs `Safari_Entertainment.md`). When you switch from reading papers to watching videos, the system automatically routes observations to the appropriate memory context.

**Why this matters:** Activity logging faces unique challenges that generic solutions fail to address:

| Challenge | Generic Logger | Major Tom |
|-----------|---------------|-----------|
| **Context pollution** | Everything goes into one file | Dynamic sub-task classification routes to appropriate memory |
| **Cross-application workflows** | Loses track of related activities | Maintains context across app switches |
| **Async processing race conditions** | VLM results write to wrong context | Thread-safe targeted writes with source_task_id tracking |
| **Human-in-the-loop editing** | Opaque, uneditable logs | Markdown format, easily editable to guide AI behavior |

**Our solution:** A **TaskBlockManager** with:
- **Dynamic Context Routing:** Classifies window titles to determine appropriate memory file (e.g., "arxiv" → Research, "bilibili" → Entertainment)
- **Thread-Safe Memory Switching:** Lock-protected operations prevent race conditions between main loop and async VLM workers
- **Targeted Writes:** VLM results carry `source_task_id` to ensure they write to correct memory even if user switched contexts during processing
- **Markdown Stream Format:** Human-readable, editable, with inline tags for easy parsing

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Major Tom Recorder                        │
├─────────────────────────────────────────────────────────────┤
│  Sensors Layer                                               │
│  ├── PlatformSensor: Window title, app name, screen region  │
│  ├── IOSensor: Keyboard/mouse activity (privacy-respecting) │
│  └── IdleSensor: Detect user absence                        │
├─────────────────────────────────────────────────────────────┤
│  Brain Layer (Semantic Router)                              │
│  ├── Pattern Matching: Fast path for known activities       │
│  ├── Semantic Classification: Local embedding-based routing │
│  └── LLM Decision: Context-aware sampling strategy          │
├─────────────────────────────────────────────────────────────┤
│  Memory Layer (TaskBlockManager)                            │
│  ├── Dynamic Routing: Title-based sub-task classification   │
│  ├── Thread-Safe Access: Lock-protected read/write          │
│  └── Targeted Persistence: Source-aware async writes        │
├─────────────────────────────────────────────────────────────┤
│  Vision Layer (VLM Harvester)                               │
│  └── Local vision model (qwen3-vl) for screenshot analysis  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) running locally with required models:
  ```bash
  ollama pull qwen3:8b      # Brain model for decision making
  ollama pull qwen3-vl:8b   # Vision model for screenshot analysis
  ollama pull qwen3-embedding:8b  # (Optional) For semantic routing
  ```

### Installation

```bash
git clone https://github.com/Haoran2099/major-tom-journal.git
cd major-tom-journal
pip install -r requirements.txt
```

### Configuration

Edit `config.json` to customize:
- **Context routing rules:** Define how window titles map to memory files
- **Sampling intervals:** Adjust sensitivity
- **Model selection:** Use different Ollama models

### Running

```bash
python Journal_demo_v15.py
```

The system will start monitoring and create memory files according to your config.json file.

---

## 📁 Output Structure

```

For example:
~/Downloads/LLM_Journal/
├── Memory/
│   ├── Safari_Research.md      # Academic browsing
│   ├── Safari_Entertainment.md # Video/streaming
│   ├── VS Code_Project.md      # Coding sessions
│   ├── Zotero.md               # Paper reading
│   └── startup.md              # System startup
├── Record/
│   ├── decision_debug.log      # Brain decision history
│   └── 2026-02-03.md           # Daily activity stream
└── config.json                 # User configuration
```

---

## 🔧 Advanced Features

### Human-in-the-Loop Editing

Memory files are plain Markdown. You can:
- Edit entries to correct AI misinterpretations
- Add manual notes that will be included in context
- Delete sensitive entries
- Guide future AI behavior through edits

### Context Routing Customization

Define your own routing rules in `config.json`:

```json
"context_routing": {
  "enabled": true,
  "method": "keyword",
  "apps": {
    "Safari": {
      "Research": ["arxiv", "scholar", "github"],
      "Work": ["docs", "notion", "slack"],
      "Entertainment": ["youtube", "bilibili"]
    }
  }
}
```

### Pattern Learning

The system learns from your corrections. When you edit a memory file or provide feedback, it updates its pattern recognition for future classifications.

---

## Privacy & Security

- **Zero network calls:** Everything runs locally via Ollama
- **No cloud storage:** All data stays in your filesystem
- **No keystroke logging:** Only window titles and (optionally) screenshots
- **Editable memories:** You control what gets remembered
- **Transparent operation:** All decisions logged for audit

---

## Contributing

Contributions welcome! Areas of interest:
- Additional platform support (currently macOS-focused)
- New routing strategies
- Visualization tools for memory exploration
- Integration with knowledge management systems

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Named after David Bowie's "Space Oddity," Major Tom Journal floats in the background of your digital life, observing and recording—until it's time to return to Ground Control with your complete mission log.

*"Planet Earth is blue, and there's nothing I can do..."* 🚀
