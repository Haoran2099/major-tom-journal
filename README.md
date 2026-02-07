# Major Tom Journal

> *"Ground Control to Major Tom..."* — David Bowie, *Space Oddity*

**A fully local, AI-powered digital activity journal system**

Major Tom Journal passively observes your screen activities, leverages local large language models to understand your actions, and automatically organizes them into structured work memory files. All data remains strictly on your machine — no internet connection, no uploads, complete privacy.

---

## Core Problem

Every day, you switch between applications — coding, reading papers, browsing information, writing documents. But after a few days, you might wonder:

- What was I researching that afternoon?
- What was the solution to that bug?
- What was the title of that paper?

Major Tom Journal acts as your personal "**black box recorder**." It observes every window switch, every activity, and automatically generates timeline logs and structured memory files, enabling you to revisit your past work context at any time.

---

## Core Workflow

```
<<<<<<< HEAD
Your screen activity
    │
    ▼
┌──────────────────────────────────┐
│  Sensor Layer                    │  Detects current window, input activity, idle duration
│  Platform / Input / Idle Sensor  │
└────────────┬─────────────────────┘
             │
             ▼
┌──────────────────────────────────┐
│  Decision Brain (Router)         │  "Is this activity worth recording?"
│  Semantic Gating → Cache → LLM   │  Output: SNAPSHOT (record) or SKIP (ignore)
└────────────┬─────────────────────┘
             │
             ├─ SKIP ─→ [Ignore]
             │
             └─ SNAPSHOT ─→ Content Collection
                              │
                              ├─ Text Collection (file content edits)
                              └─ Screenshot Analysis (VLM visual extraction)
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │  Memory Layer            │
                         │  Task-isolated storage   │
                         │  Writes Markdown files   │
                         │  Safari_Research.md      │
                         │  VS Code_Project.md      │
                         └──────────────────────────┘
=======
┌─────────────────────────────────────────────────────────────┐
│                    Major Tom Recorder                       │
├─────────────────────────────────────────────────────────────┤
│  Sensors Layer                                              │
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
>>>>>>> ab3274d992d8714ebafbf4d07e549cdec11f4254
```

### Three Technical Highlights

| Dimension | Feature | Benefit |
|-----------|---------|---------|
| **Adaptive Sampling** | 10-second polling during code editing, 2-minute polling during video playback | Saves over 80% system resources |
| **Context Isolation** | Writes different tasks within the same app to separate memory files | Prevents information contamination |
| **Asynchronous VLM Thread Safety** | Screenshot analysis runs in the background, tracked via `source_task_id` | Fast response, no main loop blocking |

---

## Project Structure

```
major-tom-journal/
├── Major_Tom_Journal.py               # Legacy entry point for backward compatibility
├── pyproject.toml                     # Package management configuration
│
├── src/major_tom/                     # Core package
│   ├── __main__.py                    # Entry point for `python -m major_tom`
│   ├── config.py                      # Configuration management (loads from config.json)
│   ├── constants.py                   # Constant definitions
│   ├── recorder.py                    # Main loop orchestrator
│   │
│   ├── sensors/                       # [Sensor Layer]
│   │   ├── platform_sensor.py         #   Window tracking (app name, title, position)
│   │   ├── idle_sensor.py             #   Idle time detection (macOS ioreg)
│   │   └── input_sensor.py            #   Keyboard/mouse activity stats (KPM/CPM)
│   │
│   ├── brain/                         # [Decision Layer]
│   │   ├── context_classifier.py      #   Window title → subtask classification
│   │   ├── semantic_gating.py         #   Semantic gating (fast vector matching)
│   │   └── context_router.py          #   Three-level routing (semantic → cache → LLM)
│   │
│   ├── vision/                        # [Visual Analysis]
│   │   └── visual_harvester.py        #   VLM screenshot analysis (background async)
│   │
│   ├── memory/                        # [Memory Layer]
│   │   ├── task_block_manager.py      #   Pagination manager (task isolation)
│   │   ├── markdown_logger.py         #   Markdown log writer
│   │   └── audit_logger.py            #   Decision audit log
│   │
│   ├── tools/
│   │   └── context_tools.py           #   Utility tools (file content reading)
│   │
│   ├── llm/                           # [LLM Abstraction Layer]
│   │   ├── base.py                    #   Interface declaration
│   │   ├── ollama_backend.py          #   Ollama implementation (production)
│   │   └── mock_backend.py            #   Mock implementation (testing)
│   │
│   ├── metrics/                       # [Metrics Collection]
│   │   ├── types.py                   #   Data structures
│   │   ├── collector.py               #   Thread-safe collector
│   │   └── exporters.py               #   Proxy-mode exporters
│   │
│   ├── experiments/                   # [Experimental Framework]
│   │   ├── config.py                  #   YAML configuration loader
│   │   ├── trace.py                   #   Trace recording/replay
│   │   ├── runner.py                  #   Experiment orchestrator
│   │   ├── ablation.py                #   Ablation experiment manager
│   │   ├── evaluator.py               #   Metrics evaluation
│   │   ├── statistics.py              #   Statistical tests
│   │   └── annotate.py                #   Manual annotation tools
│   │
│   └── web/                           # [Web Dashboard]
│       ├── app.py                     #   FastAPI application
│       ├── event_bus.py               #   Event bus (thread → async decoupling)
│       ├── models.py                  #   Pydantic data models
│       ├── routers/
│       │   ├── journal.py             #     Journal API
│       │   ├── monitoring.py          #     Monitoring + WebSocket
│       │   ├── experiments.py         #     Experiment API
│       │   └── metrics.py             #     Analytics API
│       └── static/
│           └── index.html             #   UI (4 tabs)
│
├── experiments/                       # Experimental data
│   ├── configs/                       #   14 configurations (3 dimensions of ablation)
│   ├── traces/                        #   Activity trace recordings
│   └── results/                       #   Output results
│
└── tests/                             # Test suite (90+ tests)
    ├── conftest.py                    #   Shared fixtures
    ├── test_brain_pipeline.py         #   Decision routing integration tests
    ├── test_memory_pipeline.py        #   Memory system integration tests
    ├── test_recorder_lifecycle.py     #   Main loop lifecycle tests
    └── ... (other unit tests)
```

### Core Module Overview

| Module | Responsibility | Key Classes |
|--------|----------------|-------------|
| **sensors** | Real-time user activity detection | PlatformSensor, IdleSensor, InputSensor |
| **brain** | Decision routing engine | ContextRouter, SemanticGating |
| **vision** | Screenshot content analysis | VisualHarvester (async background thread) |
| **memory** | Structured storage | TaskBlockManager, MarkdownLogger |
| **llm** | Local model invocation | LLMBackend (Ollama/Mock) |
| **web** | Web dashboard and API | FastAPI app + EventBus decoupling |

---

## Quick Start

### System Requirements

- **Operating System**: macOS (Apple Silicon or Intel)
- **Python**: 3.10+
- **Ollama**: Local LLM model service (see next step)
- **Disk Space**: At least 20GB (for model storage)

### Step 1: Install and Start Ollama

Ollama is a necessary tool for running local large language models. Major Tom Journal uses Ollama to perform AI analysis locally, ensuring complete data privacy.

**1.1 Install Ollama**

Visit the official website to download: [https://ollama.com](https://ollama.com)

Or use Homebrew (macOS):

```bash
brew install ollama
```

**1.2 Start the Ollama Service**

```bash
# Ollama usually starts as a background service after installation
# Verify if Ollama is running (should return a JSON response)
curl http://localhost:11434/api/tags

# If not running, start manually:
ollama serve
```

**1.3 Pull Required Models**

```bash
# Decision reasoning model (required)
ollama pull qwen3:8b

# Visual analysis model (required)
ollama pull qwen3-vl:8b

# Semantic vector model (optional, for advanced context routing)
ollama pull qwen3-embedding:8b
```

> **Note**: Pulling models for the first time requires downloading 8-14GB of data. Ensure a stable network connection and sufficient disk space.

### Step 2: Install the Project

```bash
git clone https://github.com/Haoran2099/major-tom-journal.git
cd major-tom-journal

# Recommended installation (includes Web dashboard and development tools)
pip install -e ".[web,dev]"

# Minimal installation (core functionality only)
# pip install -e .

# Includes experimental framework (for performance testing)
# pip install -e ".[web,dev,experiments]"
```

**Explanation**:
- `[web]`: Web dashboard dependencies (FastAPI, WebSocket, etc.)
- `[dev]`: Development and testing tools (pytest, etc.)
- `[experiments]`: Experimental framework (data analysis, statistical tests)
- If unspecified, only core functionality (sensors, decision layer, local storage) is installed.

### Step 3: Start the Recorder and Web Dashboard

```bash
# Recommended: Start both Recorder and Web dashboard
python -m major_tom

# Or use the legacy entry point (backward compatibility)
python Major_Tom_Journal.py
```

By default, the above commands will:
- Start the **Activity Recorder** (monitors screen activity) in the background
- Launch the **Web Dashboard** at **http://localhost:8000** (optional for viewing)
- Automatically generate log files in `~/Downloads/LLM_Journal/`

For more options on running modes, see the "Running Modes" section below.

### Running Modes

`python -m major_tom` supports multiple startup modes for different use cases:

| Command | Components Started | Use Case |
|---------|--------------------|----------|
| `python -m major_tom` | Recorder + Web (default) | View logs and analysis via Web dashboard |
| `python -m major_tom --no-web` | Recorder only | Server mode, minimal resource usage |
| `python -m major_tom --web-only` | Web dashboard only | Connect to an already running Recorder process |
| `python -m major_tom --port 8080` | Specify Web port | Default port 8000 is occupied |
| `python -m major_tom --record-trace <DIR>` | Record activity trace | For offline experimental replay |

**Features**:
- Recorder and Web run independently, without interfering with each other
- Press Ctrl+C to gracefully stop all processes
- Log files are generated in real-time, ensuring no data loss even during system crashes

### Step 4: Run Tests (Optional)

Verify that all components are working correctly:

```bash
pytest                    # Run all 90+ tests
pytest --cov=major_tom    # With coverage report
```

---

## Logs vs Memory

The system generates two types of documents:

| Dimension | Daily Log (Activity Stream) | Task Memory (Task Context) |
|-----------|-----------------------------|-----------------------------|
| **File Location** | `~/Downloads/LLM_Journal/Record/YYYY-MM-DD.md` | `~/Downloads/LLM_Journal/Memory/{TaskID}.md` |
| **Number of Files** | 1 per day | Varies by task (Safari_Research, VS Code, etc.) |
| **Content** | All activity events in chronological order | Long-term context and summaries for each task |
| **Update Frequency** | Real-time append | On task switch / Background async updates |
| **Purpose** | Revisit what was done at a specific time | Provide context to AI, influencing future analysis and decisions |
| **Editability** | Editable, but affects time stream | ✅ **Strongly recommended to edit** to guide AI understanding |
| **Data Retention** | Permanently saved, query by date | Up to 200 recent records (to prevent file bloat) |

**Usage Recommendations**:
- **Daily Log**: Review your work at the end of the day, search for activities at specific times
- **Task Memory**: Regularly edit and supplement, helping AI to more accurately understand your task context

---

## Output Description

### Directory Structure

The system will generate the following files and directories under `~/Downloads/LLM_Journal/`:

```
LLM_Journal/
├── Memory/                        # 【Task Memory Files】
│   ├── Safari_Research.md         #   Memory when browsing academic materials
│   ├── Safari_Entertainment.md    #   Memory when browsing entertainment content
│   ├── VS_Code_Project.md         #   Memory for coding tasks
│   └── ... (automatically created by task)
│
├── Record/
│   ├── 2026-02-07.md              #   Complete activity log for a specific day (chronological)
│   ├── 2026-02-06.md
│   ├── decision_cache.json         #   【Routing Decision Cache】to avoid re-analysis
│   └── decision_debug.log         #   【Debug Log】detailed process of each decision
│
└── config.json                    #   【User Configuration】(copy in project root)
```

### Detailed Explanation of Each File

| File Type | File Location | Update Method | Function | When to View |
|-----------|---------------|---------------|----------|--------------|
| **Daily Log** | `Record/YYYY-MM-DD.md` | Real-time append | Chronological record of all activities: focus shifts, screenshot analyses, file edits, idle detections | Review work for a specific day, find information from a specific time |
| **Task Memory** | `Memory/{TaskID}.md` | On task switch + Background updates | Long-term memory organized by task, includes VLM analyses and text snapshots | AI decision reference, manual editing for optimization |
| **Decision Cache** | `Record/decision_cache.json` | Incremental updates | Window title → decision mapping, avoids repeated LLM analysis for the same window | (For debugging) Check cache hits |
| **Debug Log** | `Record/decision_debug.log` | Real-time append | Detailed process of each routing decision: semantic scores, cache matches, LLM inference times | Performance optimization, problem diagnosis |
| **config.json** | Project root + `Record/config.json` | Generated on first run, can be manually edited later | User configuration (models, sampling intervals, routing policies, etc.) | Adjust system behavior |

### Sample Content

**Task Memory Example** (`Safari_Research.md`):
```markdown
# Context Memory: Safari_Research
> Last Active: 2026-02-07 14:23
> **Tip**: You can edit this file to guide the Agent's context.

## Analysis History

- **[VLM_ANALYSIS]** (14:20:15) ➣ User reading arXiv paper on LLM evaluation
  > Detected mathematics equations, code snippets, research methodology section

- **[TEXT_SNAPSHOT]** (14:15:42) ➣ Chrome tab: "arxiv.org/abs/2401.12345"
  > [Paper title, abstract...]
```

### Key Features

- All files are plain text (Markdown / JSON / logs), and can be edited directly
- Memory files (`Memory/`) are the core of the **Human-in-the-loop** system: your edits will influence subsequent AI understanding
- Decision cache ensures the same window does not consume tokens repeatedly
- Debug logs help diagnose performance issues and optimize sampling strategies

---

## Web Dashboard User Guide

If you started the system using `python -m major_tom` (default method), the Web dashboard is automatically available at `http://localhost:8000`.

### 4 Core Tabs

#### 1. **Dashboard** - Real-time Monitoring
- System status (is Recorder running, current task, active application)
- Real-time activity stream (recent window switches, screenshot analysis events)
- Keyboard and mouse activity levels (KPM - keystrokes per minute, CPM - clicks per minute)
- Token usage statistics (this hour / today's total)

#### 2. **Journal** - Log Viewing and Searching
- Calendar selection: Quickly view activities from a specific date
- Log content: Displays the full chronological Daily Log
- Memory file browsing: Lists all Task Memory (click to edit)
- Full-text search: Search activity keywords across dates

#### 3. **Experiments** - Experiment Management (Advanced)
- Experiment configuration: Load and manage YAML config files
- Result comparison: Visualize performance across different configurations
- Data export: Export results as CSV / JSON

#### 4. **Analytics** - Analysis and Statistics
- Token usage trends: Graphs of token consumption over time
- Decision distribution: Pie chart of SNAPSHOT vs SKIP ratios
- Latency statistics: Distribution of VLM analysis and LLM inference times

### Common Questions

**Q: Web port 8000 is occupied, how to change?**

```bash
python -m major_tom --port 8080
# Then access http://localhost:8080
```

**Q: After modifying config.json, when does it take effect?**

- Recorder reloads configuration periodically (around every 30 seconds)
- No need to restart, new parameters are applied automatically

**Q: How to disable the Web dashboard and run in the background?**

```bash
python -m major_tom --no-web
```

The Recorder will continue to generate log files, and you can later view them using `python -m major_tom --web-only`.

**Q: How to connect to an already running Recorder?**

If the Recorder is running in another terminal, you can use:

```bash
python -m major_tom --web-only --port 8001
```

To connect to the same Recorder instance (sharing the Event Bus).

**Q: Log files are too large, how to clean up?**

- Delete `~/Downloads/LLM_Journal/Record/2026-01-*.md` (delete expired logs by date)
- Task Memory will automatically retain the most recent 200 records, no manual cleanup needed
- config.json and decision_cache.json can be safely retained

---

## Configuration

Edit `config.json` (automatically generated in project root on first run):

```json
{
  "models": {
    "brain_model": "qwen3:8b",
    "eye_model": "qwen3-vl:8b",
    "embedding_model": "qwen3-embedding:8b"
  },
  "parameters": {
    "sample_interval": 5,
    "idle_threshold": 180,
    "vlm_cooldown": 60
  },
  "context_routing": {
    "enabled": true,
    "method": "keyword",
    "apps": {
      "Safari": {
        "Research": ["arxiv", "scholar", "github", "stackoverflow"],
        "Entertainment": ["youtube", "bilibili", "netflix"]
      }
    }
  }
}
```

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `sample_interval` | Main loop polling interval (seconds) | 5 |
| `idle_threshold` | Considered idle after how many seconds | 180 |
| `vlm_cooldown` | Minimum interval between two screenshot analyses (seconds) | 60 |
| `context_routing.method` | Routing method: `keyword` (keyword matching) or `semantic` (embedding vector matching) | keyword |

---

## Experimental Framework

The project includes an experimental framework for systematically evaluating the performance of various system components.

### Three Experimental Dimensions

**1. Token Efficiency (te_c0..c4)**: Ablation studies on the routing pipeline

| Configuration | Semantic Gating | Decision Cache | Adaptive Sampling | Purpose |
|---------------|-----------------|----------------|-------------------|--------|
| C0 Naive | - | - | - | Baseline: Calls LLM every time |
| C1 Cache-Only | - | ON | - | Uses cache only |
| C2 Semantic-Only | ON | - | - | Uses semantic gating only |
| C3 Sem+Cache | ON | ON | - | Semantic gating + cache |
| C4 Full System | ON | ON | ON | Complete system |

**2. Memory Mechanism (mm_m0..m3)**: Evaluating context isolation effects

| Configuration | Routing Method | Purpose |
|---------------|----------------|---------|
| M0 Global | None (single global file) | All activities written to the same memory |
| M1 App-Only | By app name | Separate only by application |
| M2 Keyword | Keyword matching | Separate subtasks by title keywords |
| M3 Semantic | Embedding vector | Separate subtasks by semantic similarity |

**3. Log Quality (jq_q0..q4)**: Impact of different model combinations on log quality

| Configuration | Brain | Eye (VLM) | Description |
|---------------|-------|-----------|-------------|
| Q0 | qwen3:4b | qwen3-vl:4b | Small models |
| Q1 | qwen3:8b | qwen3-vl:8b | Default configuration |
| Q2 | qwen3:14b | qwen3-vl:14b | Large models |
| Q3 | gemma3:12b | gemma3-vl:12b | Different model family |
| Q4 | qwen3:8b | (no VLM) | Text only, no screenshot analysis |

### Running Experiments

```bash
# 1. Record a real activity trace (for later replay)
python -m major_tom --record-trace experiments/traces/session_001/

# 2. Replay the trace with specified configuration, collect metrics
python -m major_tom.experiments.runner \
  --trace experiments/traces/session_001/ \
  --config experiments/configs/te_c4_full_system.yaml

# 3. Results are output to experiments/results/
```

### Evaluation Metrics

- **Token Efficiency**: token_savings_vs_naive, semantic_hit_rate, cache_hit_rate
- **Memory Isolation**: CPR (Context Pollution Rate), CRA (Context Recall Accuracy), MIS (Memory Isolation Score)
- **Log Quality**: RS (Relevance), RR (Redundancy Rate), ACR (Activity Coverage Rate), ID (Information Density)
- **Statistical Tests**: ANOVA + Tukey HSD + Cohen's d + 95% confidence interval

---

## Technical Details

### LLM Abstraction Layer

All LLM calls go through the `LLMBackend` abstract interface, supporting:
- `OllamaBackend`: Production, calls local Ollama
- `MockBackend`: Testing, returns pre-configured fixed responses
- `MetricsCollectingBackend`: Proxy mode, transparently records token count and latency for each call

### EventBus

The `EventBus` singleton decouples Recorder (synchronous thread) and Web UI (asynchronous FastAPI):
- Recorder publishes events using `bus.publish("decision", {...})`
- WebSocket handler receives events using `bus.subscribe_async("*", callback)`
- Safely schedules from synchronous thread to asynchronous event loop using `asyncio.run_coroutine_threadsafe()`

### Safety Measures

- All Web API path parameters are validated with `path.resolve().is_relative_to(root)` to prevent path traversal
- No `shell=True` used for executing subprocesses
- VLM worker threads gracefully stop using `shutdown_event`, avoiding interrupted file writes
- MetricsCollector has a 50,000 event limit to prevent infinite memory growth

---

## Privacy

- All processing is done locally, with Ollama calling local models
- No network requests are sent
- Keyboard input content is not logged, only activity levels (KPM/CPM)
- Memory files are plain text, editable or deletable at any time
- Every decision has an audit log for traceability

---

## Platform Support

Currently, only macOS is supported (window detection and screenshot depend on macOS APIs). Windows and Linux support are planned.

---

## Acknowledgments

The project is named after David Bowie's *Space Oddity*. Major Tom runs quietly in the background of your digital life, observing, understanding, and remembering.

The following AI-assisted tools were used during the development of this project:
- **Claude Opus 4.5** (Anthropic) - Code refactoring, experimental framework, Web UI, testing, code review
- **Kimi K2.5** (Moonshot AI) - Early documentation writing
- **Gemini 3 Pro** (Google) - Early code development
