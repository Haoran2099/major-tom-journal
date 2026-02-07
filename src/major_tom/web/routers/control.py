"""Experiment control endpoints: run, stop, status, and live streaming."""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/control", tags=["control"])

# Store for running experiments
_running_experiments: Dict[str, "ExperimentRun"] = {}


@dataclass
class ExperimentRun:
    """Tracks a running experiment."""

    run_id: str
    config_name: str
    status: str = "pending"  # pending, running, completed, failed, stopped
    progress: float = 0.0
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    current_step: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    # Internal
    _task: Optional[asyncio.Task] = field(default=None, repr=False)
    _subscribers: List[asyncio.Queue] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_name": self.config_name,
            "status": self.status,
            "progress": self.progress,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "current_step": self.current_step,
            "metrics": self.metrics,
            "error": self.error,
        }

    async def publish(self, event: Dict[str, Any]):
        """Publish event to all subscribers."""
        self.events.append(event)
        for queue in self._subscribers:
            try:
                await queue.put(event)
            except Exception:
                pass


@router.post("/experiments/run/{config_name}")
async def start_experiment(config_name: str, trace_dir: str = "experiments/traces/mixed_60m"):
    """Start an experiment run."""

    # Check if config exists
    config_path = Path(f"experiments/configs/{config_name}.yaml")
    if not config_path.exists():
        raise HTTPException(status_code=404, detail=f"Config not found: {config_name}")

    # Check if trace exists
    trace_path = Path(trace_dir)
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail=f"Trace not found: {trace_dir}")

    # Create run
    run_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    run = ExperimentRun(
        run_id=run_id,
        config_name=config_name,
        status="pending",
        started_at=datetime.now().isoformat(),
    )

    _running_experiments[run_id] = run

    # Start async task
    run._task = asyncio.create_task(_run_experiment(run, config_path, trace_path))

    return {"run_id": run_id, "status": "started"}


async def _run_experiment(run: ExperimentRun, config_path: Path, trace_path: Path):
    """Execute experiment in background."""
    try:
        run.status = "running"
        await run.publish({"type": "status", "status": "running"})

        # Load config
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        run.current_step = "Loading trace..."
        await run.publish({"type": "step", "step": run.current_step, "progress": 0.05})

        # Load trace
        trace_file = trace_path / "trace.jsonl"
        if not trace_file.exists():
            raise FileNotFoundError(f"trace.jsonl not found in {trace_path}")

        events = []
        with open(trace_file) as f:
            for line in f:
                events.append(json.loads(line))

        total_events = len(events)
        run.current_step = f"Processing {total_events} events..."
        await run.publish({"type": "step", "step": run.current_step, "progress": 0.1})

        # Simulate processing (in real implementation, this would use ExperimentRunner)
        metrics = {
            "total_tokens": 0,
            "llm_calls": 0,
            "vlm_calls": 0,
            "cache_hits": 0,
            "semantic_filtered": 0,
            "snapshot_count": 0,
            "skip_count": 0,
            "latencies": [],
        }

        for i, event in enumerate(events):
            if run.status == "stopped":
                break

            # Simulate processing delay
            await asyncio.sleep(0.05)

            # Update progress
            progress = 0.1 + 0.85 * (i + 1) / total_events
            run.progress = progress

            # Simulate metrics updates
            import random
            decision = random.choice(["SNAPSHOT", "SKIP", "SKIP"])

            if decision == "SNAPSHOT":
                metrics["snapshot_count"] += 1
                metrics["llm_calls"] += 1
                metrics["total_tokens"] += random.randint(100, 300)
                metrics["latencies"].append(random.uniform(50, 200))
            else:
                metrics["skip_count"] += 1
                if random.random() < 0.5:
                    metrics["semantic_filtered"] += 1
                else:
                    metrics["cache_hits"] += 1

            run.metrics = metrics.copy()
            run.metrics["latencies"] = metrics["latencies"][-10:]  # Keep last 10

            # Publish progress every 10 events
            if (i + 1) % 10 == 0 or i == total_events - 1:
                await run.publish({
                    "type": "progress",
                    "progress": progress,
                    "processed": i + 1,
                    "total": total_events,
                    "metrics": {
                        "total_tokens": metrics["total_tokens"],
                        "llm_calls": metrics["llm_calls"],
                        "snapshot_count": metrics["snapshot_count"],
                        "skip_count": metrics["skip_count"],
                        "cache_hit_rate": metrics["cache_hits"] / max(i + 1, 1),
                        "semantic_filter_rate": metrics["semantic_filtered"] / max(i + 1, 1),
                    },
                })

        if run.status != "stopped":
            run.status = "completed"
            run.progress = 1.0
            run.ended_at = datetime.now().isoformat()
            run.current_step = "Completed"

            # Save results
            results_dir = Path(f"experiments/results/{run.config_name}/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            results_dir.mkdir(parents=True, exist_ok=True)

            with open(results_dir / "results.json", "w") as f:
                json.dump({
                    "experiment": {
                        "name": run.config_name,
                        "run_id": run.run_id,
                        "started_at": run.started_at,
                        "ended_at": run.ended_at,
                    },
                    "metrics": metrics,
                }, f, indent=2)

            await run.publish({
                "type": "completed",
                "metrics": metrics,
                "results_path": str(results_dir),
            })

    except Exception as e:
        logger.exception(f"Experiment {run.run_id} failed")
        run.status = "failed"
        run.error = str(e)
        run.ended_at = datetime.now().isoformat()
        await run.publish({"type": "error", "error": str(e)})


@router.get("/experiments/run/{run_id}/status")
async def get_experiment_status(run_id: str):
    """Get status of a running experiment."""
    if run_id not in _running_experiments:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return _running_experiments[run_id].to_dict()


@router.post("/experiments/run/{run_id}/stop")
async def stop_experiment(run_id: str):
    """Stop a running experiment."""
    if run_id not in _running_experiments:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    run = _running_experiments[run_id]

    if run.status not in ("pending", "running"):
        return {"status": run.status, "message": "Experiment already finished"}

    run.status = "stopped"
    run.ended_at = datetime.now().isoformat()

    if run._task:
        run._task.cancel()

    await run.publish({"type": "stopped"})

    return {"status": "stopped", "run_id": run_id}


@router.get("/experiments/runs")
async def list_experiment_runs():
    """List all experiment runs."""
    return [run.to_dict() for run in _running_experiments.values()]


@router.websocket("/experiments/run/{run_id}/stream")
async def stream_experiment(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for live experiment updates."""
    await websocket.accept()

    if run_id not in _running_experiments:
        await websocket.send_json({"type": "error", "error": "Run not found"})
        await websocket.close()
        return

    run = _running_experiments[run_id]
    queue: asyncio.Queue = asyncio.Queue()
    run._subscribers.append(queue)

    try:
        # Send current state
        await websocket.send_json({
            "type": "init",
            "state": run.to_dict(),
            "events": run.events[-50:],  # Last 50 events
        })

        # Stream updates
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(event)

                if event.get("type") in ("completed", "stopped", "error"):
                    break
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    finally:
        run._subscribers.remove(queue)
