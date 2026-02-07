"""Experiment configuration, execution, and results endpoints."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from major_tom.web.models import ExperimentConfigInfo, ExperimentResultInfo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/experiments", tags=["experiments"])

CONFIGS_DIR = Path("experiments/configs")
RESULTS_DIR = Path("experiments/results")


@router.get("/configs", response_model=List[ExperimentConfigInfo])
def list_configs():
    """List available experiment configs."""
    configs = []
    if CONFIGS_DIR.exists():
        for yaml_file in sorted(CONFIGS_DIR.glob("*.yaml")):
            try:
                import yaml
                with open(yaml_file) as f:
                    data = yaml.safe_load(f)
                exp = data.get("experiment", {})
                configs.append(ExperimentConfigInfo(
                    name=exp.get("name", yaml_file.stem),
                    dimension=exp.get("dimension", ""),
                    description=exp.get("description", ""),
                    path=str(yaml_file),
                ))
            except Exception:
                configs.append(ExperimentConfigInfo(
                    name=yaml_file.stem, dimension="", description="", path=str(yaml_file),
                ))
    return configs


@router.get("/results", response_model=List[Dict[str, Any]])
def list_results():
    """List experiment results."""
    results = []
    if RESULTS_DIR.exists():
        for exp_dir in sorted(RESULTS_DIR.iterdir()):
            if exp_dir.is_dir():
                for run_dir in sorted(exp_dir.iterdir()):
                    result_file = run_dir / "results.json"
                    if result_file.exists():
                        try:
                            data = json.loads(result_file.read_text())
                            results.append(data)
                        except (json.JSONDecodeError, OSError):
                            pass
    return results


@router.get("/results/{name}")
def get_result(name: str):
    """Get results for a specific experiment."""
    exp_dir = RESULTS_DIR / name
    if not exp_dir.resolve().is_relative_to(RESULTS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="Invalid experiment name")
    if not exp_dir.exists():
        raise HTTPException(status_code=404, detail=f"No results for {name}")

    runs = []
    for run_dir in sorted(exp_dir.iterdir()):
        result_file = run_dir / "results.json"
        if result_file.exists():
            try:
                runs.append(json.loads(result_file.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
    return {"name": name, "runs": runs}


@router.get("/compare")
def compare_experiments(ids: str = Query(..., description="Comma-separated experiment names")):
    """Compare two or more experiment results."""
    names = [n.strip() for n in ids.split(",")]
    comparison = {}
    for name in names:
        exp_dir = RESULTS_DIR / name
        if not exp_dir.resolve().is_relative_to(RESULTS_DIR.resolve()):
            continue
        if exp_dir.exists():
            runs = []
            for run_dir in sorted(exp_dir.iterdir()):
                result_file = run_dir / "results.json"
                if result_file.exists():
                    try:
                        runs.append(json.loads(result_file.read_text()))
                    except (json.JSONDecodeError, OSError):
                        pass
            comparison[name] = runs
    return comparison
