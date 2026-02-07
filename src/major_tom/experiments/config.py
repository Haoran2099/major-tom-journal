"""Experiment configuration loader from YAML."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Unified configuration for all experiment dimensions."""

    # Experiment metadata
    name: str = "unnamed"
    dimension: str = "token_efficiency"
    description: str = ""
    repeat: int = 3
    seed: int = 42

    # Models
    brain_model: str = "qwen3:8b"
    eye_model: str = "qwen3-vl:8b"
    embedding_model: str = "qwen3-embedding:8b"

    # Components toggle
    semantic_gating_enabled: bool = True
    semantic_gating_threshold: float = 0.30
    decision_cache_enabled: bool = True
    vlm_enabled: bool = True
    vlm_cooldown: int = 60
    vlm_diff_threshold: float = 0.9
    context_routing_enabled: bool = True
    context_routing_method: str = "keyword"
    adaptive_sampling_enabled: bool = True
    pattern_learning_enabled: bool = True

    # LLM options
    temperature: float = 0.1
    num_ctx: int = 4096
    num_predict: int = 500

    # Metrics
    collect_all: bool = True
    export_format: str = "both"
    export_on_complete: bool = True

    # Raw YAML data for extensions
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        """Load experiment configuration from a YAML file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cfg = cls(raw=data)

        # Experiment section
        exp = data.get("experiment", {})
        cfg.name = exp.get("name", cfg.name)
        cfg.dimension = exp.get("dimension", cfg.dimension)
        cfg.description = exp.get("description", cfg.description)
        cfg.repeat = exp.get("repeat", cfg.repeat)
        cfg.seed = exp.get("seed", cfg.seed)

        # Models section
        models = data.get("models", {})
        cfg.brain_model = models.get("brain", cfg.brain_model)
        cfg.eye_model = models.get("eye", cfg.eye_model)
        cfg.embedding_model = models.get("embedding", cfg.embedding_model)

        # Components section
        components = data.get("components", {})

        sg = components.get("semantic_gating", {})
        cfg.semantic_gating_enabled = sg.get("enabled", cfg.semantic_gating_enabled)
        cfg.semantic_gating_threshold = sg.get("threshold", cfg.semantic_gating_threshold)

        dc = components.get("decision_cache", {})
        cfg.decision_cache_enabled = dc.get("enabled", cfg.decision_cache_enabled)

        vlm = components.get("vlm", {})
        cfg.vlm_enabled = vlm.get("enabled", cfg.vlm_enabled)
        cfg.vlm_cooldown = vlm.get("cooldown", cfg.vlm_cooldown)
        cfg.vlm_diff_threshold = vlm.get("diff_threshold", cfg.vlm_diff_threshold)

        cr = components.get("context_routing", {})
        cfg.context_routing_enabled = cr.get("enabled", cfg.context_routing_enabled)
        cfg.context_routing_method = cr.get("method", cfg.context_routing_method)

        adap = components.get("adaptive_sampling", {})
        cfg.adaptive_sampling_enabled = adap.get("enabled", cfg.adaptive_sampling_enabled)

        pl = components.get("pattern_learning", {})
        cfg.pattern_learning_enabled = pl.get("enabled", cfg.pattern_learning_enabled)

        # LLM options
        llm_opts = data.get("llm_options", {})
        cfg.temperature = llm_opts.get("temperature", cfg.temperature)
        cfg.num_ctx = llm_opts.get("num_ctx", cfg.num_ctx)
        cfg.num_predict = llm_opts.get("num_predict", cfg.num_predict)

        # Metrics section
        metrics = data.get("metrics", {})
        cfg.collect_all = metrics.get("collect_all", cfg.collect_all)
        cfg.export_format = metrics.get("export_format", cfg.export_format)
        cfg.export_on_complete = metrics.get("export_on_complete", cfg.export_on_complete)

        logger.info("Loaded experiment config: %s (%s)", cfg.name, cfg.dimension)
        return cfg
