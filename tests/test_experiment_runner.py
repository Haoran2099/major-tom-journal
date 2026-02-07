"""Tests for ExperimentConfig and ExperimentRunner."""

import json

import pytest
import yaml

from major_tom.experiments.config import ExperimentConfig


class TestExperimentConfig:
    def test_load_yaml(self, tmp_path):
        config_data = {
            "experiment": {
                "name": "test_exp",
                "dimension": "token_efficiency",
                "description": "Test experiment",
                "repeat": 2,
                "seed": 123,
            },
            "models": {
                "brain": "test:1b",
                "eye": "test-vl:1b",
                "embedding": "test-embed:1b",
            },
            "components": {
                "semantic_gating": {"enabled": False, "threshold": 0.5},
                "decision_cache": {"enabled": True},
                "vlm": {"enabled": True, "cooldown": 30, "diff_threshold": 0.85},
                "context_routing": {"enabled": True, "method": "semantic"},
                "adaptive_sampling": {"enabled": False},
                "pattern_learning": {"enabled": False},
            },
            "llm_options": {
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 256,
            },
            "metrics": {
                "collect_all": True,
                "export_format": "json",
                "export_on_complete": True,
            },
        }

        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = ExperimentConfig.load(config_path)
        assert cfg.name == "test_exp"
        assert cfg.dimension == "token_efficiency"
        assert cfg.repeat == 2
        assert cfg.brain_model == "test:1b"
        assert cfg.semantic_gating_enabled is False
        assert cfg.vlm_cooldown == 30
        assert cfg.temperature == 0.2
        assert cfg.export_format == "json"

    def test_defaults(self, tmp_path):
        config_path = tmp_path / "minimal.yaml"
        config_path.write_text("experiment:\n  name: minimal\n")

        cfg = ExperimentConfig.load(config_path)
        assert cfg.name == "minimal"
        assert cfg.brain_model == "qwen3:8b"
        assert cfg.repeat == 3
        assert cfg.semantic_gating_enabled is True


class TestAblationManager:
    def test_apply_to_config(self, tmp_path):
        from major_tom.config import Config
        from major_tom.experiments.ablation import AblationManager

        config_data = {
            "experiment": {"name": "ablation_test", "dimension": "token_efficiency"},
            "models": {"brain": "tiny:1b", "eye": "tiny-vl:1b", "embedding": "tiny-embed"},
            "components": {
                "semantic_gating": {"enabled": False},
                "vlm": {"cooldown": 120},
            },
        }

        config_path = tmp_path / "ablation.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        cfg = ExperimentConfig.load(config_path)
        ablation = AblationManager(cfg)

        orig_brain = Config.BRAIN_MODEL
        try:
            ablation.apply_to_config()
            assert Config.BRAIN_MODEL == "tiny:1b"
            assert Config.SEMANTIC_ENABLED is False
            assert Config.VLM_COOLDOWN == 120
        finally:
            Config.BRAIN_MODEL = orig_brain


class TestTraceReplayerLoad:
    def test_load_trace(self, tmp_path):
        trace_dir = tmp_path / "trace"
        trace_dir.mkdir()

        events = [
            {"timestamp": 1.0, "elapsed_ms": 0, "app": "VS Code", "title": "main.py",
             "region": None, "kpm": 100, "cpm": 5, "idle_seconds": 0,
             "file_events": [], "screenshot_path": None, "ground_truth_description": ""},
            {"timestamp": 2.0, "elapsed_ms": 1000, "app": "Safari", "title": "arxiv",
             "region": [0, 0, 800, 600], "kpm": 0, "cpm": 10, "idle_seconds": 0,
             "file_events": [], "screenshot_path": None, "ground_truth_description": ""},
        ]

        with open(trace_dir / "trace.jsonl", "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        from major_tom.experiments.trace import TraceReplayer
        replayer = TraceReplayer(trace_dir)
        assert len(replayer.events) == 2
        assert replayer.has_next()

        app, title, region = replayer.get_active_window()
        assert app == "VS Code"

        stats = replayer.get_and_reset_stats(5.0)
        assert stats["kpm"] == 100

        assert replayer.has_next()
        app2, title2, _ = replayer.get_active_window()
        assert app2 == "Safari"

    def test_missing_trace_file(self, tmp_path):
        from major_tom.experiments.trace import TraceReplayer
        with pytest.raises(FileNotFoundError):
            TraceReplayer(tmp_path / "nonexistent")
