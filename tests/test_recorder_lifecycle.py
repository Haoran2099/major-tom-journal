"""Integration test: full recorder lifecycle with mocks."""

import json
from unittest.mock import patch, MagicMock

import pytest

from major_tom.config import Config
from major_tom.llm.base import EmbeddingResponse, LLMResponse
from major_tom.llm.mock_backend import MockBackend


def _mock_io_sensor():
    """Create a mock InputActivitySensor that doesn't start pynput listeners."""
    sensor = MagicMock()
    sensor.get_and_reset_stats.return_value = {"kpm": 0, "cpm": 0}
    return sensor


@pytest.fixture
def mock_recorder(tmp_config):
    """Build a recorder with pynput mocked out."""
    mock = MockBackend()
    Config.SEMANTIC_ENABLED = False

    with patch("major_tom.recorder.InputActivitySensor", side_effect=lambda: _mock_io_sensor()):
        from major_tom.recorder import Major_Tom_Recorder
        recorder = Major_Tom_Recorder(llm_backend=mock)
    return recorder, mock


class TestRecorderLifecycle:
    def test_recorder_construction_with_mock(self, mock_recorder):
        recorder, mock = mock_recorder
        # _llm is now a MetricsCollectingBackend wrapping the mock
        assert recorder._llm._wrapped is mock
        assert recorder.metrics_collector is not None
        assert recorder.current_task_id == "startup"
        assert recorder.is_away is False

    def test_decision_callback(self, mock_recorder):
        recorder, _ = mock_recorder
        recorder._on_router_decision(
            {"action": "SKIP", "next_check_delay": 30},
            "VS_Code_Python", "main.py", None,
        )
        assert recorder.current_interval == 30

    def test_snapshot_decision_with_text(self, mock_recorder):
        recorder, _ = mock_recorder
        test_file = Config.MONITOR_PATH / "test_file.py"
        test_file.write_text("print('hello')")

        recorder._on_router_decision(
            {"action": "SNAPSHOT", "prompt": "Analyze code", "next_check_delay": 10},
            "VS_Code", "test_file.py - VS Code", None,
        )
        assert recorder.pending_snapshot is None

    def test_snapshot_decision_pending_vlm(self, mock_recorder):
        recorder, _ = mock_recorder
        recorder._on_router_decision(
            {
                "action": "SNAPSHOT", "prompt": "Analyze screen",
                "next_check_delay": 10, "region_mode": "ACTIVE_WINDOW",
            },
            "Safari_Research", "arxiv.org - Paper Title", (0, 0, 800, 600),
        )
        assert recorder.pending_snapshot is not None
        assert recorder.pending_snapshot["task_id"] == "Safari_Research"

    def test_metrics_collecting_backend(self):
        from major_tom.metrics.collector import MetricsCollector
        from major_tom.metrics.exporters import MetricsCollectingBackend

        mock = MockBackend()
        collector = MetricsCollector(experiment_id="test")
        instrumented = MetricsCollectingBackend(mock, collector)

        mock.set_generate_response(LLMResponse(
            text="test output",
            prompt_tokens=50, completion_tokens=20, total_tokens=70,
            model="mock", latency_ms=100.0,
        ))
        res = instrumented.generate("mock", "test prompt")
        assert res.text == "test output"

        instrumented.embed("mock", "test text")

        summary = collector.get_summary()
        assert summary["token_metrics"]["total_tokens"] > 0
        assert summary["token_metrics"]["brain_tokens_total"] == 70

        events = collector.get_events()
        assert len(events) == 2
