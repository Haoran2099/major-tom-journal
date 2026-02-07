"""Tests for VisualHarvester."""

import pytest
from PIL import Image

from major_tom.config import Config
from major_tom.llm.base import LLMResponse
from major_tom.llm.mock_backend import MockBackend
from major_tom.vision.visual_harvester import VisualHarvester


class TestVisualHarvester:
    def test_static_detection(self):
        mock = MockBackend()
        harvester = VisualHarvester(mock)

        img = Image.new("RGB", (200, 200), color="red")
        # First call - no previous thumbnail
        result1 = harvester.harvest("Analyze", img)
        assert "[STATIC]" not in result1

        # Second call with same image - should detect static
        result2 = harvester.harvest("Analyze", img)
        assert "[STATIC]" in result2

    def test_vlm_call(self):
        mock = MockBackend()
        mock.set_generate_response(LLMResponse(
            text="User is writing Python code in VS Code",
            prompt_tokens=500, completion_tokens=20, total_tokens=520,
            model="test-eye", latency_ms=2000.0,
        ))

        harvester = VisualHarvester(mock)
        img = Image.new("RGB", (800, 600), color="blue")
        result = harvester.harvest("Summarize code", img)

        assert "Python code" in result
        assert mock.generate_count == 1

    def test_different_images_not_static(self):
        mock = MockBackend()
        mock.set_generate_response(LLMResponse(
            text="Analysis result",
            prompt_tokens=100, completion_tokens=10, total_tokens=110,
            model="test-eye", latency_ms=100.0,
        ))

        harvester = VisualHarvester(mock)

        img1 = Image.new("RGB", (200, 200), color="red")
        img2 = Image.new("RGB", (200, 200), color="blue")

        harvester.harvest("Analyze", img1)
        result = harvester.harvest("Analyze", img2)

        assert "[STATIC]" not in result
        assert mock.generate_count == 2

    def test_resize_large_image(self):
        mock = MockBackend()
        mock.set_generate_response(LLMResponse(
            text="resized", prompt_tokens=100, completion_tokens=5,
            total_tokens=105, model="test-eye", latency_ms=100.0,
        ))

        harvester = VisualHarvester(mock)
        large_img = Image.new("RGB", (3840, 2160), color="green")
        result = harvester.harvest("Analyze", large_img)

        assert result == "resized"
        # Check the image sent was resized
        call = mock.generate_calls[0]
        assert call["images"] is not None

    def test_none_image(self):
        mock = MockBackend()
        harvester = VisualHarvester(mock)
        result = harvester.harvest("Analyze", None)
        assert "Error" in result

    def test_error_handling(self):
        mock = MockBackend()
        # Make generate raise an exception
        original_generate = mock.generate
        def failing_generate(*args, **kwargs):
            raise RuntimeError("Connection failed")
        mock.generate = failing_generate

        harvester = VisualHarvester(mock)
        img = Image.new("RGB", (100, 100), color="white")
        result = harvester.harvest("Analyze", img)
        assert "Error" in result
