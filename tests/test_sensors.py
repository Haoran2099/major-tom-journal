"""Tests for sensor modules."""

from unittest.mock import patch, MagicMock

import pytest


class TestInputActivitySensor:
    def test_get_and_reset_stats(self):
        # Mock pynput listeners to avoid needing accessibility permissions
        with patch("major_tom.sensors.input_sensor.keyboard") as mock_kb, \
             patch("major_tom.sensors.input_sensor.mouse") as mock_mouse:
            mock_kb.Listener.return_value = MagicMock()
            mock_mouse.Listener.return_value = MagicMock()

            from major_tom.sensors.input_sensor import InputActivitySensor
            sensor = InputActivitySensor()

            with sensor._lock:
                sensor._keystrokes = 30
                sensor._clicks = 5

            stats = sensor.get_and_reset_stats(10.0)
            assert stats["kpm"] == 180
            assert stats["cpm"] == 30

            stats2 = sensor.get_and_reset_stats(10.0)
            assert stats2["kpm"] == 0
            assert stats2["cpm"] == 0

    def test_zero_duration(self):
        with patch("major_tom.sensors.input_sensor.keyboard") as mock_kb, \
             patch("major_tom.sensors.input_sensor.mouse") as mock_mouse:
            mock_kb.Listener.return_value = MagicMock()
            mock_mouse.Listener.return_value = MagicMock()

            from major_tom.sensors.input_sensor import InputActivitySensor
            sensor = InputActivitySensor()
            with sensor._lock:
                sensor._keystrokes = 10
            stats = sensor.get_and_reset_stats(0.0)
            assert stats["kpm"] >= 0


class TestIdleSensor:
    def test_caching(self):
        from major_tom.sensors.idle_sensor import IdleSensor
        sensor = IdleSensor()
        sensor._cache_ttl = 100.0

        idle1 = sensor.get_idle_duration()
        idle2 = sensor.get_idle_duration()
        assert idle1 == idle2

    def test_returns_float(self):
        from major_tom.sensors.idle_sensor import IdleSensor
        sensor = IdleSensor()
        idle = sensor.get_idle_duration()
        assert isinstance(idle, float)


class TestPlatformSensor:
    def test_caching(self):
        from major_tom.sensors.platform_sensor import PlatformSensor
        sensor = PlatformSensor()
        sensor._cache_ttl = 100.0

        result1 = sensor.get_active_window()
        result2 = sensor.get_active_window()
        assert result1 == result2

    def test_returns_tuple(self):
        from major_tom.sensors.platform_sensor import PlatformSensor
        sensor = PlatformSensor()
        result = sensor.get_active_window()
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_is_valid_title(self):
        from major_tom.sensors.platform_sensor import PlatformSensor
        assert PlatformSensor._is_valid_title("main.py - VS Code") is True
        assert PlatformSensor._is_valid_title("Untitled") is False
        assert PlatformSensor._is_valid_title("") is False
        assert PlatformSensor._is_valid_title("New Tab") is False
