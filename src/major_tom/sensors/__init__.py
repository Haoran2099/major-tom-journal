"""Sensor modules for input, idle, and platform detection."""

from major_tom.sensors.idle_sensor import IdleSensor
from major_tom.sensors.input_sensor import InputActivitySensor
from major_tom.sensors.platform_sensor import PlatformSensor

__all__ = ["InputActivitySensor", "IdleSensor", "PlatformSensor"]
