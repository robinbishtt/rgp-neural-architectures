"""
src/utils/telemetry_logger.py

Backward-compatible shim. Canonical implementation is at src/telemetry/telemetry_logger.py
"""
from src.telemetry.telemetry_logger import TelemetryLogger  # noqa: F401

__all__ = ["TelemetryLogger"]
