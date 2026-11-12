"""
Mock Service Package — Stand-in model service for high-concurrency load testing.
"""

from mock_service.calibration_data import CALIBRATION_PROFILES, ServiceCalibrationProfile
from mock_service.latency_simulator import LatencySimulator
from mock_service.load_degradation import LoadDegradationEngine
from mock_service.mock_server import app as mock_fastapi_app

__all__ = [
    "CALIBRATION_PROFILES",
    "ServiceCalibrationProfile",
    "LatencySimulator",
    "LoadDegradationEngine",
    "mock_fastapi_app",
]

