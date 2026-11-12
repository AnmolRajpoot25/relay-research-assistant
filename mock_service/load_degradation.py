"""
Load Degradation Engine — Simulates server contention and performance collapse under load.

Senior Engineering Perspective:
Hardware accelerators (GPUs/LPUs) and shared gateway queues do not scale indefinitely.
Under increasing concurrency:
1. Low load (1-5 concurrent requests): Near-nominal latency (slight queuing overhead).
2. Moderate load (6-12 concurrent requests): Linear degradation due to scheduling contention.
3. Severe overload (>12 concurrent requests): Non-linear / quadratic degradation where
   queuing delay dominates compute time, and gateway buffers begin dropping packets (HTTP 503).
"""

import logging
import random
import threading
from typing import Tuple

from mock_service.calibration_data import (
    CONCURRENT_LOAD_INFLECTION_POINT,
    DEGRADATION_SLOPE,
)

logger = logging.getLogger("LoadDegradation")


class LoadDegradationEngine:
    """Monitors live concurrency and computes dynamic slowdown and failure probabilities."""

    def __init__(
        self,
        degradation_slope: float = DEGRADATION_SLOPE,
        inflection_threshold: int = CONCURRENT_LOAD_INFLECTION_POINT,
    ):
        self.degradation_slope = degradation_slope
        self.inflection_threshold = inflection_threshold

        self._active_connections = 0
        self._peak_concurrency = 0
        self._lock = threading.Lock()

    def record_connection_start(self) -> int:
        """Call when an HTTP connection begins inference."""
        with self._lock:
            self._active_connections += 1
            if self._active_connections > self._peak_concurrency:
                self._peak_concurrency = self._active_connections
            return self._active_connections

    def record_connection_end(self) -> int:
        """Call when an HTTP connection finishes."""
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)
            return self._active_connections

    @property
    def active_connections(self) -> int:
        with self._lock:
            return self._active_connections

    @property
    def peak_concurrency(self) -> int:
        with self._lock:
            return self._peak_concurrency

    def evaluate_load_status(self) -> Tuple[float, bool]:
        """
        Evaluates the current system state.
        Returns:
            (load_multiplier: float, should_fail_with_overload: bool)
        """
        with self._lock:
            current = self._active_connections

        if current <= 1:
            return 1.0, False

        if current <= self.inflection_threshold:
            # Linear degradation: e.g. 5 connections -> 1.0 + (5 * 0.12) = 1.6x
            multiplier = 1.0 + (current * self.degradation_slope)
            return multiplier, False

        # Concurrency beyond inflection threshold: Quadratic growth
        excess = current - self.inflection_threshold
        quadratic_penalty = (excess ** 1.8) * self.degradation_slope
        multiplier = 1.0 + (self.inflection_threshold * self.degradation_slope) + quadratic_penalty

        # Compute probability of service failure (HTTP 503 / gateway drop)
        failure_odds = min(0.40, excess * 0.04)
        should_fail = random.random() < failure_odds

        return multiplier, should_fail

