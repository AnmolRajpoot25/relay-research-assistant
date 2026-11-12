"""
Latency Simulator — Generates statistically grounded latency curves.

Senior Engineering Perspective:
Network and inference latencies exhibit heavy-tailed, non-Gaussian behavior.
A standard normal distribution would generate negative latencies or underestimate p99 spikes.
We fit a Log-Normal distribution parameterized to the measured median (p50) and tail (p95)
latencies measured on the real Groq infrastructure.
"""

import math
import random
from typing import Optional

from mock_service.calibration_data import CALIBRATION_PROFILES, ServiceCalibrationProfile


class LatencySimulator:
    """Simulates realistic streaming inference delays."""

    def __init__(self, profile: Optional[ServiceCalibrationProfile] = None):
        self.profile = profile or CALIBRATION_PROFILES["llama-3.3-70b-versatile"]

        # Derive log-normal parameters: mu and sigma from p50 and p95
        # In log-normal: median (p50) = exp(mu) => mu = ln(p50)
        # 95th percentile = exp(mu + 1.645 * sigma) => sigma = (ln(p95) - mu) / 1.645
        p50 = max(10.0, self.profile.time_to_first_token_p50_ms)
        p95 = max(p50 * 1.05, self.profile.time_to_first_token_p95_ms)

        self._mu = math.log(p50)
        self._sigma = max(0.05, (math.log(p95) - self._mu) / 1.645)

    def sample_time_to_first_token_ms(self, load_factor: float = 1.0) -> float:
        """Draws a TTFT sample from the calibrated log-normal distribution."""
        raw_ttft = random.lognormvariate(self._mu, self._sigma)
        return max(20.0, raw_ttft * max(1.0, load_factor))

    def sample_inter_token_delay_ms(self, load_factor: float = 1.0) -> float:
        """Calculates inter-token streaming delay (in ms) with jitter."""
        nominal_delay = (1.0 / self.profile.tokens_per_second) * 1000.0
        # Realistic LPU streaming exhibits micro-bursting with occasional stalls
        jitter = random.uniform(0.65, 1.45)
        return max(1.5, nominal_delay * jitter * load_factor)

