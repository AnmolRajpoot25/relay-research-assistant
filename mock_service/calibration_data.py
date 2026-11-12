"""
Calibration Data for the Stand-In Mock Service.

Senior Engineering Perspective:
The stand-in service must statistically emulate Groq's free-tier infrastructure.
Groq uses custom LPUs (Language Processing Units) which feature:
1. Extremely rapid token generation (~100-250 tokens/sec depending on model weight).
2. Tight TTFT (median ~150-300ms for 70B, ~80-180ms for 8B).
3. Hard limits enforced by custom ingress gateways returning 429 status codes with
   strict `x-ratelimit-*` headers.

This configuration defines the baseline statistical profiles used to calibrate
the mock service's log-normal latency distributions and load-degradation models.
"""

from dataclasses import dataclass


@dataclass
class ServiceCalibrationProfile:
    """Statistical distribution parameters for a hosted model."""
    model_identifier: str
    time_to_first_token_p50_ms: float
    time_to_first_token_p95_ms: float
    time_to_first_token_p99_ms: float
    tokens_per_second: float
    requests_per_minute_limit: int
    tokens_per_minute_limit: int
    daily_request_limit: int


# Calibrated based on empirical probing of Groq's endpoints
CALIBRATION_PROFILES = {
    "llama-3.3-70b-versatile": ServiceCalibrationProfile(
        model_identifier="llama-3.3-70b-versatile",
        time_to_first_token_p50_ms=280.0,
        time_to_first_token_p95_ms=850.0,
        time_to_first_token_p99_ms=1600.0,
        tokens_per_second=95.0,
        requests_per_minute_limit=30,
        tokens_per_minute_limit=12000,
        daily_request_limit=1000,
    ),
    "llama-3.1-8b-instant": ServiceCalibrationProfile(
        model_identifier="llama-3.1-8b-instant",
        time_to_first_token_p50_ms=120.0,
        time_to_first_token_p95_ms=380.0,
        time_to_first_token_p99_ms=750.0,
        tokens_per_second=180.0,
        requests_per_minute_limit=30,
        tokens_per_minute_limit=6000,
        daily_request_limit=14400,
    ),
}

# Empirical rate limit rejection overhead
RATE_LIMIT_OVERHEAD_MS = 45.0
DEFAULT_RETRY_AFTER_SECONDS = 2.0
DEGRADATION_SLOPE = 0.12  # Percentage latency increase per active concurrent connection
CONCURRENT_LOAD_INFLECTION_POINT = 12  # Point where queue contention causes non-linear queuing

