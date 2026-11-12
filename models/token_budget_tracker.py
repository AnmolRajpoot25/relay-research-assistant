"""
Token Budget Tracker — proactive quota monitoring.

The critical insight: we don't want to DISCOVER we've hit a limit by getting
a 429 error mid-pipeline. That wastes the tokens already spent on earlier
steps. Instead, we track usage ourselves and switch models BEFORE hitting
the wall.

This tracker is conservative — it'd rather switch to a fallback slightly
early than waste tokens on a request that will be rejected.
"""

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

from config import GroqRateLimits

logger = logging.getLogger(__name__)


@dataclass
class UsageWindow:
    """Tracks usage within a time window (minute or day)."""
    window_duration_seconds: float      # 60 for per-minute, 86400 for per-day
    max_requests: int
    max_tokens: int

    # Rolling counters
    requests_used: int = 0
    tokens_used: int = 0
    window_start_time: float = field(default_factory=time.time)

    def record_usage(self, tokens: int) -> None:
        """Record that we made a request using this many tokens."""
        self._maybe_reset_window()
        self.requests_used += 1
        self.tokens_used += tokens

    def has_budget_for(self, estimated_tokens: int) -> bool:
        """Check if we can afford another request of this size."""
        self._maybe_reset_window()
        return (
            self.requests_used < self.max_requests
            and self.tokens_used + estimated_tokens < self.max_tokens
        )

    @property
    def usage_fraction(self) -> float:
        """The higher of request or token usage as a fraction (0.0 to 1.0)."""
        self._maybe_reset_window()
        request_fraction = self.requests_used / max(self.max_requests, 1)
        token_fraction = self.tokens_used / max(self.max_tokens, 1)
        return max(request_fraction, token_fraction)

    @property
    def seconds_until_reset(self) -> float:
        """How long until this window resets."""
        elapsed = time.time() - self.window_start_time
        remaining = self.window_duration_seconds - elapsed
        return max(0.0, remaining)

    def _maybe_reset_window(self) -> None:
        """Reset counters if the current window has expired."""
        elapsed = time.time() - self.window_start_time
        if elapsed >= self.window_duration_seconds:
            self.requests_used = 0
            self.tokens_used = 0
            self.window_start_time = time.time()

    def update_from_server(self, remaining_requests: int, remaining_tokens: int) -> None:
        """
        Sync our local counters with the server's reported remaining values.
        The server is the source of truth — if it says we have fewer remaining
        than we think (e.g., because another process used the same API key),
        we trust the server.
        """
        if remaining_requests >= 0:
            server_used = self.max_requests - remaining_requests
            if server_used > self.requests_used:
                logger.info(
                    f"Server reports higher usage ({server_used} requests) "
                    f"than local tracking ({self.requests_used}). Syncing."
                )
                self.requests_used = server_used

        if remaining_tokens >= 0:
            server_used = self.max_tokens - remaining_tokens
            if server_used > self.tokens_used:
                self.tokens_used = server_used


class TokenBudgetTracker:
    """
    Monitors quota usage across models and decides when to switch.

    Thread-safe: multiple concurrent pipeline requests may update usage
    simultaneously.
    """

    def __init__(self):
        self._lock = Lock()
        self._model_trackers: dict[str, dict[str, UsageWindow]] = {}
        logger.info("TokenBudgetTracker initialized")

    def register_model(self, model_name: str) -> None:
        """Set up tracking for a model based on known Groq limits."""
        limits = GroqRateLimits.get_limits(model_name)

        with self._lock:
            self._model_trackers[model_name] = {
                "per_minute": UsageWindow(
                    window_duration_seconds=60,
                    max_requests=limits["requests_per_minute"],
                    max_tokens=limits["tokens_per_minute"],
                ),
                "per_day": UsageWindow(
                    window_duration_seconds=86400,
                    max_requests=limits["requests_per_day"],
                    max_tokens=limits["tokens_per_day"],
                ),
            }

        logger.info(
            f"Registered model '{model_name}' — "
            f"{limits['requests_per_minute']} RPM, {limits['tokens_per_minute']} TPM, "
            f"{limits['requests_per_day']} RPD, {limits['tokens_per_day']} TPD"
        )

    def record_usage(self, model_name: str, total_tokens: int) -> None:
        """Record that a request consumed tokens on this model."""
        with self._lock:
            if model_name not in self._model_trackers:
                logger.warning(f"Usage recorded for unregistered model: {model_name}")
                return

            for window in self._model_trackers[model_name].values():
                window.record_usage(total_tokens)

    def can_afford_request(
        self,
        model_name: str,
        estimated_tokens: int = 2000,
    ) -> bool:
        """
        Check if we can afford another request on this model.

        The estimated_tokens parameter should be a conservative estimate of
        total tokens (prompt + completion) for the request. When in doubt,
        overestimate — it's better to switch to a fallback slightly early
        than to waste tokens on a rejected request.
        """
        with self._lock:
            if model_name not in self._model_trackers:
                return True  # Unknown model — let it try

            return all(
                window.has_budget_for(estimated_tokens)
                for window in self._model_trackers[model_name].values()
            )

    def get_usage_fraction(self, model_name: str) -> float:
        """
        Get the highest usage fraction across all windows for this model.
        Returns 0.0 (empty) to 1.0 (fully used).
        """
        with self._lock:
            if model_name not in self._model_trackers:
                return 0.0

            return max(
                window.usage_fraction
                for window in self._model_trackers[model_name].values()
            )

    def sync_with_server_state(
        self,
        model_name: str,
        remaining_requests: int,
        remaining_tokens: int,
    ) -> None:
        """Update local tracking from Groq's response headers."""
        with self._lock:
            if model_name not in self._model_trackers:
                return
            # Only update the per-minute window from headers
            # (daily limits aren't in the standard headers)
            self._model_trackers[model_name]["per_minute"].update_from_server(
                remaining_requests, remaining_tokens
            )

    def get_status_summary(self) -> dict[str, dict]:
        """Get a human-readable status of all tracked models."""
        with self._lock:
            summary = {}
            for model_name, windows in self._model_trackers.items():
                summary[model_name] = {
                    "per_minute": {
                        "usage": f"{windows['per_minute'].usage_fraction:.1%}",
                        "requests": f"{windows['per_minute'].requests_used}/{windows['per_minute'].max_requests}",
                        "tokens": f"{windows['per_minute'].tokens_used}/{windows['per_minute'].max_tokens}",
                        "resets_in": f"{windows['per_minute'].seconds_until_reset:.0f}s",
                    },
                    "per_day": {
                        "usage": f"{windows['per_day'].usage_fraction:.1%}",
                        "requests": f"{windows['per_day'].requests_used}/{windows['per_day'].max_requests}",
                        "tokens": f"{windows['per_day'].tokens_used}/{windows['per_day'].max_tokens}",
                    },
                }
            return summary
