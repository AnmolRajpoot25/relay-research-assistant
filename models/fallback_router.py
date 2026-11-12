"""
Fallback Router — ensures no request is lost when quota runs out.

The routing strategy has three tiers:
  1. Primary model (llama-3.3-70b — best quality)
  2. Fallback model (llama-3.1-8b — lower quality but higher daily limits)
  3. Mock service (when ALL real quotas are exhausted)

Switching between tiers is seamless: the pipeline doesn't know or care
which model answered. The router makes the decision based on quota state
BEFORE sending the request, so we never waste tokens on a request that
will be rejected.

Critical design decision: we switch proactively at 80% usage (configurable),
not at 100%. This gives us a buffer for in-flight requests and for the
slight delay between our local tracking and the server's actual state.
"""

import logging
from enum import Enum
from typing import Optional

from openai import OpenAI

from models.groq_client import GroqClient, ModelResponse
from models.token_budget_tracker import TokenBudgetTracker

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Which tier of service we're currently using."""
    PRIMARY = "primary"
    FALLBACK = "fallback"
    MOCK = "mock"


class FallbackRouter:
    """Routes model calls to the best available model based on quota state."""

    def __init__(
        self,
        primary_client: GroqClient,
        fallback_client: GroqClient,
        budget_tracker: TokenBudgetTracker,
        mock_service_url: str = "http://localhost:8001",
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.95,
    ):
        self.primary_client = primary_client
        self.fallback_client = fallback_client
        self.budget_tracker = budget_tracker
        self.mock_service_url = mock_service_url

        # Thresholds for switching tiers
        self.warning_threshold = warning_threshold    # Switch primary → fallback
        self.critical_threshold = critical_threshold  # Switch fallback → mock

        # Register both models with the budget tracker
        self.budget_tracker.register_model(primary_client.model_name)
        self.budget_tracker.register_model(fallback_client.model_name)

        # Optional: mock client (created lazily when needed)
        self._mock_client: Optional[GroqClient] = None
        self.forced_backend: Optional[str] = None

        logger.info(
            f"FallbackRouter ready: "
            f"primary={primary_client.model_name}, "
            f"fallback={fallback_client.model_name}, "
            f"switch at {warning_threshold:.0%}/{critical_threshold:.0%}"
        )

    def choose_model_tier(self, estimated_tokens: int = 2000) -> ModelTier:
        """
        Decide which model tier to use right now.

        This is called BEFORE each model request. It checks quota state
        and returns the best available tier.
        """
        # Strict enforcement when a scenario specifies an explicit backend
        if self.forced_backend == "stand-in":
            return ModelTier.MOCK
        elif self.forced_backend == "real":
            if not self.primary_client.has_valid_credentials:
                raise RuntimeError("Forced backend is 'real', but no valid GROQ_API_KEY credentials exist.")
            return ModelTier.PRIMARY

        # If no valid Groq credentials are configured, default to stand-in mock
        if not self.primary_client.has_valid_credentials:
            return ModelTier.MOCK

        # Check primary model
        primary_usage = self.budget_tracker.get_usage_fraction(self.primary_client.model_name)
        primary_can_afford = self.budget_tracker.can_afford_request(
            self.primary_client.model_name, estimated_tokens
        )

        if primary_usage < self.warning_threshold and primary_can_afford:
            return ModelTier.PRIMARY

        # Primary is running hot — try fallback
        fallback_usage = self.budget_tracker.get_usage_fraction(self.fallback_client.model_name)
        fallback_can_afford = self.budget_tracker.can_afford_request(
            self.fallback_client.model_name, estimated_tokens
        )

        if fallback_usage < self.critical_threshold and fallback_can_afford:
            if primary_usage >= self.warning_threshold:
                logger.info(
                    f"Primary model at {primary_usage:.0%} usage — "
                    f"switching to fallback ({self.fallback_client.model_name})"
                )
            return ModelTier.FALLBACK

        # Both real models are exhausted — fall back to mock
        logger.warning(
            f"All real model quotas near exhaustion "
            f"(primary: {primary_usage:.0%}, fallback: {fallback_usage:.0%}) "
            f"— routing to mock service"
        )
        return ModelTier.MOCK

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stream: bool = True,
        estimated_tokens: int = 2000,
    ) -> ModelResponse:
        """
        Send a completion request to the best available model.

        If the chosen model fails with a rate-limit error, we automatically
        try the next tier. This ensures no request is lost — the worst case
        is a slightly slower or lower-quality response.
        """
        tier = self.choose_model_tier(estimated_tokens)

        # Try the chosen tier, falling through on rate-limit errors
        last_response: Optional[ModelResponse] = None
        for attempt_tier in self._tier_cascade(starting_from=tier):
            client = self._get_client_for_tier(attempt_tier)
            if client is None:
                continue

            response = client.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            last_response = response

            # Update our budget tracker with actual usage
            if not response.error_message:
                total_tokens = response.tokens_in + response.tokens_out
                self.budget_tracker.record_usage(response.model_used, total_tokens)

                # Sync with server-reported limits if available
                rate_state = response.rate_limit_state
                if rate_state.remaining_requests >= 0:
                    self.budget_tracker.sync_with_server_state(
                        response.model_used,
                        rate_state.remaining_requests,
                        rate_state.remaining_tokens,
                    )

                return response

            # If rate-limited, try the next tier
            if response.was_rate_limited:
                logger.info(f"Rate limited on {attempt_tier.value} — trying next tier")
                continue

            # For non-rate-limit errors, still return the response
            # (the pipeline's retry policy will decide what to do)
            return response

        # All tiers exhausted
        was_rl = last_response.was_rate_limited if last_response else False
        rate_st = last_response.rate_limit_state if last_response else RateLimitState()
        return ModelResponse(
            error_message="All model tiers exhausted (primary, fallback, and mock)",
            model_used="none",
            was_rate_limited=was_rl,
            rate_limit_state=rate_st,
        )

    def _tier_cascade(self, starting_from: ModelTier) -> list[ModelTier]:
        """Return the tiers to try in order, respecting forced backend boundaries."""
        if self.forced_backend == "stand-in":
            return [ModelTier.MOCK]
        elif self.forced_backend == "real":
            return [ModelTier.PRIMARY, ModelTier.FALLBACK]

        all_tiers = [ModelTier.PRIMARY, ModelTier.FALLBACK, ModelTier.MOCK]
        start_index = all_tiers.index(starting_from)
        return all_tiers[start_index:]

    def _get_client_for_tier(self, tier: ModelTier) -> Optional[GroqClient]:
        """Get the GroqClient for a given tier."""
        if tier == ModelTier.PRIMARY:
            return self.primary_client
        elif tier == ModelTier.FALLBACK:
            return self.fallback_client
        elif tier == ModelTier.MOCK:
            return self._get_or_create_mock_client()
        return None

    def _get_or_create_mock_client(self) -> Optional[GroqClient]:
        """Create a client pointing at the mock service (same API shape as Groq)."""
        if self._mock_client is None:
            try:
                self._mock_client = GroqClient(
                    api_key="mock-key",
                    model_name="mock-model",
                    timeout_seconds=10.0,
                )
                # Override the base URL to point at our mock
                self._mock_client._client = OpenAI(
                    base_url=f"{self.mock_service_url}/v1",
                    api_key="mock-key",
                    max_retries=0,
                    timeout=10.0,
                )
                logger.info(f"Mock client created pointing at {self.mock_service_url}")
            except Exception as e:
                logger.error(f"Failed to create mock client: {e}")
                return None
        return self._mock_client

    def get_current_status(self) -> dict:
        """Get a summary of which tier we'd use and why."""
        return {
            "current_tier": self.choose_model_tier().value,
            "budget_status": self.budget_tracker.get_status_summary(),
        }
