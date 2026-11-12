"""
Model Client — the single entry point the pipeline uses for all LLM calls.

Why this abstraction? The pipeline steps (question analysis, drafting, etc.)
shouldn't know or care about model fallback, rate limiting, or retries.
They just call `model_client.complete(messages)` and get a response.

This separation means we can:
1. Swap the entire model layer for tests
2. Add new providers without touching pipeline code
3. Track all model calls in one place for metrics
"""

import logging
from typing import Optional

from config import settings
from models.groq_client import GroqClient, ModelResponse
from models.fallback_router import FallbackRouter
from models.token_budget_tracker import TokenBudgetTracker

logger = logging.getLogger(__name__)


class ModelClient:
    """Unified interface for all LLM interactions in the pipeline."""

    def __init__(self, groq_api_key: Optional[str] = None):
        api_key = groq_api_key or settings.groq_api_key

        if not api_key:
            logger.warning(
                "No Groq API key provided. Only the mock service will work. "
                "Set GROQ_API_KEY in your .env file."
            )

        # Create clients for each model tier
        self._primary_client = GroqClient(
            api_key=api_key,
            model_name=settings.primary_model,
            timeout_seconds=settings.model_call_timeout,
        )

        self._fallback_client = GroqClient(
            api_key=api_key,
            model_name=settings.fallback_model,
            timeout_seconds=settings.model_call_timeout,
        )

        # Budget tracker monitors quota across both models
        self._budget_tracker = TokenBudgetTracker()

        # The router decides which model to use for each call
        self._router = FallbackRouter(
            primary_client=self._primary_client,
            fallback_client=self._fallback_client,
            budget_tracker=self._budget_tracker,
            mock_service_url=settings.mock_service_url,
            warning_threshold=settings.quota_warning_threshold,
            critical_threshold=settings.quota_critical_threshold,
        )

        # Running totals for metrics reporting
        self.total_calls = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.calls_by_model: dict[str, int] = {}

        logger.info("ModelClient initialized with fallback routing")

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stream: bool = True,
    ) -> ModelResponse:
        """
        Send a chat completion request.

        The router handles model selection and fallback automatically.
        The caller doesn't need to know which model answered.
        """
        response = self._router.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        # Update running totals
        self.total_calls += 1
        self.total_tokens_in += response.tokens_in
        self.total_tokens_out += response.tokens_out
        model = response.model_used or "unknown"
        self.calls_by_model[model] = self.calls_by_model.get(model, 0) + 1

        return response

    @property
    def forced_backend(self) -> Optional[str]:
        return self._router.forced_backend

    @forced_backend.setter
    def forced_backend(self, value: Optional[str]):
        self._router.forced_backend = value

    @property
    def current_tier(self) -> str:
        """Which model tier we'd use right now."""
        return self._router.choose_model_tier().value

    @property
    def budget_status(self) -> dict:
        """Current quota usage across all models."""
        return self._budget_tracker.get_status_summary()

    def get_metrics_summary(self) -> dict:
        """Summary of all model calls made so far."""
        return {
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
            "calls_by_model": self.calls_by_model,
            "current_tier": self.current_tier,
            "budget_status": self.budget_status,
        }
