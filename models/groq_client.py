"""
Groq Client — thin wrapper around the OpenAI SDK configured for Groq.

Why use the OpenAI SDK instead of raw HTTP? Groq's API is fully OpenAI-compatible,
and the SDK handles streaming, retries, and serialization reliably. We disable the
SDK's built-in retries because our pipeline has its own retry policy that needs to
see every failure.

Key design: Every call returns a ModelResponse that includes timing, token usage,
and the current rate-limit state from response headers. The FallbackRouter reads
these to decide when to switch models.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Current rate-limit counters extracted from Groq's response headers."""
    remaining_requests: int = -1       # -1 means unknown
    remaining_tokens: int = -1
    limit_requests: int = -1
    limit_tokens: int = -1
    reset_requests_seconds: float = 0  # Seconds until request limit resets
    reset_tokens_seconds: float = 0    # Seconds until token limit resets
    retry_after_seconds: float = 0     # Only set on 429 responses

    @property
    def request_usage_fraction(self) -> float:
        """How much of the request quota has been used (0.0 to 1.0)."""
        if self.limit_requests <= 0:
            return 0.0
        return 1.0 - (self.remaining_requests / self.limit_requests)

    @property
    def token_usage_fraction(self) -> float:
        """How much of the token quota has been used (0.0 to 1.0)."""
        if self.limit_tokens <= 0:
            return 0.0
        return 1.0 - (self.remaining_tokens / self.limit_tokens)


@dataclass
class ModelResponse:
    """Complete result from a model call, including all the metadata we need."""
    content: str = ""
    model_used: str = ""
    tokens_in: int = 0                   # Prompt tokens
    tokens_out: int = 0                  # Completion tokens
    time_to_first_token_ms: float = 0    # Milliseconds to first streamed token
    total_time_ms: float = 0             # Milliseconds for the full response
    rate_limit_state: RateLimitState = field(default_factory=RateLimitState)
    was_rate_limited: bool = False
    error_message: str = ""
    is_from_mock: bool = False


class GroqClient:
    """
    Communicates with Groq's hosted LLM API.

    Each instance is bound to a specific model. The FallbackRouter creates
    one GroqClient per available model and switches between them.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        timeout_seconds: float = 30.0,
    ):
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key if (api_key and api_key.strip()) else "unconfigured_key"
        self.has_valid_credentials = (self.api_key != "unconfigured_key")

        # Disable the SDK's built-in retries — we handle retries ourselves
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key,
            max_retries=0,
            timeout=timeout_seconds,
        )

        # Track the most recent rate-limit state
        self.latest_rate_limit_state = RateLimitState()

        logger.info(f"GroqClient initialized for model: {model_name}")

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stream: bool = True,
    ) -> ModelResponse:
        """
        Send a chat completion request to Groq.

        When stream=True, we measure time-to-first-token accurately by
        recording when the first chunk arrives. This is critical for the
        assignment's latency measurements.
        """
        response = ModelResponse(model_used=self.model_name)
        call_start_time = time.perf_counter()

        try:
            if stream:
                response = self._complete_streaming(
                    messages, temperature, max_tokens, call_start_time
                )
            else:
                response = self._complete_non_streaming(
                    messages, temperature, max_tokens, call_start_time
                )
        except Exception as api_error:
            error_name = type(api_error).__name__
            error_text = str(api_error)
            response.error_message = f"{error_name}: {error_text}"
            response.total_time_ms = (time.perf_counter() - call_start_time) * 1000

            # Check if this is a rate-limit error (HTTP 429)
            if "429" in error_text or "rate_limit" in error_text.lower():
                response.was_rate_limited = True
                # Try to extract retry-after from the error
                self._extract_retry_after_from_error(api_error, response)
                logger.warning(
                    f"Rate limited on {self.model_name}: "
                    f"retry after {response.rate_limit_state.retry_after_seconds}s"
                )
            else:
                logger.error(f"Model call failed ({self.model_name}): {response.error_message}")

        # Mark source fidelity
        base_url_str = str(getattr(self._client, "base_url", ""))
        if "mock" in self.model_name.lower() or "127.0.0.1" in base_url_str or "localhost" in base_url_str:
            response.is_from_mock = True
        else:
            response.is_from_mock = False

        return response

    def _complete_streaming(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        call_start_time: float,
    ) -> ModelResponse:
        """Handle a streaming completion, measuring time-to-first-token."""
        response = ModelResponse(model_used=self.model_name)
        collected_content: list[str] = []
        first_token_received = False

        stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        for chunk in stream:
            if not first_token_received and chunk.choices and chunk.choices[0].delta.content:
                response.time_to_first_token_ms = (time.perf_counter() - call_start_time) * 1000
                first_token_received = True

            if chunk.choices and chunk.choices[0].delta.content:
                collected_content.append(chunk.choices[0].delta.content)

            # The last chunk in Groq's stream includes usage info
            if hasattr(chunk, 'usage') and chunk.usage is not None:
                response.tokens_in = chunk.usage.prompt_tokens
                response.tokens_out = chunk.usage.completion_tokens

        response.content = "".join(collected_content)
        response.total_time_ms = (time.perf_counter() - call_start_time) * 1000

        # Extract rate limit headers from the stream's response
        if hasattr(stream, 'response') and stream.response is not None:
            self._update_rate_limit_state_from_headers(stream.response.headers, response)

        return response

    def _complete_non_streaming(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        call_start_time: float,
    ) -> ModelResponse:
        """Handle a non-streaming completion."""
        response = ModelResponse(model_used=self.model_name)

        completion = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        end_time = time.perf_counter()
        response.content = completion.choices[0].message.content or ""
        response.tokens_in = completion.usage.prompt_tokens if completion.usage else 0
        response.tokens_out = completion.usage.completion_tokens if completion.usage else 0
        response.time_to_first_token_ms = (end_time - call_start_time) * 1000  # Same as total for non-streaming
        response.total_time_ms = (end_time - call_start_time) * 1000

        # Extract rate limit state from response headers
        if hasattr(completion, '_raw_response') and completion._raw_response is not None:
            self._update_rate_limit_state_from_headers(completion._raw_response.headers, response)

        return response

    def _update_rate_limit_state_from_headers(self, headers: dict, response: ModelResponse) -> None:
        """Parse Groq's rate-limit headers into our RateLimitState."""
        state = RateLimitState()

        def safe_int(header_name: str, default: int = -1) -> int:
            value = headers.get(header_name)
            return int(value) if value is not None else default

        def safe_float(header_name: str, default: float = 0.0) -> float:
            value = headers.get(header_name)
            if value is None:
                return default
            # Groq sometimes returns "1.234s" format
            return float(value.rstrip('s')) if isinstance(value, str) else float(value)

        state.remaining_requests = safe_int("x-ratelimit-remaining-requests")
        state.remaining_tokens = safe_int("x-ratelimit-remaining-tokens")
        state.limit_requests = safe_int("x-ratelimit-limit-requests")
        state.limit_tokens = safe_int("x-ratelimit-limit-tokens")
        state.reset_requests_seconds = safe_float("x-ratelimit-reset-requests")
        state.reset_tokens_seconds = safe_float("x-ratelimit-reset-tokens")
        state.retry_after_seconds = safe_float("retry-after")

        response.rate_limit_state = state
        self.latest_rate_limit_state = state

    def _extract_retry_after_from_error(self, error: Exception, response: ModelResponse) -> None:
        """Try to get retry-after info from a rate-limit error."""
        error_text = str(error)
        # Look for patterns like "retry after 2.5s" or "retry-after: 3"
        match = re.search(r'retry[- ]after[:\s]+(\d+\.?\d*)', error_text, re.IGNORECASE)
        if match:
            response.rate_limit_state.retry_after_seconds = float(match.group(1))
