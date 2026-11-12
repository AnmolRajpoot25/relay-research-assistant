"""
Retry Policy — exponential backoff with jitter for failed model calls.

Why jitter? Without it, all retries after a rate-limit error fire at the
same time (the "thundering herd" problem), immediately hitting the limit
again. Random jitter spreads the retries across time, giving the limit
window a chance to reset.

Design decision: We use "full jitter" (random between 0 and computed delay)
rather than "equal jitter" or "decorrelated jitter" because it produces the
widest spread of retry times, which is best for shared-quota scenarios like
Groq's org-level limits.

Reference: AWS Architecture Blog — "Exponential Backoff And Jitter"
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryOutcome:
    """Records what happened across all retry attempts."""
    succeeded: bool
    total_attempts: int
    total_delay_seconds: float  # How long we spent waiting between retries
    final_error: Optional[str] = None
    
    @property
    def retries_needed(self) -> int:
        """How many retries were needed (0 if first attempt succeeded)."""
        return max(0, self.total_attempts - 1)


class RetryPolicy:
    """
    Configurable retry policy with exponential backoff and full jitter.
    
    Usage:
        policy = RetryPolicy(max_attempts=3, base_delay=1.0)
        result, outcome = policy.execute(some_function, arg1, arg2)
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        jitter_factor: float = 0.5,
        retryable_check: Optional[Callable[[Exception], bool]] = None,
    ):
        """
        Args:
            max_attempts: Total attempts including the first try (so 3 means 1 try + 2 retries)
            base_delay_seconds: Starting delay before first retry
            max_delay_seconds: Maximum delay cap (prevents absurdly long waits)
            jitter_factor: How much randomness to add (0 = none, 1 = full jitter)
            retryable_check: Optional function that returns True if an exception should be retried
        """
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.jitter_factor = jitter_factor
        self.retryable_check = retryable_check or self._default_retryable_check
    
    def execute(self, function: Callable[..., T], *args, **kwargs) -> tuple[Optional[T], RetryOutcome]:
        """
        Execute a function with retries.
        
        Returns (result, outcome) where result is None if all attempts failed.
        """
        total_delay = 0.0
        last_error = None
        
        for attempt_number in range(1, self.max_attempts + 1):
            try:
                result = function(*args, **kwargs)
                return result, RetryOutcome(
                    succeeded=True,
                    total_attempts=attempt_number,
                    total_delay_seconds=total_delay,
                )
            except Exception as error:
                last_error = error
                
                # Check if this error is worth retrying
                if not self.retryable_check(error):
                    logger.info(
                        f"Non-retryable error on attempt {attempt_number}: {error}"
                    )
                    return None, RetryOutcome(
                        succeeded=False,
                        total_attempts=attempt_number,
                        total_delay_seconds=total_delay,
                        final_error=str(error),
                    )
                
                # Don't sleep after the last attempt
                if attempt_number < self.max_attempts:
                    delay = self._calculate_delay(attempt_number)
                    
                    # If the error includes a retry-after hint, use that instead
                    server_delay = self._extract_retry_after(error)
                    if server_delay is not None:
                        delay = max(delay, server_delay)
                    
                    logger.info(
                        f"Attempt {attempt_number}/{self.max_attempts} failed: {error}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                    total_delay += delay
        
        return None, RetryOutcome(
            succeeded=False,
            total_attempts=self.max_attempts,
            total_delay_seconds=total_delay,
            final_error=str(last_error),
        )
    
    def _calculate_delay(self, attempt_number: int) -> float:
        """
        Calculate the delay before the next retry.
        
        Formula: min(max_delay, base_delay * 2^(attempt-1)) * random(1-jitter, 1+jitter)
        """
        # Exponential: 1s, 2s, 4s, 8s, ...
        exponential_delay = self.base_delay_seconds * (2 ** (attempt_number - 1))
        
        # Cap at maximum
        capped_delay = min(exponential_delay, self.max_delay_seconds)
        
        # Add jitter: multiply by a random factor in [1-jitter, 1+jitter]
        jitter_min = 1.0 - self.jitter_factor
        jitter_max = 1.0 + self.jitter_factor
        jittered_delay = capped_delay * random.uniform(jitter_min, jitter_max)
        
        return max(0.1, jittered_delay)  # Never wait less than 100ms
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Try to extract a retry-after value from the error message."""
        import re
        error_text = str(error)
        match = re.search(r'retry[- ]?after[:\s]+(\d+\.?\d*)', error_text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        return None
    
    @staticmethod
    def _default_retryable_check(error: Exception) -> bool:
        """
        Default: retry on rate limits, timeouts, and server errors.
        Don't retry on auth errors, validation errors, etc.
        """
        error_text = str(error).lower()
        
        # Always retry rate limits
        if "429" in error_text or "rate_limit" in error_text:
            return True
        
        # Retry server errors (5xx)
        if "500" in error_text or "502" in error_text or "503" in error_text:
            return True
        
        # Retry timeouts
        if "timeout" in error_text or "timed out" in error_text:
            return True
        
        # Retry connection errors
        if "connection" in error_text:
            return True
        
        # Don't retry client errors (4xx except 429)
        if "401" in error_text or "403" in error_text or "400" in error_text:
            return False
        
        # Default: retry (be optimistic)
        return True
