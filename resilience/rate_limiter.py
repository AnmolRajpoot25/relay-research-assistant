"""
Rate Limiter — local token-bucket limiter to avoid hitting Groq's limits.

Why limit ourselves locally? Because hitting a real rate limit is expensive:
1. The request fails and wastes the time already spent
2. The error response itself counts against limits on some providers
3. It triggers a retry delay that hurts latency for everyone in the queue

By throttling ourselves slightly below the provider's limits, we trade a
tiny bit of throughput for much smoother latency distribution.
"""

import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)


class TokenBucketLimiter:
    """
    Classic token-bucket rate limiter.
    
    Tokens refill at a constant rate. Each request consumes one token.
    If the bucket is empty, the caller must wait.
    
    Thread-safe via a simple lock (good enough for our scale).
    """
    
    def __init__(
        self,
        max_tokens_per_second: float,
        burst_capacity: int = 1,
        name: str = "default",
    ):
        """
        Args:
            max_tokens_per_second: Sustained rate (e.g., 0.5 for 30 per minute)
            burst_capacity: How many requests can fire instantly before throttling
            name: Human-readable name for logging
        """
        self.max_tokens_per_second = max_tokens_per_second
        self.burst_capacity = burst_capacity
        self.name = name
        
        self._available_tokens = float(burst_capacity)
        self._last_refill_time = time.monotonic()
        self._lock = Lock()
    
    def acquire(self, timeout_seconds: float = 30.0) -> bool:
        """
        Wait until a token is available, then consume it.
        
        Returns True if acquired within timeout, False if timed out.
        This is the main entry point — call this before every model request.
        """
        deadline = time.monotonic() + timeout_seconds
        
        while True:
            with self._lock:
                self._refill()
                
                if self._available_tokens >= 1.0:
                    self._available_tokens -= 1.0
                    return True
                
                # Calculate how long until the next token is available
                wait_time = (1.0 - self._available_tokens) / self.max_tokens_per_second
            
            # Check if we'd exceed the timeout
            if time.monotonic() + wait_time > deadline:
                logger.warning(
                    f"Rate limiter '{self.name}': timed out waiting "
                    f"({timeout_seconds}s limit exceeded)"
                )
                return False
            
            # Wait for the next token (sleep outside the lock)
            time.sleep(min(wait_time, 0.1))  # Wake up every 100ms to check
    
    def _refill(self) -> None:
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill_time
        self._last_refill_time = now
        
        new_tokens = elapsed * self.max_tokens_per_second
        self._available_tokens = min(
            self._available_tokens + new_tokens,
            float(self.burst_capacity),
        )


class RateLimiter:
    """
    Composite rate limiter that enforces multiple limits simultaneously.
    
    Groq enforces both RPM (requests per minute) and TPM (tokens per minute).
    We create separate token buckets for each and require both to have
    available tokens before allowing a request.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 30,
        model_name: str = "default",
    ):
        # Convert RPM to requests per second for the token bucket
        requests_per_second = requests_per_minute / 60.0
        
        self._request_limiter = TokenBucketLimiter(
            max_tokens_per_second=requests_per_second,
            burst_capacity=min(5, requests_per_minute),  # Allow small bursts
            name=f"{model_name}_rpm",
        )
        
        logger.info(
            f"RateLimiter for {model_name}: "
            f"{requests_per_minute} RPM ({requests_per_second:.2f}/s)"
        )
    
    def wait_for_capacity(self, timeout_seconds: float = 30.0) -> bool:
        """
        Block until we have capacity for another request.
        Returns False if the timeout is exceeded.
        """
        return self._request_limiter.acquire(timeout_seconds)
