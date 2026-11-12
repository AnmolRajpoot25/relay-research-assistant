"""
Circuit Breaker — stops sending requests to a failing service.

The circuit breaker pattern prevents cascading failures. If a service starts
failing (e.g., Groq returns 503s), we "open" the circuit and stop sending
requests for a cooldown period. This:

1. Gives the failing service time to recover
2. Stops wasting tokens on requests that will fail
3. Allows the fallback router to redirect to another model immediately

States:
  CLOSED → calls pass through normally
  OPEN → calls are rejected immediately (fail-fast)
  HALF_OPEN → let one test call through; if it succeeds, close the circuit
"""

import logging
import time
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"         # Normal operation
    OPEN = "open"             # Failing — reject calls immediately
    HALF_OPEN = "half_open"   # Testing if service has recovered


class CircuitBreaker:
    """
    Circuit breaker with automatic state transitions.
    
    Usage:
        breaker = CircuitBreaker(name="groq-primary")
        
        if not breaker.allow_request():
            # Circuit is open — use fallback
            return fallback_response()
        
        try:
            result = make_api_call()
            breaker.record_success()
            return result
        except Exception as e:
            breaker.record_failure()
            raise
    """
    
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        success_threshold: int = 2,
    ):
        """
        Args:
            name: Human-readable name for logging
            failure_threshold: Consecutive failures before opening the circuit
            recovery_timeout_seconds: How long to stay open before trying half-open
            success_threshold: Consecutive successes in half-open before closing
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._last_failure_time = 0.0
        self._lock = Lock()
        
        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"opens after {failure_threshold} failures, "
            f"recovers after {recovery_timeout_seconds}s"
        )
    
    @property
    def state(self) -> CircuitState:
        """Current state of the circuit breaker."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state
    
    def allow_request(self) -> bool:
        """
        Should we allow a request through?
        
        CLOSED: yes
        OPEN: no (fail fast)
        HALF_OPEN: yes (one test request at a time)
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                return True  # Allow the test request
            else:  # OPEN
                logger.debug(
                    f"CircuitBreaker '{self.name}' is OPEN — "
                    f"rejecting request (recovering in "
                    f"{self._seconds_until_half_open():.1f}s)"
                )
                return False
    
    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._consecutive_successes += 1
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_failures = 0
                    self._consecutive_successes = 0
                    logger.info(f"CircuitBreaker '{self.name}': HALF_OPEN → CLOSED (recovered)")
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0  # Reset on success
    
    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._last_failure_time = time.monotonic()
            
            if self._state == CircuitState.HALF_OPEN:
                # Test request failed — back to open
                self._state = CircuitState.OPEN
                logger.info(f"CircuitBreaker '{self.name}': HALF_OPEN → OPEN (test failed)")
                
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"CircuitBreaker '{self.name}': CLOSED → OPEN "
                        f"({self._consecutive_failures} consecutive failures)"
                    )
    
    def _maybe_transition_to_half_open(self) -> None:
        """Check if enough time has passed to try half-open (must hold lock)."""
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout_seconds:
                self._state = CircuitState.HALF_OPEN
                self._consecutive_successes = 0
                logger.info(
                    f"CircuitBreaker '{self.name}': OPEN → HALF_OPEN "
                    f"(testing after {elapsed:.1f}s)"
                )
    
    def _seconds_until_half_open(self) -> float:
        """How long until we'll try half-open (must hold lock)."""
        if self._state != CircuitState.OPEN:
            return 0.0
        elapsed = time.monotonic() - self._last_failure_time
        return max(0.0, self.recovery_timeout_seconds - elapsed)
    
    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            previous_state = self._state
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            logger.info(f"CircuitBreaker '{self.name}': {previous_state.value} → CLOSED (manual reset)")
