"""
Unit & Integration Tests for Resilience Components: RetryPolicy, RequestQueue, CircuitBreaker.
"""

import time
import pytest
from resilience.retry_policy import RetryPolicy
from resilience.request_queue import RequestQueue, RequestPriority
from resilience.circuit_breaker import CircuitBreaker, CircuitState


def test_retry_policy_succeeds_on_transient_error():
    attempts = 0

    def flaky_service():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectionError("Transient network glitch")
        return "SUCCESS"

    policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.01)
    result, outcome = policy.execute(flaky_service)
    assert result == "SUCCESS"
    assert outcome.succeeded is True
    assert outcome.total_attempts == 2


def test_retry_policy_fails_gracefully_when_exhausted():
    def always_failing():
        raise TimeoutError("Model timed out")

    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.01)
    result, outcome = policy.execute(always_failing)
    assert result is None
    assert outcome.succeeded is False
    assert outcome.total_attempts == 2


def test_request_queue_patience_timeout_eviction():
    # Queue with short patience timeout (0.05 seconds)
    queue = RequestQueue(max_queue_size=10, patience_timeout_seconds=0.05)
    req = queue.enqueue("Test prompt", priority=RequestPriority.NORMAL)
    assert req is not None

    # Wait for patience to expire
    time.sleep(0.08)

    # Dequeue should evict the stale request and return None
    dequeued = queue.dequeue(timeout_seconds=0.01)
    assert dequeued is None
    assert queue.total_dropped_patience == 1


def test_circuit_breaker_transitions_to_open_on_repeated_failures():
    breaker = CircuitBreaker(name="test_breaker", failure_threshold=2, recovery_timeout_seconds=0.1)
    assert breaker.allow_request() is True
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow_request() is False

