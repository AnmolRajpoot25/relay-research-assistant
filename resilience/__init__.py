"""
Resilience module — protects the pipeline from failures and overload.

Components:
  - RetryPolicy: exponential backoff with jitter
  - RateLimiter: local token-bucket rate limiting
  - RequestQueue: priority queue with timeout for incoming requests
  - CircuitBreaker: stops hammering a failing service
"""
from resilience.retry_policy import RetryPolicy
from resilience.rate_limiter import RateLimiter
from resilience.request_queue import RequestQueue
from resilience.circuit_breaker import CircuitBreaker
