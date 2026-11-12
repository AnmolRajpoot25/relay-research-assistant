"""
Request Queue — manages incoming requests with priority and timeouts.

When load exceeds capacity, requests queue up. But users don't wait forever:
they have a patience timeout (configurable, default 45s). If a request has
been waiting longer than the patience timeout, we drop it rather than
process it — the user has probably already left.

Priority levels allow us to prioritize certain requests (e.g., retries of
partially-completed pipelines) over fresh requests.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from queue import PriorityQueue, Empty
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """Lower number = higher priority (PriorityQueue is a min-heap)."""
    RETRY = 0           # Retries of partially-completed work (most valuable)
    NORMAL = 1          # Regular incoming requests
    LOW_PRIORITY = 2    # Best-effort requests (e.g., pre-fetching)


@dataclass(order=True)
class QueuedRequest:
    """A request waiting in the queue."""
    priority: int = field(compare=True)
    enqueued_at: float = field(compare=True, default_factory=time.time)
    
    # These fields don't participate in ordering
    request_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4())[:8])
    question: str = field(compare=False, default="")
    metadata: dict = field(compare=False, default_factory=dict)
    
    @property
    def wait_time_seconds(self) -> float:
        """How long this request has been waiting."""
        return time.time() - self.enqueued_at


class RequestQueue:
    """
    Thread-safe priority queue with patience-based eviction.
    
    Design: We check patience timeout at dequeue time, not in a background
    thread. This is simpler and good enough — a request that times out
    between checks just gets dropped a little late, which is fine.
    """
    
    def __init__(
        self,
        max_queue_size: int = 100,
        patience_timeout_seconds: float = 45.0,
    ):
        self._queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        self.patience_timeout_seconds = patience_timeout_seconds
        self.max_queue_size = max_queue_size
        
        # Metrics
        self.total_enqueued = 0
        self.total_dropped_patience = 0   # Dropped because user would have given up
        self.total_dropped_full = 0       # Dropped because queue was full
        self.peak_queue_length = 0
        
        logger.info(
            f"RequestQueue initialized: max_size={max_queue_size}, "
            f"patience={patience_timeout_seconds}s"
        )
    
    def enqueue(
        self,
        question: str,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[dict] = None,
    ) -> Optional[QueuedRequest]:
        """
        Add a request to the queue.
        Returns the QueuedRequest if accepted, None if the queue is full.
        """
        request = QueuedRequest(
            priority=priority.value,
            question=question,
            metadata=metadata or {},
        )
        
        try:
            self._queue.put_nowait(request)
            self.total_enqueued += 1
            current_size = self._queue.qsize()
            self.peak_queue_length = max(self.peak_queue_length, current_size)
            
            logger.debug(
                f"Enqueued request {request.request_id} "
                f"(priority={priority.name}, queue_size={current_size})"
            )
            return request
            
        except Exception:
            self.total_dropped_full += 1
            logger.warning(
                f"Queue full ({self.max_queue_size}) — dropping request. "
                f"Total dropped: {self.total_dropped_full}"
            )
            return None
    
    def dequeue(self, timeout_seconds: float = 1.0) -> Optional[QueuedRequest]:
        """
        Get the next request from the queue.
        
        Automatically drops requests that have exceeded the patience timeout.
        Returns None if the queue is empty or all pending requests have expired.
        """
        while True:
            try:
                request = self._queue.get(timeout=timeout_seconds)
            except Empty:
                return None
            
            # Check if this request has been waiting too long
            if request.wait_time_seconds > self.patience_timeout_seconds:
                self.total_dropped_patience += 1
                logger.info(
                    f"Dropping request {request.request_id} — waited "
                    f"{request.wait_time_seconds:.1f}s (patience: {self.patience_timeout_seconds}s). "
                    f"The user has probably given up."
                )
                continue  # Skip this one, try the next
            
            return request
    
    @property
    def current_size(self) -> int:
        """How many requests are currently waiting."""
        return self._queue.qsize()
    
    def get_metrics(self) -> dict:
        """Queue health metrics for reporting."""
        return {
            "current_size": self.current_size,
            "peak_size": self.peak_queue_length,
            "total_enqueued": self.total_enqueued,
            "dropped_patience": self.total_dropped_patience,
            "dropped_full": self.total_dropped_full,
        }
