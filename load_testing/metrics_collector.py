"""
Metrics Collector — Records per-request telemetries and aggregates statistical metrics.

Senior Engineering Perspective:
Per the assignment brief:
- "Every number goes in these tables, completely filled in. Reviewers read these first
   and recompute them from your raw file, so they must match."
- "Raw file: Submit one file named results.csv with one row per request."

This collector ensures data integrity, thread-safe metric appending, and exact percentile
calculations (p50, p95, p99) matching numpy's standard linear interpolation.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("MetricsCollector")


@dataclass
class RequestLogRecord:
    """Exact schema matching the assignment brief's results.csv specifications."""
    request_id: str
    scenario: str
    source: str                     # "real" or "stand-in"
    start_time: str                 # ISO timestamp
    first_token_ms: float           # Milliseconds
    total_ms: float                 # Milliseconds
    status: str                     # "ok", "rate_limited", "timeout", "error"
    retries: int                    # Non-negative integer
    tokens_in: int                  # Prompt tokens consumed
    tokens_out: int                 # Completion tokens generated
    retrieval_setting: str          # e.g. "chunk_512_top5"
    answer_grade: str               # "correct", "wrong", "refused_correctly", "refused_wrongly"

    def to_csv_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario": self.scenario,
            "source": self.source,
            "start_time": self.start_time,
            "first_token_ms": round(self.first_token_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "status": self.status,
            "retries": self.retries,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "retrieval_setting": self.retrieval_setting,
            "answer_grade": self.answer_grade,
        }


class MetricsCollector:
    """Thread-safe telemetry accumulator for load testing runs."""

    def __init__(self):
        self._records: List[RequestLogRecord] = []
        self._scenario_peak_queues: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, record: RequestLogRecord) -> None:
        with self._lock:
            self._records.append(record)

    def record_peak_queue_length(self, scenario_name: str, peak_queue: int) -> None:
        with self._lock:
            self._scenario_peak_queues[scenario_name] = max(
                self._scenario_peak_queues.get(scenario_name, 0), peak_queue
            )

    def get_all_records(self) -> List[RequestLogRecord]:
        with self._lock:
            return list(self._records)

    def compute_scenario_statistics(self, scenario_name: str) -> Dict[str, Any]:
        """Calculates Table 2a and Table 2b metrics for a given scenario."""
        with self._lock:
            matching = [r for r in self._records if r.scenario == scenario_name]

        if not matching:
            return {"scenario": scenario_name, "error": "No data"}

        total_count = len(matching)
        source = matching[0].source

        # Latency statistics
        ttft_values = [r.first_token_ms for r in matching if r.first_token_ms > 0]
        total_time_values = [r.total_ms for r in matching if r.total_ms > 0]

        ttft_p50 = float(np.percentile(ttft_values, 50)) if ttft_values else 0.0
        ttft_p95 = float(np.percentile(ttft_values, 95)) if ttft_values else 0.0
        ttft_p99 = float(np.percentile(ttft_values, 99)) if ttft_values else 0.0

        total_p50 = float(np.percentile(total_time_values, 50)) if total_time_values else 0.0
        total_p95 = float(np.percentile(total_time_values, 95)) if total_time_values else 0.0
        total_p99 = float(np.percentile(total_time_values, 99)) if total_time_values else 0.0

        # Outcome percentages (Table 2b)
        succeeded_count = sum(1 for r in matching if r.status == "ok")
        rate_limited_count = sum(1 for r in matching if r.status == "rate_limited")
        timed_out_or_dropped = sum(1 for r in matching if r.status in ("timeout", "error", "refused"))

        succeeded_pct = round((succeeded_count / total_count) * 100.0, 1)
        rate_limited_pct = round((rate_limited_count / total_count) * 100.0, 1)
        timed_out_pct = round((timed_out_or_dropped / total_count) * 100.0, 1)

        # Retries
        retries_total = sum(r.retries for r in matching)
        avg_retries = round(retries_total / total_count, 2)

        # Tokens
        tokens_per_req = [r.tokens_in + r.tokens_out for r in matching if (r.tokens_in + r.tokens_out) > 0]
        avg_tokens = int(round(float(np.mean(tokens_per_req)))) if tokens_per_req else 0

        # Peak Queue Length: genuine observed queued requests from task execution queue
        peak_queue = self._scenario_peak_queues.get(scenario_name, 0)

        # Estimated cost per 1000 answers (assuming Groq API standard rate of $0.00 for free tier,
        # or $0.70/1M tokens if modeled commercially)
        cost_per_1000 = round((avg_tokens * 1000 / 1_000_000) * 0.70, 3)

        return {
            "scenario": scenario_name,
            "source": source,
            "requests_sent": total_count,
            "ttft_p50_ms": round(ttft_p50, 1),
            "ttft_p95_ms": round(ttft_p95, 1),
            "ttft_p99_ms": round(ttft_p99, 1),
            "total_time_p50_ms": round(total_p50, 1),
            "total_time_p95_ms": round(total_p95, 1),
            "total_time_p99_ms": round(total_p99, 1),
            "succeeded_pct": succeeded_pct,
            "rate_limited_pct": rate_limited_pct,
            "timed_out_pct": timed_out_pct,
            "retries_per_request": avg_retries,
            "peak_queue_length": peak_queue,
            "tokens_per_answer": avg_tokens,
            "cost_per_1000_answers": cost_per_1000,
        }

