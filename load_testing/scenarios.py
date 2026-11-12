"""
Load Testing Scenarios — Definitions for workload characterization.

Senior Engineering Perspective:
The assignment brief requires evaluating the pipeline under at least three load shapes:
steady, sudden burst, and sustained overload, plus two edge scenarios:
- per-minute limit halved (simulating provider capacity downgrades)
- daily budget exhausted (3pm cutoff behavior)

Each scenario configuration specifies:
1. Target offered rate (requests per minute)
2. Total request volume and duration
3. Concurrency pattern (poisson arrivals, spike bursts, or saturated queue)
4. Model backend: real Groq versus calibrated stand-in
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WorkloadScenario:
    """Defines a discrete load-testing workload profile."""
    scenario_id: str
    display_title: str
    target_backend: str             # "real" or "stand-in"
    total_requests: int
    duration_seconds: float
    offered_rate_per_min: float
    concurrency_workers: int
    burst_batch_size: Optional[int] = None
    halve_per_minute_limit: bool = False
    exhaust_daily_budget: bool = False
    notes: str = ""


SCENARIO_DEFINITIONS: Dict[str, WorkloadScenario] = {
    # ── 1. Steady Baseline (Real Groq Probing) ──────────────────────────────
    "steady_real": WorkloadScenario(
        scenario_id="steady_real",
        display_title="Steady (Real Groq)",
        target_backend="real",
        total_requests=15,
        duration_seconds=90.0,
        offered_rate_per_min=10.0,     # Safely below Groq's 30 RPM limit
        concurrency_workers=2,
        notes="Gentle steady baseline against hosted Groq to measure genuine p50/p95 latency.",
    ),

    # ── 2. Steady (Stand-in Calibrated) ──────────────────────────────────────
    "steady_stand_in": WorkloadScenario(
        scenario_id="steady_stand_in",
        display_title="Steady (Stand-In)",
        target_backend="stand-in",
        total_requests=30,
        duration_seconds=75.0,
        offered_rate_per_min=24.0,     # Safely within nominal 30 RPM limit
        concurrency_workers=3,
        notes="Validation run confirming stand-in emulates real service baseline.",
    ),

    # ── 3. Sudden Burst (Incident Spike) ─────────────────────────────────────
    "sudden_burst": WorkloadScenario(
        scenario_id="sudden_burst",
        display_title="Sudden Burst (Stand-In)",
        target_backend="stand-in",
        total_requests=40,
        duration_seconds=15.0,
        offered_rate_per_min=160.0,    # 5.3x over nominal RPM limit
        concurrency_workers=15,
        burst_batch_size=15,
        notes="Sudden wave of requests arriving immediately following user incident.",
    ),

    # ── 4. Sustained Overload (Peak Load) ────────────────────────────────────
    "sustained_overload": WorkloadScenario(
        scenario_id="sustained_overload",
        display_title="Sustained Overload (Stand-In)",
        target_backend="stand-in",
        total_requests=50,
        duration_seconds=50.0,
        offered_rate_per_min=60.0,     # 2x sustained over-capacity
        concurrency_workers=10,
        notes="Prolonged heavy traffic testing queue depth, circuit breaking, and drop rates.",
    ),

    # ── 5. Per-Minute Limit Halved (Upstream Throttling) ─────────────────────
    "per_minute_limit_halved": WorkloadScenario(
        scenario_id="per_minute_limit_halved",
        display_title="Per-Minute Limit Halved (Stand-In)",
        target_backend="stand-in",
        total_requests=30,
        duration_seconds=45.0,
        offered_rate_per_min=40.0,
        concurrency_workers=5,
        halve_per_minute_limit=True,
        notes="Simulates provider reducing RPM from 30 to 15. Tests adaptive token budgeting.",
    ),

    # ── 6. Daily Budget Exhausted (3pm Cutoff Scenario) ──────────────────────
    "daily_budget_exhausted": WorkloadScenario(
        scenario_id="daily_budget_exhausted",
        display_title="Daily Budget Exhausted (Stand-In)",
        target_backend="stand-in",
        total_requests=20,
        duration_seconds=15.0,
        offered_rate_per_min=80.0,
        concurrency_workers=4,
        exhaust_daily_budget=True,
        notes="What the person sees at 3pm when daily budget is completely gone.",
    ),
}

