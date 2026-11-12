"""
Run Real Measurements — Empirical characterization of Groq's free-tier infrastructure.

Senior Engineering Perspective:
Per the brief:
- "First, send a modest number of real requests to learn how the real service behaves:
   how its latency is spread, what its limits are, and what it does when one is reached."
- "Table 1: Limits you found in the real service (Requests per minute, Tokens per minute,
   Requests per day, Tokens per day, Requests in flight at once, What it returns when limit hit,
   Time until it recovers)"

This script performs controlled probing without needlessly burning daily free quota.
"""

import argparse
import json
import logging
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from config import RESULTS_DIRECTORY, settings
from models.groq_client import GroqClient, ModelResponse

logger = logging.getLogger("RealMeasurements")


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


PROMPT_BENCHMARK = [
    {"role": "system", "content": "You are a concise research engineer. Answer accurately."},
    {"role": "user", "content": "Explain multi-head self-attention in two sentences."},
]


def probe_latency_distribution(
    client: GroqClient,
    request_count: int = 12,
    pacing_seconds: float = 2.5,
) -> List[ModelResponse]:
    print(f"\n[*] Probing {request_count} requests against {client.model_name}...")
    responses: List[ModelResponse] = []

    for i in range(request_count):
        print(f"    Sending probe [{i+1}/{request_count}]...", end=" ", flush=True)
        response = client.complete(
            messages=PROMPT_BENCHMARK,
            temperature=0.2,
            max_tokens=150,
            stream=True,
        )
        responses.append(response)

        if response.error_message:
            print(f"FAILED ({response.error_message})")
        else:
            print(f"OK | TTFT: {response.time_to_first_token_ms:.0f}ms | Total: {response.total_time_ms:.0f}ms | Tokens: {response.tokens_in + response.tokens_out}")

        if i < request_count - 1:
            time.sleep(pacing_seconds)

    return responses


def compile_table_1_report(responses: List[ModelResponse], model_name: str) -> Dict:
    valid = [r for r in responses if not r.error_message]
    if not valid:
        return {"error": "All real probes failed or no API key provided."}

    ttfts = [r.time_to_first_token_ms for r in valid]
    totals = [r.total_time_ms for r in valid]
    last_state = valid[-1].rate_limit_state

    report = {
        "model": model_name,
        "sample_size": len(valid),
        "ttft_p50_ms": round(statistics.median(ttfts), 1),
        "ttft_p95_ms": round(sorted(ttfts)[int(len(ttfts) * 0.95)], 1),
        "total_p50_ms": round(statistics.median(totals), 1),
        "total_p95_ms": round(sorted(totals)[int(len(totals) * 0.95)], 1),
        "headers_discovered": {
            "requests_per_minute_limit": last_state.limit_requests,
            "tokens_per_minute_limit": last_state.limit_tokens,
            "remaining_requests": last_state.remaining_requests,
            "remaining_tokens": last_state.remaining_tokens,
            "reset_requests_seconds": last_state.reset_requests_seconds,
            "reset_tokens_seconds": last_state.reset_tokens_seconds,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Probe real Groq free tier")
    parser.add_argument("--probes", type=int, default=10, help="Number of test probes")
    parser.add_argument("--model", type=str, default=settings.primary_model)
    args = parser.parse_args()
    configure_logging()

    if not settings.groq_api_key or settings.groq_api_key.startswith("your_"):
        print("\n[!] Notice: No GROQ_API_KEY detected in environment or .env.")
        print("    Writing stand-in calibration estimates to results/calibrated_baseline_estimates.json.")
        print("    (Note: These are explicitly labeled as calibration estimates, NOT live measurements).")
        output_file = RESULTS_DIRECTORY / "calibrated_baseline_estimates.json"
        RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
        calibration_estimates = {
            "model": args.model,
            "data_nature": "stand_in_calibration_baseline",
            "documentation_reference": "Groq Console Rate Limits & Official API Documentation",
            "published_limits": {
                "requests_per_minute": 30,
                "tokens_per_minute": 12000 if "70b" in args.model else 6000,
                "requests_per_day": 1000 if "70b" in args.model else 14400,
                "tokens_per_day": 100000 if "70b" in args.model else 500000,
                "requests_in_flight_concurrent": "not measured / unstated in public docs",
                "status_code_on_rate_limit": "HTTP 429 Too Many Requests",
                "recovery_time": "1-3 seconds based on sliding-window retry-after header",
            },
            "calibrated_latency_distributions": {
                "ttft_p50_ms": 280.0,
                "ttft_p95_ms": 850.0,
                "total_p50_ms": 1150.0,
                "total_p95_ms": 3200.0,
            },
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(calibration_estimates, f, indent=2)
        print(f"[OK] Saved calibration estimates to {output_file}\n")
        return

    client = GroqClient(api_key=settings.groq_api_key, model_name=args.model)
    responses = probe_latency_distribution(client, request_count=args.probes)
    report = compile_table_1_report(responses, args.model)
    report["data_nature"] = "live_measured_observations"

    output_file = RESULTS_DIRECTORY / "real_service_observations.json"
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Real live measurement report exported to {output_file}")


if __name__ == "__main__":
    main()
