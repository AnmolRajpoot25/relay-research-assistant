"""
Run Load Tests — Execution harness for load shapes and results.csv generation.

Senior Engineering Perspective:
Orchestrates the entire load campaign across the mandated scenarios:
  1. steady_real (or steady_stand_in)
  2. sudden_burst
  3. sustained_overload
  4. per_minute_limit_halved
  5. daily_budget_exhausted

Ensures the mock server is running for stand-in loads, runs the scenarios,
grades answers against ground truth, and writes `results/results.csv`.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import httpx

from config import INDEX_DIRECTORY, RESULTS_DIRECTORY, settings
from indexing.passage_retriever import PassageRetriever
from load_testing.scenarios import SCENARIO_DEFINITIONS
from load_testing.test_runner import LoadTestRunner
from models.model_client import ModelClient


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def verify_mock_service_available(mock_url: str = "http://127.0.0.1:8001") -> bool:
    try:
        response = httpx.get(f"{mock_url}/health", timeout=1.5)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    return False


def main():
    parser = argparse.ArgumentParser(description="Run The Relay Load Testing Campaign")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help=f"Specific scenarios to run. Choices: {list(SCENARIO_DEFINITIONS.keys())}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIRECTORY),
        help="Directory to save results.csv",
    )
    args = parser.parse_args()
    configure_logging()

    scenarios_to_run = args.scenarios or [
        "steady_stand_in",
        "sudden_burst",
        "sustained_overload",
        "per_minute_limit_halved",
        "daily_budget_exhausted",
    ]

    # Verify if any scenario needs mock service
    needs_mock = any(
        SCENARIO_DEFINITIONS[s].target_backend == "stand-in"
        for s in scenarios_to_run
        if s in SCENARIO_DEFINITIONS
    )

    if needs_mock and not verify_mock_service_available(settings.mock_service_url):
        print("\n[!] Error: Stand-in mock service is not detected at http://127.0.0.1:8001.")
        print("    Please start it in another terminal before executing load tests:")
        print("    python -m mock_service.mock_server --port 8001\n")
        sys.exit(1)

    retriever = PassageRetriever(index_directory=INDEX_DIRECTORY)
    if not retriever.load_index():
        print("\n[!] Search index not found. Building index first...")
        from run_pipeline import build_index
        retriever = build_index("medium")

    model_client = ModelClient()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = LoadTestRunner(
        passage_retriever=retriever,
        model_client=model_client,
        output_directory=output_dir,
    )

    print("\n" + "=" * 70)
    print(f"STARTING LOAD TESTING CAMPAIGN: {len(scenarios_to_run)} SCENARIOS")
    print("=" * 70)

    runner.run_all_scenarios(scenarios_to_run)

    print("\n[OK] Load testing campaign completed successfully!")
    print(f"    Raw results written to: {output_dir / 'results.csv'}")
    print(f"    Markdown summary table: {output_dir / 'load_test_summary.md'}\n")


if __name__ == "__main__":
    main()
