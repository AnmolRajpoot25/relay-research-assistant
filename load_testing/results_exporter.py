"""
Results Exporter — Produces results.csv and formatted assignment tables.

Senior Engineering Perspective:
Per the brief:
"Raw file. Submit one file named results.csv with one row per request and these columns:
 request_id, scenario, source (real or stand-in), start_time, first_token_ms, total_ms,
 status (ok, rate_limited, timeout, error), retries, tokens_in, tokens_out,
 retrieval_setting, answer_grade (correct, wrong, refused_correctly, refused_wrongly).
 Every figure in the tables must be computable from this file."

This module guarantees format fidelity, sorting, and tabular output.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List

from load_testing.metrics_collector import MetricsCollector, RequestLogRecord

logger = logging.getLogger("ResultsExporter")

RESULTS_CSV_HEADERS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]


class ResultsExporter:
    """Exports metrics to CSV and generates matching markdown tables."""

    def __init__(self, output_directory: Path):
        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def write_results_csv(self, collector: MetricsCollector, filename: str = "results.csv") -> Path:
        target_path = self.output_directory / filename
        records = collector.get_all_records()

        with open(target_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=RESULTS_CSV_HEADERS)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_dict())

        logger.info(f"Wrote {len(records)} test records to {target_path.resolve()}")
        return target_path

    def generate_markdown_report(self, collector: MetricsCollector, scenario_names: List[str]) -> str:
        """Constructs formatted GitHub-flavored markdown tables for the writeup."""
        lines = [
            "### Table 2a: Latency, one row per load scenario\n",
            "| Scenario | Real or stand-in | Requests sent | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for s_name in scenario_names:
            stats = collector.compute_scenario_statistics(s_name)
            if "error" in stats:
                continue
            ttft_str = f"{stats['ttft_p50_ms']} / {stats['ttft_p95_ms']} / {stats['ttft_p99_ms']}"
            total_str = f"{stats['total_time_p50_ms']} / {stats['total_time_p95_ms']} / {stats['total_time_p99_ms']}"
            lines.append(
                f"| {stats['scenario']} | {stats['source']} | {stats['requests_sent']} | {ttft_str} | {total_str} |"
            )

        lines.extend([
            "\n### Table 2b: Outcomes, same rows\n",
            "| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers ($) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        for s_name in scenario_names:
            stats = collector.compute_scenario_statistics(s_name)
            if "error" in stats:
                continue
            lines.append(
                f"| {stats['scenario']} | {stats['succeeded_pct']}% | {stats['rate_limited_pct']}% | "
                f"{stats['timed_out_pct']}% | {stats['retries_per_request']} | {stats['peak_queue_length']} | "
                f"{stats['tokens_per_answer']} | ${stats['cost_per_1000_answers']:.3f} |"
            )

        return "\n".join(lines)

