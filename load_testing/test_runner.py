"""
Load Test Runner — Executes workload scenarios and grades results.

Senior Engineering Perspective:
Simulates real traffic generation using a concurrent thread pool with controlled inter-request arrival rates.
For each query:
1. Dynamically reconfigures the stand-in service via its admin API to enforce exact scenario conditions
   (e.g. halved rate limits, exhausted daily budget) in total isolation.
2. Dispatches prompts through PipelineOrchestrator while tracking live queue depth.
3. Measures high-resolution telemetry (TTFT, total time, tokens, retries).
4. Feeds output to AnswerGrader to compute ground-truth grades.
5. Executes the 3-setting retrieval experiment (piece size, pieces fetched, reranking) for Table 3.
6. Stores complete records in MetricsCollector for results.csv export.
"""

import concurrent.futures
import datetime
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from config import settings
from evaluation.answer_grader import AnswerGrader
from evaluation.test_questions import EVALUATION_QUESTIONS
from indexing.paper_parser import PaperParser
from indexing.passage_retriever import PassageRetriever
from indexing.text_chunker import CHUNKING_EXPERIMENTS, TextChunker
from load_testing.metrics_collector import MetricsCollector, RequestLogRecord
from load_testing.results_exporter import ResultsExporter
from load_testing.scenarios import SCENARIO_DEFINITIONS, WorkloadScenario
from models.model_client import ModelClient
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger("LoadTestRunner")


class LoadTestRunner:
    """Orchestrates multi-scenario load testing campaigns."""

    def __init__(
        self,
        passage_retriever: PassageRetriever,
        model_client: ModelClient,
        output_directory: Optional[Path] = None,
    ):
        self.passage_retriever = passage_retriever
        self.model_client = model_client
        self.output_directory = output_directory or Path("results")
        self.collector = MetricsCollector()
        self.exporter = ResultsExporter(self.output_directory)
        self.grader = AnswerGrader()

        # Thread-safe queue depth tracking
        self._active_in_flight = 0
        self._peak_in_flight = 0
        self._peak_queue_len = 0
        self._queue_lock = threading.Lock()

    def _track_request_enter(self) -> int:
        with self._queue_lock:
            self._active_in_flight += 1
            if self._active_in_flight > self._peak_in_flight:
                self._peak_in_flight = self._active_in_flight
            return self._active_in_flight

    def _track_request_exit(self) -> None:
        with self._queue_lock:
            self._active_in_flight = max(0, self._active_in_flight - 1)

    def _configure_mock_service(self, scenario: WorkloadScenario) -> None:
        """Configures stand-in model service to match exact scenario parameters."""
        if scenario.target_backend != "stand-in":
            return

        payload = {
            "rate_limiting_active": True,
            "daily_budget_exhausted": scenario.exhaust_daily_budget,
            "rpm_limit": 15 if scenario.halve_per_minute_limit else 30,
            "tpm_limit": 6000 if scenario.halve_per_minute_limit else 12000,
            "reset_window": True,
        }

        try:
            resp = httpx.post(f"{settings.mock_service_url}/admin/configure", json=payload, timeout=4.0)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Stand-in admin configuration rejected with HTTP {resp.status_code}: {resp.text}"
                )
            confirmed_cfg = resp.json()
            if confirmed_cfg.get("rpm_limit") != payload["rpm_limit"] or confirmed_cfg.get("daily_budget_exhausted") != payload["daily_budget_exhausted"]:
                raise RuntimeError(f"Stand-in configuration mismatch: requested {payload}, got {confirmed_cfg}")

            logger.info(
                f"Verified stand-in service state for '{scenario.scenario_id}': "
                f"rpm={payload['rpm_limit']}, daily_exhausted={payload['daily_budget_exhausted']}"
            )
            time.sleep(0.15)
        except Exception as error:
            logger.error(f"FATAL: Failed to configure stand-in service for '{scenario.scenario_id}': {error}")
            raise RuntimeError(f"Scenario setup failed for '{scenario.scenario_id}': {error}") from error

    def run_scenario(self, scenario: WorkloadScenario) -> Dict[str, Any]:
        """Runs an individual load scenario with isolation and queue tracking."""
        logger.info(
            f"=== Initiating Load Scenario: {scenario.display_title} "
            f"({scenario.total_requests} requests @ {scenario.offered_rate_per_min} RPM) ==="
        )

        # Reset queue metrics and configure mock limiter
        with self._queue_lock:
            self._active_in_flight = 0
            self._peak_in_flight = 0
            self._peak_queue_len = 0

        self._configure_mock_service(scenario)
        self.model_client.forced_backend = scenario.target_backend

        pipeline = PipelineOrchestrator(
            model_client=self.model_client,
            passage_retriever=self.passage_retriever,
        )

        # Distribute question corpus across the request volume
        question_count = len(EVALUATION_QUESTIONS)
        scenario_questions = [
            EVALUATION_QUESTIONS[idx % question_count]
            for idx in range(scenario.total_requests)
        ]

        # Arrival interval
        interval_seconds = 60.0 / max(1.0, scenario.offered_rate_per_min)

        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrency_workers) as executor:
            future_to_meta = {}

            for req_idx, q_item in enumerate(scenario_questions):
                req_id = f"{scenario.scenario_id}_{req_idx + 1:04d}"

                future = executor.submit(
                    self._dispatch_single_request,
                    pipeline=pipeline,
                    question_item=q_item,
                    request_id=req_id,
                    scenario=scenario,
                    retrieval_setting="512_tok_top5_rerank",
                )
                future_to_meta[future] = req_id

                # Sample actual waiting queue depth in the worker pool
                cur_q = executor._work_queue.qsize()
                with self._queue_lock:
                    if cur_q > self._peak_queue_len:
                        self._peak_queue_len = cur_q

                if req_idx < scenario.total_requests - 1:
                    time.sleep(interval_seconds)

            for future in concurrent.futures.as_completed(future_to_meta):
                req_id = future_to_meta[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"[{req_id}] Worker error: {exc}")

        # Record observed peak queue length
        self.collector.record_peak_queue_length(scenario.scenario_id, self._peak_queue_len)

        summary = self.collector.compute_scenario_statistics(scenario.scenario_id)
        logger.info(
            f"=== Scenario {scenario.display_title} Complete: "
            f"p50={summary.get('total_time_p50_ms')}ms, "
            f"Success={summary.get('succeeded_pct')}%, "
            f"Rate-Limited={summary.get('rate_limited_pct')}%, "
            f"Peak Queue={summary.get('peak_queue_length')} ==="
        )
        return summary

    def _dispatch_single_request(
        self,
        pipeline: PipelineOrchestrator,
        question_item: Dict,
        request_id: str,
        scenario: WorkloadScenario,
        retrieval_setting: str = "512_tok_top5_rerank",
        rerank: bool = True,
        top_k: int = 5,
    ) -> None:
        self._track_request_enter()
        start_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        question_text = question_item["question"]

        try:
            execution = pipeline.process_question(
                question=question_text,
                request_id=request_id,
                retrieval_setting=retrieval_setting,
                rerank=rerank,
                top_k=top_k,
            )

            # Grade the answer
            grade_result = self.grader.grade(
                question_data=question_item,
                generated_answer=execution.answer,
            )

            record = RequestLogRecord(
                request_id=request_id,
                scenario=scenario.scenario_id,
                source=scenario.target_backend,
                start_time=start_timestamp,
                first_token_ms=execution.first_token_ms,
                total_ms=execution.total_ms,
                status=execution.status,
                retries=execution.retries,
                tokens_in=execution.total_tokens_in,
                tokens_out=execution.total_tokens_out,
                retrieval_setting=execution.retrieval_setting,
                answer_grade=grade_result.assigned_grade,
            )

        except Exception as error:
            logger.error(f"[{request_id}] Pipeline failure: {error}")
            record = RequestLogRecord(
                request_id=request_id,
                scenario=scenario.scenario_id,
                source=scenario.target_backend,
                start_time=start_timestamp,
                first_token_ms=0.0,
                total_ms=0.0,
                status="error",
                retries=0,
                tokens_in=0,
                tokens_out=0,
                retrieval_setting=retrieval_setting,
                answer_grade="wrong",
            )
        finally:
            self._track_request_exit()

        self.collector.record(record)

    def run_retrieval_experiment(self) -> List[Dict[str, Any]]:
        """
        Executes the mandatory 3-setting retrieval experiment (Table 3):
        - Setting 1: 256 tokens, top-3, no rerank
        - Setting 2: 512 tokens, top-5, with rerank
        - Setting 3: 1024 tokens, top-8, no rerank
        """
        logger.info("\n=== INITIATING 3-SETTING RETRIEVAL EXPERIMENT (TABLE 3) ===")

        # Ensure stand-in service is in unthrottled mode for clean retrieval quality testing
        try:
            httpx.post(
                f"{settings.mock_service_url}/admin/configure",
                json={
                    "rate_limiting_active": False,
                    "daily_budget_exhausted": False,
                    "rpm_limit": 1000,
                    "tpm_limit": 1000000,
                    "reset_window": True,
                },
                timeout=4.0,
            )
            logger.info("Configured stand-in service into unthrottled mode for retrieval accuracy evaluation.")
        except Exception as err:
            logger.warning(f"Could not configure stand-in for retrieval experiment: {err}")

        self.model_client.forced_backend = "stand-in"

        settings_specs = [
            {
                "setting_label": "256_tok_top3_norerank",
                "display_name": "Piece 256, fetched 3, re-rank NO",
                "chunk_size": 256,
                "overlap": 32,
                "top_k": 3,
                "rerank": False,
            },
            {
                "setting_label": "512_tok_top5_rerank",
                "display_name": "Piece 512, fetched 5, re-rank YES",
                "chunk_size": 512,
                "overlap": 64,
                "top_k": 5,
                "rerank": True,
            },
            {
                "setting_label": "1024_tok_top8_norerank",
                "display_name": "Piece 1024, fetched 8, re-rank NO",
                "chunk_size": 1024,
                "overlap": 128,
                "top_k": 8,
                "rerank": False,
            },
        ]

        table3_results = []
        parser = PaperParser(papers_directory=self.passage_retriever.index_directory.parent / "papers")
        sections = parser.parse_all_papers()

        for spec in settings_specs:
            setting_label = spec["setting_label"]
            logger.info(f"\n--- Testing Retrieval Setting: {spec['display_name']} ---")

            # Build fresh chunks for this chunk size
            chunk_cfg = CHUNKING_EXPERIMENTS.get(
                "small" if spec["chunk_size"] == 256 else ("large" if spec["chunk_size"] == 1024 else "medium")
            )
            chunker = TextChunker(config=chunk_cfg)
            chunks = chunker.chunk_sections(sections)

            # Build localized retriever
            temp_retriever = PassageRetriever(index_directory=self.output_directory / f"idx_{setting_label}")
            temp_retriever.build_index(chunks)

            pipeline = PipelineOrchestrator(
                model_client=self.model_client,
                passage_retriever=temp_retriever,
                passages_to_retrieve=spec["top_k"],
            )

            setting_records: List[RequestLogRecord] = []

            for q_idx, q_item in enumerate(EVALUATION_QUESTIONS):
                req_id = f"retrieval_{setting_label}_{q_idx + 1:02d}"
                start_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

                exec_res = pipeline.process_question(
                    question=q_item["question"],
                    request_id=req_id,
                    retrieval_setting=setting_label,
                    rerank=spec["rerank"],
                    top_k=spec["top_k"],
                )

                grade_res = self.grader.grade(question_data=q_item, generated_answer=exec_res.answer)

                rec = RequestLogRecord(
                    request_id=req_id,
                    scenario=f"retrieval_{setting_label}",
                    source="stand-in",
                    start_time=start_ts,
                    first_token_ms=exec_res.first_token_ms,
                    total_ms=exec_res.total_ms,
                    status=exec_res.status,
                    retries=exec_res.retries,
                    tokens_in=exec_res.total_tokens_in,
                    tokens_out=exec_res.total_tokens_out,
                    retrieval_setting=setting_label,
                    answer_grade=grade_res.assigned_grade,
                )
                self.collector.record(rec)
                setting_records.append(rec)

            # Compute Table 3 summary numbers
            total_q = len(setting_records)
            correct_cited = sum(1 for r in setting_records if r.answer_grade == "correct")
            wrong = sum(1 for r in setting_records if r.answer_grade == "wrong")
            correct_refused = sum(1 for r in setting_records if r.answer_grade == "refused_correctly")

            correct_cited_pct = round((correct_cited / total_q) * 100.0, 1)
            wrong_pct = round((wrong / total_q) * 100.0, 1)
            correct_refused_pct = round((correct_refused / total_q) * 100.0, 1)

            tokens_all = [r.tokens_in + r.tokens_out for r in setting_records]
            avg_tokens = int(round(float(np.mean(tokens_all)))) if tokens_all else 0

            ttfts = [r.first_token_ms for r in setting_records if r.first_token_ms > 0]
            totals = [r.total_ms for r in setting_records if r.total_ms > 0]

            ttft_p50 = float(np.percentile(ttfts, 50)) if ttfts else 0.0
            ttft_p95 = float(np.percentile(ttfts, 95)) if ttfts else 0.0
            total_p50 = float(np.percentile(totals, 50)) if totals else 0.0
            total_p95 = float(np.percentile(totals, 95)) if totals else 0.0

            row = {
                "setting": spec["display_name"],
                "setting_label": setting_label,
                "questions_asked": total_q,
                "correct_and_cited_pct": correct_cited_pct,
                "wrong_pct": wrong_pct,
                "correctly_refused_pct": correct_refused_pct,
                "tokens_per_answer": avg_tokens,
                "ttft_p50_ms": round(ttft_p50, 1),
                "ttft_p95_ms": round(ttft_p95, 1),
                "total_p50_ms": round(total_p50, 1),
                "total_p95_ms": round(total_p95, 1),
            }
            table3_results.append(row)
            logger.info(
                f"[Table 3] {spec['display_name']} -> Correct={correct_cited_pct}%, "
                f"Tokens={avg_tokens}, TTFT p50={ttft_p50:.1f}ms, Total p50={total_p50:.1f}ms"
            )

        return table3_results

    def run_all_scenarios(self, scenario_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        keys_to_run = scenario_keys or [
            "steady_stand_in",
            "sudden_burst",
            "sustained_overload",
            "per_minute_limit_halved",
            "daily_budget_exhausted",
        ]
        summaries = []

        for key in keys_to_run:
            if key not in SCENARIO_DEFINITIONS:
                logger.warning(f"Skipping undefined scenario: {key}")
                continue
            scenario_def = SCENARIO_DEFINITIONS[key]
            summary = self.run_scenario(scenario_def)
            summaries.append(summary)

        # Run the mandatory 3-setting retrieval experiment
        table3_data = self.run_retrieval_experiment()

        # Export final results.csv
        self.exporter.write_results_csv(self.collector)

        # Generate markdown report with Tables 2a, 2b, and Table 3
        report_md = self.exporter.generate_markdown_report(self.collector, keys_to_run)

        # Append Table 3 markdown
        table3_lines = [
            "\n### Table 3: Retrieval settings (at least three)\n",
            "| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for t3 in table3_data:
            table3_lines.append(
                f"| {t3['setting']} | {t3['questions_asked']} | {t3['correct_and_cited_pct']}% | "
                f"{t3['wrong_pct']}% | {t3['correctly_refused_pct']}% | {t3['tokens_per_answer']} | "
                f"{t3['ttft_p50_ms']} / {t3['ttft_p95_ms']} | {t3['total_p50_ms']} / {t3['total_p95_ms']} |"
            )

        full_report = report_md + "\n" + "\n".join(table3_lines)
        summary_path = self.output_directory / "load_test_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(full_report)

        manifest = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "scenarios_executed": keys_to_run,
            "scenario_details": {
                k: {
                    "display_title": SCENARIO_DEFINITIONS[k].display_title,
                    "target_backend": SCENARIO_DEFINITIONS[k].target_backend,
                    "offered_rate_per_min": SCENARIO_DEFINITIONS[k].offered_rate_per_min,
                    "concurrency_workers": SCENARIO_DEFINITIONS[k].concurrency_workers,
                    "total_requests": SCENARIO_DEFINITIONS[k].total_requests,
                    "duration_seconds": SCENARIO_DEFINITIONS[k].duration_seconds,
                    "peak_queue_length": self.collector._scenario_peak_queues.get(k, 0),
                }
                for k in keys_to_run if k in SCENARIO_DEFINITIONS
            },
            "retrieval_settings": table3_data,
        }
        with open(self.output_directory / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Comprehensive benchmark report with Tables 2a, 2b, 3 exported to {summary_path}")
        return summaries
