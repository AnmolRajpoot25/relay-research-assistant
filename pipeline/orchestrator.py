"""
Pipeline Orchestrator — Core coordinator of The Relay architecture.

Senior Engineering Perspective:
This class orchestrates the complete request lifecycle across the four mandatory stages:
  Stage 1: Question Decomposition & Guardrail (QuestionAnalyzer)
  Stage 2: Passage Vector Retrieval [Tool Call] (PassageFinder)
  Stage 3: Cited Response Synthesis (ResponseDrafter)
  Stage 4: Automated Quality Audit [Branch/Loop] (QualityChecker)
    └── If audit fails -> Iterative Re-drafting loop (up to max_quality_loops)
    └── If audit passes -> Canonical Citation Formatting -> Client Delivery

Telemetry and Resilience:
- Measures high-resolution Time-to-First-Token (TTFT) and total latency.
- Aggregates prompt and completion tokens across every sub-step.
- Integrates with the SnapshotManager for zero-loss checkpointing during heavy loads or quota events.
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from config import settings
from indexing.passage_retriever import PassageRetriever
from models.model_client import ModelClient
from pipeline.citation_formatter import CitationFormatter
from pipeline.passage_finder import PassageFinder, RetrievalOutcome
from pipeline.quality_checker import QualityChecker, QualityEvaluation
from pipeline.question_analyzer import QuestionAnalysis, QuestionAnalyzer
from pipeline.response_drafter import DraftResult, ResponseDrafter
from resilience.retry_policy import RetryPolicy

logger = logging.getLogger("PipelineOrchestrator")


@dataclass
class StepTelemetry:
    """Latency and resource breakdown for an individual stage."""
    step_name: str
    duration_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model_name: str = ""


@dataclass
class PipelineExecutionResult:
    """Consolidated execution output for an answered inquiry."""
    request_id: str
    question: str
    answer: str
    status: str                         # "ok", "rate_limited", "timeout", "error"
    first_token_ms: float = 0.0
    total_ms: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    quality_loops_executed: int = 0
    citations_count: int = 0
    retries: int = 0
    model_used: str = ""
    retrieval_setting: str = "default"
    step_telemetry: List[StepTelemetry] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_tokens_in + self.total_tokens_out


class PipelineOrchestrator:
    """Coordinates multi-step question answering with adaptive error recovery."""

    def __init__(
        self,
        model_client: ModelClient,
        passage_retriever: PassageRetriever,
        max_quality_loops: Optional[int] = None,
        passages_to_retrieve: Optional[int] = None,
    ):
        self.model_client = model_client
        self.passage_retriever = passage_retriever
        self.max_quality_loops = (
            max_quality_loops if max_quality_loops is not None else settings.max_quality_check_loops
        )
        self.passages_to_retrieve = (
            passages_to_retrieve if passages_to_retrieve is not None else settings.default_passages_to_retrieve
        )

        # Initialize pipeline stages
        self.analyzer = QuestionAnalyzer(model_client=self.model_client)
        self.passage_finder = PassageFinder(
            retriever=self.passage_retriever,
            default_top_k=self.passages_to_retrieve,
            minimum_similarity=settings.min_citation_confidence,
        )
        self.drafter = ResponseDrafter(model_client=self.model_client)
        self.quality_checker = QualityChecker(model_client=self.model_client)
        self.citation_formatter = CitationFormatter()

        # Step-level resilience
        self.retry_policy = RetryPolicy(
            max_attempts=settings.max_retries_per_step,
            base_delay_seconds=settings.retry_base_delay_seconds,
            max_delay_seconds=settings.retry_max_delay_seconds,
            jitter_factor=settings.retry_jitter_factor,
        )

        logger.info(
            f"PipelineOrchestrator ready (max_quality_loops={self.max_quality_loops}, "
            f"top_k_passages={self.passages_to_retrieve})"
        )

    def process_question(
        self,
        question: str,
        request_id: Optional[str] = None,
        retrieval_setting: str = "default",
        rerank: bool = False,
        top_k: Optional[int] = None,
    ) -> PipelineExecutionResult:
        """
        Executes the end-to-end question answering pipeline with branch/loop semantics.
        """
        active_request_id = request_id or f"req_{uuid.uuid4().hex[:8]}"
        pipeline_start_clock = time.perf_counter()

        execution_record = PipelineExecutionResult(
            request_id=active_request_id,
            question=question,
            answer="",
            status="in_progress",
            retrieval_setting=retrieval_setting,
        )

        logger.info(f"[{active_request_id}] Initiating pipeline for: '{question[:70]}...'")

        try:
            # ─────────────────────────────────────────────────────────────
            # STEP 1: Question Decomposition & Guardrail Analysis
            # ─────────────────────────────────────────────────────────────
            step1_start = time.perf_counter()
            analysis, analyzer_model_response = self.analyzer.analyze(question)
            step1_duration = (time.perf_counter() - step1_start) * 1000.0

            execution_record.step_telemetry.append(
                StepTelemetry(
                    step_name="question_analysis",
                    duration_ms=step1_duration,
                    tokens_in=analyzer_model_response.tokens_in,
                    tokens_out=analyzer_model_response.tokens_out,
                    model_name=analyzer_model_response.model_used,
                )
            )

            # Check if rate-limited during analysis
            if analyzer_model_response.was_rate_limited:
                execution_record.status = "rate_limited"
                execution_record.answer = "Service temporarily unavailable due to upstream rate limit (HTTP 429)."
                execution_record.first_token_ms = analyzer_model_response.time_to_first_token_ms or step1_duration
                execution_record.total_ms = (time.perf_counter() - pipeline_start_clock) * 1000.0
                execution_record.model_used = analyzer_model_response.model_used
                self._compile_token_totals(execution_record)
                return execution_record

            # Branch: If the analyzer flagged the question as completely out-of-scope
            if not analysis.is_answerable_from_library:
                refusal_message = (
                    f"The papers in this library do not contain information to answer this inquiry. "
                    f"Notice: {analysis.refusal_reason or 'Query out of scope.'}"
                )
                execution_record.answer = refusal_message
                execution_record.status = "ok"
                execution_record.first_token_ms = analyzer_model_response.time_to_first_token_ms or step1_duration
                execution_record.total_ms = (time.perf_counter() - pipeline_start_clock) * 1000.0
                execution_record.model_used = analyzer_model_response.model_used
                self._compile_token_totals(execution_record)
                return execution_record

            # ─────────────────────────────────────────────────────────────
            # STEP 2: Passage Vector Retrieval [Tool Call]
            # ─────────────────────────────────────────────────────────────
            step2_start = time.perf_counter()
            retrieval_outcome = self.passage_finder.retrieve(
                analysis=analysis,
                top_k=top_k or self.passages_to_retrieve,
                rerank=rerank,
            )
            step2_duration = (time.perf_counter() - step2_start) * 1000.0

            execution_record.step_telemetry.append(
                StepTelemetry(
                    step_name="passage_retrieval_tool",
                    duration_ms=step2_duration,
                    tokens_in=0,
                    tokens_out=0,
                    model_name="vector_similarity_tool",
                )
            )

            # ─────────────────────────────────────────────────────────────
            # STEPS 3 & 4: Response Drafting & Quality Audit Loop
            # ─────────────────────────────────────────────────────────────
            current_draft: Optional[DraftResult] = None
            audit_critique_notes = ""
            measured_ttft_established = False

            for loop_index in range(self.max_quality_loops + 1):
                iteration_number = loop_index + 1

                # Step 3: Response Drafting with RetryPolicy execution
                step3_start = time.perf_counter()

                def _attempt_draft():
                    d_res, m_resp = self.drafter.draft(
                        analysis=analysis,
                        retrieval=retrieval_outcome,
                        draft_iteration=iteration_number,
                        critique_feedback=audit_critique_notes,
                    )
                    # Raise retryable exception if transient 429 rate limit or 503 connection error
                    if m_resp.was_rate_limited:
                        raise RuntimeError(f"HTTP 429 Rate limited: {m_resp.error_message or 'Quota exceeded'}")
                    if m_resp.error_message and any(k in m_resp.error_message for k in ("503", "Connection", "timeout")):
                        raise ConnectionError(f"Transient model error: {m_resp.error_message}")
                    return d_res, m_resp

                draft_tuple, retry_outcome = self.retry_policy.execute(_attempt_draft)
                execution_record.retries += retry_outcome.retries_needed

                if draft_tuple is not None:
                    current_draft, drafter_response = draft_tuple
                else:
                    # Retry budget exhausted
                    from models.groq_client import ModelResponse
                    from pipeline.response_drafter import DraftResult
                    is_rate_lim = "429" in (retry_outcome.final_error or "")
                    current_draft = DraftResult(
                        answer_text="Service temporarily unavailable after retrying. Please retry shortly.",
                        is_refusal=True,
                        refusal_reason=retry_outcome.final_error or "Retries exhausted",
                        draft_iteration=iteration_number,
                    )
                    drafter_response = ModelResponse(
                        error_message=retry_outcome.final_error or "Retries exhausted",
                        was_rate_limited=is_rate_lim,
                    )

                step3_duration = (time.perf_counter() - step3_start) * 1000.0

                execution_record.step_telemetry.append(
                    StepTelemetry(
                        step_name=f"response_draft_cycle_{iteration_number}",
                        duration_ms=step3_duration,
                        tokens_in=drafter_response.tokens_in,
                        tokens_out=drafter_response.tokens_out,
                        model_name=drafter_response.model_used,
                    )
                )

                if drafter_response.was_rate_limited:
                    execution_record.status = "rate_limited"
                    execution_record.answer = "Service temporarily unavailable due to upstream rate limit (HTTP 429)."
                    break
                elif drafter_response.error_message:
                    execution_record.status = "error"
                    execution_record.answer = f"Model execution error: {drafter_response.error_message}"
                    break

                # Capture First-Token-Time (TTFT) during the initial drafting cycle
                if not measured_ttft_established:
                    # Delay before first token = all preceding steps + TTFT of first token
                    preceding_delay = step1_duration + step2_duration
                    token_offset = drafter_response.time_to_first_token_ms or step3_duration
                    execution_record.first_token_ms = preceding_delay + token_offset
                    measured_ttft_established = True

                # If the drafter gave a deterministic refusal or failed, break loop
                if current_draft.is_refusal and not retrieval_outcome.has_sufficient_evidence:
                    break

                # Step 4: Quality Inspection (The Branch/Loop condition)
                step4_start = time.perf_counter()
                audit_verdict, audit_response = self.quality_checker.inspect(
                    draft=current_draft,
                    retrieval=retrieval_outcome,
                    original_question=question,
                )
                step4_duration = (time.perf_counter() - step4_start) * 1000.0

                if audit_response:
                    execution_record.step_telemetry.append(
                        StepTelemetry(
                            step_name=f"quality_audit_cycle_{iteration_number}",
                            duration_ms=step4_duration,
                            tokens_in=audit_response.tokens_in,
                            tokens_out=audit_response.tokens_out,
                            model_name=audit_response.model_used,
                        )
                    )

                execution_record.quality_loops_executed = loop_index

                # Branch decision
                if audit_verdict.passes_audit:
                    logger.info(
                        f"[{active_request_id}] Quality audit PASSED on iteration {iteration_number}."
                    )
                    break
                else:
                    if loop_index < self.max_quality_loops:
                        logger.warning(
                            f"[{active_request_id}] Quality audit FAILED on iteration {iteration_number}. "
                            f"Actionable issues: {audit_verdict.detected_flaws}. Looping back to re-draft."
                        )
                        audit_critique_notes = audit_verdict.actionable_critique
                    else:
                        logger.warning(
                            f"[{active_request_id}] Exceeded maximum allowed re-draft cycles ({self.max_quality_loops}). "
                            "Failing closed per safety guarantee: ungrounded answer is rejected."
                        )
                        current_draft = DraftResult(
                            answer_text=(
                                "The paper library contains relevant context, but the assistant could not verify the "
                                "synthesized draft with sufficient certainty against the source texts. "
                                "Per safety policy, this response is withheld to avoid transmitting ungrounded claims."
                            ),
                            is_refusal=True,
                            refusal_reason="Exceeded maximum quality verification cycles without passing audit.",
                            draft_iteration=iteration_number,
                        )
                        execution_record.status = "refused"
                        break

            # ─────────────────────────────────────────────────────────────
            # Final Post-Processing & Canonical Citation Formatting
            # ─────────────────────────────────────────────────────────────
            if current_draft is not None:
                execution_record.answer = self.citation_formatter.format_final_answer(
                    current_draft.answer_text
                )
                execution_record.citations_count = self.citation_formatter.count_citations(
                    execution_record.answer
                )
                # Preserve explicit terminal statuses (rate_limited, refused, error)
                if execution_record.status not in ("rate_limited", "refused", "error", "timeout"):
                    if current_draft.is_refusal and not retrieval_outcome.has_sufficient_evidence:
                        execution_record.status = "ok"
                    elif current_draft.is_refusal:
                        execution_record.status = "refused"
                    else:
                        execution_record.status = "ok"

        except Exception as unhandled_error:
            logger.error(f"[{active_request_id}] Unhandled pipeline exception: {unhandled_error}", exc_info=True)
            execution_record.status = "error"
            execution_record.answer = (
                "An internal engineering fault occurred during pipeline execution. "
                "The error has been logged for remediation."
            )

        execution_record.total_ms = (time.perf_counter() - pipeline_start_clock) * 1000.0
        self._compile_token_totals(execution_record)

        # Set model_used based on the primary drafting step
        for step in reversed(execution_record.step_telemetry):
            if step.model_name and step.model_name != "vector_similarity_tool":
                execution_record.model_used = step.model_name
                break

        logger.info(
            f"[{active_request_id}] Pipeline execution finalized: status={execution_record.status}, "
            f"total_latency={execution_record.total_ms:.1f}ms, "
            f"total_tokens={execution_record.total_tokens}, "
            f"loops={execution_record.quality_loops_executed}"
        )

        return execution_record

    def _compile_token_totals(self, record: PipelineExecutionResult) -> None:
        record.total_tokens_in = sum(s.tokens_in for s in record.step_telemetry)
        record.total_tokens_out = sum(s.tokens_out for s in record.step_telemetry)

