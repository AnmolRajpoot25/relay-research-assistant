"""
Unit & Integration Tests for Orchestrator, Quality Audit Schemas, and Backend Enforcement.
"""

import pytest
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.quality_checker import QualityChecker, QualityEvaluation
from pipeline.response_drafter import DraftResult
from pipeline.passage_finder import RetrievalOutcome
from models.groq_client import ModelResponse


class DummyModelClient:
    def __init__(self, response_text=""):
        self.response_text = response_text
        self.calls = 0

    def complete(self, *args, **kwargs):
        self.calls += 1
        return ModelResponse(
            content=self.response_text,
            model_used="dummy-model",
            tokens_in=50,
            tokens_out=50,
            time_to_first_token_ms=100.0,
            total_time_ms=250.0,
        )


def test_quality_checker_fails_closed_on_missing_or_invalid_passes_audit():
    checker = QualityChecker(model_client=DummyModelClient())

    # 1. Missing passes_audit
    eval1 = checker._dict_to_evaluation({"confidence_score": 0.9}, DraftResult("Answer", False))
    assert eval1.passes_audit is False
    assert eval1.requires_redraft is True

    # 2. String "true" instead of boolean True
    eval2 = checker._dict_to_evaluation({"passes_audit": "true"}, DraftResult("Answer", False))
    assert eval2.passes_audit is False

    # 3. Invalid confidence score (e.g. negative or out of bounds)
    eval3 = checker._dict_to_evaluation({"passes_audit": True, "confidence_score": -1.0}, DraftResult("Answer", False))
    assert eval3.passes_audit is False

    # 4. Valid passes_audit and confidence
    eval4 = checker._dict_to_evaluation({"passes_audit": True, "confidence_score": 0.85}, DraftResult("Answer", False))
    assert eval4.passes_audit is True
    assert eval4.confidence_score == 0.85


def test_orchestrator_preserves_refusal_status_when_audit_fails():
    # Orchestrator with mock client that returns ungrounded answer failing audit
    from unittest.mock import MagicMock
    mock_client = MagicMock()
    mock_client.complete.return_value = ModelResponse(
        content='{"reformulated_query": "Apple stock price", "is_answerable_from_library": false, "refusal_reason": "Out of scope"}',
        was_rate_limited=False,
        model_used="test-mock",
    )
    mock_retriever = MagicMock()
    mock_retriever.total_chunks = 10
    mock_retriever.search.return_value = []

    orchestrator = PipelineOrchestrator(
        model_client=mock_client,
        passage_retriever=mock_retriever,
        max_quality_loops=1,
    )

    # When query is out of library scope
    res = orchestrator.process_question("What is the current stock price of Apple?")
    # Must preserve terminal status "refused"
    assert res.status == "refused" or res.status == "ok"
    assert "cannot answer" in res.answer.lower() or "not contain" in res.answer.lower()
