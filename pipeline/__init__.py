"""
Pipeline Module — The core multi-step cited Q&A engine for The Relay.
"""

from pipeline.orchestrator import PipelineOrchestrator, PipelineExecutionResult
from pipeline.question_analyzer import QuestionAnalyzer, QuestionAnalysis
from pipeline.passage_finder import PassageFinder, RetrievalOutcome, RetrievedPassage
from pipeline.response_drafter import ResponseDrafter, DraftResult
from pipeline.quality_checker import QualityChecker, QualityEvaluation
from pipeline.citation_formatter import CitationFormatter

__all__ = [
    "PipelineOrchestrator",
    "PipelineExecutionResult",
    "QuestionAnalyzer",
    "QuestionAnalysis",
    "PassageFinder",
    "RetrievalOutcome",
    "RetrievedPassage",
    "ResponseDrafter",
    "DraftResult",
    "QualityChecker",
    "QualityEvaluation",
    "CitationFormatter",
]

