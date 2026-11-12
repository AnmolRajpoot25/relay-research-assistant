"""
Central configuration for The Relay pipeline.

This module defines every tunable parameter in one place. Values come from
environment variables (via .env) with sensible defaults. A senior engineer
should be able to read this file and understand the system's boundaries.

Design decision: We keep all magic numbers here — not scattered across modules —
so that load-test scenarios can override them cleanly.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

# ── Project paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
PAPERS_DIRECTORY = PROJECT_ROOT / "papers"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
INDEX_DIRECTORY = PROJECT_ROOT / "index_store"


class GroqRateLimits:
    """
    Known rate limits for Groq's free tier, per model.
    These are org-level limits (shared across all API keys in the same account).
    Source: Groq console documentation + empirical measurement.
    """

    MODELS = {
        "llama-3.3-70b-versatile": {
            "requests_per_minute": 30,
            "tokens_per_minute": 12_000,
            "requests_per_day": 1_000,
            "tokens_per_day": 100_000,
        },
        "llama-3.1-8b-instant": {
            "requests_per_minute": 30,
            "tokens_per_minute": 6_000,
            "requests_per_day": 14_400,
            "tokens_per_day": 500_000,
        },
    }

    @classmethod
    def get_limits(cls, model_name: str) -> dict:
        """Return limits for a model, or conservative defaults if unknown."""
        return cls.MODELS.get(model_name, {
            "requests_per_minute": 20,
            "tokens_per_minute": 5_000,
            "requests_per_day": 500,
            "tokens_per_day": 50_000,
        })


class PipelineSettings(BaseSettings):
    """Settings for the multi-step answer pipeline."""

    # ── Model configuration ──────────────────────────────────────────────
    groq_api_key: str = Field(
        default="",
        description="Groq API key",
    )
    primary_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Primary model for best quality answers",
    )
    fallback_model: str = Field(
        default="llama-3.1-8b-instant",
        description="Fallback model when primary quota runs low",
    )
    mock_service_url: str = Field(
        default="http://localhost:8001",
        description="URL for the stand-in mock model service",
    )

    # ── Timeouts (in seconds) ────────────────────────────────────────────
    model_call_timeout: float = Field(
        default=30.0,
        description="Max seconds to wait for a single model call",
    )
    total_request_timeout: float = Field(
        default=120.0,
        description="Max seconds for the entire pipeline per question",
    )
    user_patience_timeout: float = Field(
        default=45.0,
        description="After this many seconds, users typically give up",
    )

    # ── Retry policy ─────────────────────────────────────────────────────
    max_retries_per_step: int = Field(
        default=3,
        description="Maximum retries for a single pipeline step",
    )
    retry_base_delay_seconds: float = Field(
        default=1.0,
        description="Base delay for exponential backoff",
    )
    retry_max_delay_seconds: float = Field(
        default=30.0,
        description="Cap on retry delay to prevent infinite waits",
    )
    retry_jitter_factor: float = Field(
        default=0.5,
        description="Random jitter factor (0-1) added to retry delays",
    )

    # ── Retrieval settings ───────────────────────────────────────────────
    default_chunk_size: int = Field(
        default=512,
        description="Default number of tokens per text chunk",
    )
    default_chunk_overlap: int = Field(
        default=64,
        description="Overlap tokens between consecutive chunks",
    )
    default_passages_to_retrieve: int = Field(
        default=5,
        description="Number of passages to retrieve per question",
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )

    # ── Quality gates ────────────────────────────────────────────────────
    max_quality_check_loops: int = Field(
        default=2,
        description="Max times to re-draft if quality check fails",
    )
    min_citation_confidence: float = Field(
        default=0.6,
        description="Minimum cosine similarity for a passage to be relevant",
    )

    # ── Rate limiting safety margins ─────────────────────────────────────
    quota_warning_threshold: float = Field(
        default=0.8,
        description="Switch to fallback when this fraction of quota is used",
    )
    quota_critical_threshold: float = Field(
        default=0.95,
        description="Switch to mock when this fraction is used",
    )

    # ── Logging ──────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# ── Singleton instance ───────────────────────────────────────────────────────
# Import this throughout the project: `from config import settings`
settings = PipelineSettings()
