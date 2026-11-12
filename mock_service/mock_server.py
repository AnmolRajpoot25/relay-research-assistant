"""
Mock Model Server — Stand-In Inference Service.

Senior Engineering Perspective:
To rigorously test the pipeline under burst, sustained overload, and rate-limit halving
without burning free-tier account quotas or violating Groq's Terms of Service, this
FastAPI microservice emulates Groq's API contract.

Key architectural features:
1. OpenAI-Compatible API: Implements `/v1/chat/completions` supporting both SSE streaming and sync modes.
2. Token-by-Token Streaming: Generates Server-Sent Events with realistic inter-token pauses.
3. Realistic Latency Curve: Uses log-normal TTFT distribution calibrated from real measurements.
4. Concurrency Degradation: Slows down realistically as concurrent load increases.
5. Ingress Rate Limiting: Accurately replicates HTTP 429 status and `x-ratelimit-*` headers.
"""

import asyncio
import json
import logging
import random
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mock_service.calibration_data import (
    CALIBRATION_PROFILES,
    DEFAULT_RETRY_AFTER_SECONDS,
    RATE_LIMIT_OVERHEAD_MS,
)
from mock_service.latency_simulator import LatencySimulator
from mock_service.load_degradation import LoadDegradationEngine

logger = logging.getLogger("MockModelServer")


# ── Canonical Mock Knowledge Base ────────────────────────────────────────────

MOCK_DOMAIN_RESPONSES = [
    (
        "The Transformer architecture replaces recurrent and convolutional layers entirely "
        "with multi-head self-attention mechanisms, allowing significant parallelization during "
        "training and achieving superior translation quality. [Attention Is All You Need, §3.1, p.3]"
    ),
    (
        "Multi-Head Attention projects queries, keys, and values h=8 times with learned linear "
        "projections, concatenating attention outputs to capture distinct subspace representations. "
        "[Attention Is All You Need, §3.2, p.4]"
    ),
    (
        "Moshi incorporates the Mimi neural audio codec, processing streaming speech tokens with "
        "an end-to-end multi-stream transformer that sustains 160ms theoretical latency for full-duplex "
        "dialogue. [Moshi: a speech-text foundation model, §2.2, p.4]"
    ),
    (
        "Dynamic Chunking introduces differentiable learned sequence boundaries to hierarchically "
        "compress token sequences into variable-length multi-token chunks without loss of context. "
        "[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, §3, p.4]"
    ),
    (
        "The papers in this library do not contain information regarding this inquiry. "
        "I must refrain from guessing per verification policy."
    ),
]


def generate_mock_model_response(messages: List["ChatMessagePayload"]) -> str:
    """
    Intelligently responds to pipeline components:
    - QuestionAnalyzer: returns valid JSON with sub_queries and answerability.
    - QualityChecker: returns valid JSON verification verdict.
    - ResponseDrafter: returns grounded answers with valid paper & section/page citations,
      or truthful refusals for questions outside library scope.
    """
    combined = " ".join(m.content for m in messages)
    combined_lower = combined.lower()

    # 1. QualityChecker component
    if "passes_audit" in combined_lower or "qualitychecker" in combined_lower or "actionable_critique" in combined_lower:
        return json.dumps({
            "passes_audit": True,
            "confidence_score": 0.95,
            "detected_flaws": [],
            "citations_verified": 2,
            "citations_flagged": 0,
            "actionable_critique": "",
        })

    # 2. QuestionAnalyzer component
    if "sub_queries" in combined_lower or "questionanalyzer" in combined_lower or "decomposition" in combined_lower:
        is_unanswerable = any(
            k in combined_lower
            for k in ["nvidia", "gpt-4", "quantum gravity", "ashish vaswani born", "birthplace", "kyutai's total budget", "euro currency"]
        )
        if is_unanswerable:
            return json.dumps({
                "is_answerable_from_library": False,
                "sub_queries": [],
                "refusal_reason": "Inquiry is completely out of scope of the three library research papers.",
            })
        return json.dumps({
            "is_answerable_from_library": True,
            "sub_queries": ["key technical concept"],
            "refusal_reason": "",
        })

    # 3. ResponseDrafter component (or general inference)
    # Check for unanswerable / out-of-scope inquiries first
    if any(k in combined_lower for k in ["nvidia", "gpt-4", "quantum gravity", "ashish vaswani", "birthplace", "kyutai's total budget", "euro currency"]):
        return (
            "The papers in this library do not contain information regarding this inquiry. "
            "Per system policy, I must refrain from guessing."
        )

    # Attention Is All You Need domain
    if "bleu" in combined_lower or "wmt 2014" in combined_lower:
        return (
            "The big Transformer model achieved a 41.8 BLEU score on the WMT 2014 English-to-French translation task, "
            "establishing a new state of the art outperforming previous models. [Attention Is All You Need, §6.1, p.8]"
        )
    if "parallel attention heads" in combined_lower or "d_k" in combined_lower or "heads are employed" in combined_lower:
        return (
            "The base Transformer employs h=8 parallel attention heads with key and value dimension d_k = d_v = 64, "
            "allowing the model to jointly attend to information at different positions. [Attention Is All You Need, §3.2.2, p.4]"
        )
    if "optimizer" in combined_lower or "warmup" in combined_lower or "schedule" in combined_lower or "adam" in combined_lower:
        return (
            "The Transformer models were trained using the Adam optimizer with beta_1=0.9, beta_2=0.98, and epsilon=10^-9, "
            "with a learning rate warmup schedule that increases linearly for the first 4,000 steps. [Attention Is All You Need, §5.3, p.7]"
        )
    if "inject sequence order" in combined_lower or "positional encoding" in combined_lower or "sine" in combined_lower:
        return (
            "The Transformer injects sequence order information without recurrence or convolution using fixed sinusoidal positional encodings "
            "based on sine and cosine functions of different frequencies. [Attention Is All You Need, §3.5, p.6]"
        )
    if "mechanism replacing recurrence" in combined_lower or "primary architectural mechanism" in combined_lower:
        return (
            "The Transformer architecture replaces recurrent and convolutional layers entirely with multi-head self-attention mechanisms, "
            "allowing significant parallelization during training. [Attention Is All You Need, §3.1, p.3]"
        )

    # Moshi domain
    if "mimi" in combined_lower or "codec" in combined_lower or "acoustic streams" in combined_lower or "frame rate" in combined_lower:
        return (
            "Moshi employs the Mimi neural audio codec to discretize acoustic streams into tokens at a 12.5 Hz frame rate with 24 kHz audio. "
            "[Moshi: a speech-text foundation model, §2.1, p.3]"
        )
    if "full-duplex" in combined_lower or "two participants" in combined_lower or "duplex" in combined_lower:
        return (
            "Moshi achieves full-duplex conversational interaction using dual parallel streams for user and agent audio along with an inner monologue text stream. "
            "[Moshi: a speech-text foundation model, §2.2, p.4]"
        )
    if "theoretical latency" in combined_lower or ("latency" in combined_lower and "moshi" in combined_lower):
        return (
            "Moshi achieves a theoretical latency of 160ms for streaming dialogue, supporting natural, real-time conversational turn-taking. "
            "[Moshi: a speech-text foundation model, §1, p.2]"
        )
    if "modalities" in combined_lower or "multi-stream" in combined_lower:
        return (
            "Moshi's multi-stream transformer backbone jointly models speech, text, and acoustic tokens across concurrent input and output channels. "
            "[Moshi: a speech-text foundation model, §2.2, p.4]"
        )

    # Dynamic Chunking domain
    if "fundamental problem" in combined_lower or "hierarchical sequence" in combined_lower:
        return (
            "Dynamic Chunking addresses variable-length chunks and learned hierarchical boundaries to eliminate fixed-window boundary artifacts in sequence modeling. "
            "[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, §1, p.2]"
        )
    if "segment boundaries" in combined_lower or "differentiable" in combined_lower:
        return (
            "Dynamic Chunking establishes segment boundaries in an end-to-end differentiable manner using learned boundary scoring functions trained jointly with the model. "
            "[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, §3, p.4]"
        )
    if "computational complexity" in combined_lower or "compression" in combined_lower:
        return (
            "Dynamic Chunking reduces computational complexity through hierarchical token compression across multi-token segments. "
            "[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, §3.2, p.5]"
        )

    # Cross-Paper synthesis
    if "compare" in combined_lower or "cross" in combined_lower:
        return (
            "The original Transformer applies multi-head self-attention across text tokens in machine translation [Attention Is All You Need, §3.1, p.3], "
            "whereas Moshi extends self-attention to multi-stream cross-modal attention spanning concurrent speech and text streams [Moshi: a speech-text foundation model, §2.2, p.4]."
        )

    # Generic grounded fallback matching whichever paper is mentioned
    if "moshi" in combined_lower:
        return (
            "Moshi processes streaming speech and text tokens with 160ms latency for full-duplex spoken dialogue. "
            "[Moshi: a speech-text foundation model, §2.2, p.4]"
        )
    if "chunking" in combined_lower:
        return (
            "Dynamic Chunking hierarchically compresses token sequences into variable-length chunks. "
            "[Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, §3, p.4]"
        )
    return (
        "The Transformer architecture replaces recurrent and convolutional layers entirely with multi-head self-attention mechanisms. "
        "[Attention Is All You Need, §3.1, p.3]"
    )


# ── Sliding Window Rate Limiter ──────────────────────────────────────────────

class MockIngressLimiter:
    """Emulates Groq's gateway token and request counter windows."""

    def __init__(self, requests_per_minute: int = 30, tokens_per_minute: int = 12000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.daily_budget_exhausted = False
        self.rate_limiting_active = True

        self._requests_counter = 0
        self._tokens_counter = 0
        self._window_start = time.time()

    def reset_counters(self) -> None:
        """Reset usage window counters for isolated test scenario execution."""
        self._requests_counter = 0
        self._tokens_counter = 0
        self._window_start = time.time()

    def check_capacity(self, token_estimate: int = 400) -> Tuple[bool, Dict[str, str]]:
        now = time.time()
        elapsed = now - self._window_start

        # Window roll-over every 60 seconds
        if elapsed >= 60.0:
            self._requests_counter = 0
            self._tokens_counter = 0
            self._window_start = now
            elapsed = 0.0

        time_until_reset = max(0.5, 60.0 - elapsed)

        headers = {
            "x-ratelimit-limit-requests": str(self.rpm_limit),
            "x-ratelimit-remaining-requests": str(max(0, self.rpm_limit - self._requests_counter)),
            "x-ratelimit-limit-tokens": str(self.tpm_limit),
            "x-ratelimit-remaining-tokens": str(max(0, self.tpm_limit - self._tokens_counter)),
            "x-ratelimit-reset-requests": f"{time_until_reset:.1f}s",
            "x-ratelimit-reset-tokens": f"{time_until_reset:.1f}s",
        }

        if self.daily_budget_exhausted:
            headers["x-ratelimit-remaining-requests"] = "0"
            headers["x-ratelimit-remaining-tokens"] = "0"
            return False, headers

        if not self.rate_limiting_active:
            return True, headers

        if self._requests_counter >= self.rpm_limit:
            return False, headers

        if self._tokens_counter + token_estimate > self.tpm_limit:
            return False, headers

        # Consume quota
        self._requests_counter += 1
        self._tokens_counter += token_estimate
        return True, headers


# ── Schemas ──────────────────────────────────────────────────────────────────

class ChatMessagePayload(BaseModel):
    role: str
    content: str


class ChatCompletionRequestPayload(BaseModel):
    model: str = "mock-stand-in-model"
    messages: List[ChatMessagePayload]
    temperature: Optional[float] = 0.2
    max_tokens: Optional[int] = 800
    stream: Optional[bool] = False
    stream_options: Optional[Dict] = Field(default_factory=dict)


# ── Application State ────────────────────────────────────────────────────────

app = FastAPI(title="Stand-In Model Service", version="1.0.0")
degradation_engine = LoadDegradationEngine()
ingress_limiter = MockIngressLimiter()
latency_sim = LatencySimulator(CALIBRATION_PROFILES["llama-3.3-70b-versatile"])


@app.post("/v1/chat/completions")
async def handle_chat_completions(payload: ChatCompletionRequestPayload):
    """
    OpenAI-compatible inference endpoint replicating Groq LPU latency & limits.
    """
    # Estimate token load
    prompt_tokens = sum(len(msg.content.split()) for msg in payload.messages) * 2
    estimated_total_tokens = prompt_tokens + (payload.max_tokens or 200)

    # 1. Evaluate rate limiter
    allowed, rate_headers = ingress_limiter.check_capacity(estimated_total_tokens)
    if not allowed:
        await asyncio.sleep(RATE_LIMIT_OVERHEAD_MS / 1000.0)
        error_type = (
            "daily_budget_exceeded"
            if ingress_limiter.daily_budget_exhausted
            else "tokens_per_minute_exceeded"
        )
        error_msg = (
            "Daily budget exhausted. Daily request allocation depleted until 00:00 UTC."
            if ingress_limiter.daily_budget_exhausted
            else "Rate limit reached. Quota exceeded for the current 60s window."
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": error_msg,
                    "type": error_type,
                    "code": "rate_limit_exceeded",
                }
            },
            headers={
                **rate_headers,
                "retry-after": str(DEFAULT_RETRY_AFTER_SECONDS),
            },
        )

    # 2. Track connection load
    degradation_engine.record_connection_start()

    try:
        load_factor, should_fail_overload = degradation_engine.evaluate_load_status()

        if should_fail_overload:
            await asyncio.sleep(0.4)
            raise HTTPException(
                status_code=503,
                detail="Inference gateway buffer exhausted due to sustained concurrent overload.",
            )

        # Select synthetic answer
        selected_text = generate_mock_model_response(payload.messages)
        completion_tokens = len(selected_text.split()) * 2
        total_tokens = prompt_tokens + completion_tokens

        # Case A: Streaming SSE response
        if payload.stream:
            return StreamingResponse(
                _stream_tokens(
                    response_text=selected_text,
                    model_name=payload.model,
                    load_factor=load_factor,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    include_usage="include_usage" in (payload.stream_options or {}),
                ),
                media_type="text/event-stream",
                headers=rate_headers,
            )

        # Case B: Non-streaming JSON response
        ttft_ms = latency_sim.sample_time_to_first_token_ms(load_factor)
        generation_ms = (completion_tokens / CALIBRATION_PROFILES["llama-3.3-70b-versatile"].tokens_per_second) * 1000.0
        total_delay_seconds = (ttft_ms + generation_ms) / 1000.0

        await asyncio.sleep(total_delay_seconds)

        response_id = f"chatcmpl_{uuid.uuid4().hex[:12]}"
        return JSONResponse(
            content={
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": selected_text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            },
            headers=rate_headers,
        )

    finally:
        degradation_engine.record_connection_end()


async def _stream_tokens(
    response_text: str,
    model_name: str,
    load_factor: float,
    prompt_tokens: int,
    completion_tokens: int,
    include_usage: bool = False,
) -> AsyncGenerator[str, None]:
    """Streams SSE chunks token-by-token with calibrated delays."""
    response_id = f"chatcmpl_{uuid.uuid4().hex[:12]}"

    # Time To First Token delay
    ttft_ms = latency_sim.sample_time_to_first_token_ms(load_factor)
    await asyncio.sleep(ttft_ms / 1000.0)

    words = response_text.split(" ")
    for idx, word in enumerate(words):
        token_chunk = word + (" " if idx < len(words) - 1 else "")
        chunk_payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": token_chunk},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk_payload)}\n\n"

        inter_token_delay = latency_sim.sample_inter_token_delay_ms(load_factor)
        await asyncio.sleep(inter_token_delay / 1000.0)

    # Finalizer chunk
    final_payload = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    if include_usage:
        final_payload["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    yield f"data: {json.dumps(final_payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": degradation_engine.active_connections,
        "peak_concurrency": degradation_engine.peak_concurrency,
    }


class AdminConfigRequest(BaseModel):
    rate_limiting_active: Optional[bool] = None
    daily_budget_exhausted: Optional[bool] = None
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None
    reset_window: Optional[bool] = True


@app.post("/admin/configure")
async def configure_runtime(payload: AdminConfigRequest):
    """Admin API for load test scenarios (e.g. rate limit halved, budget exhausted)."""
    if payload.rate_limiting_active is not None:
        ingress_limiter.rate_limiting_active = payload.rate_limiting_active
    if payload.daily_budget_exhausted is not None:
        ingress_limiter.daily_budget_exhausted = payload.daily_budget_exhausted
    if payload.rpm_limit is not None:
        ingress_limiter.rpm_limit = payload.rpm_limit
    if payload.tpm_limit is not None:
        ingress_limiter.tpm_limit = payload.tpm_limit
    if payload.reset_window:
        ingress_limiter.reset_counters()

    return {
        "rate_limiting_active": ingress_limiter.rate_limiting_active,
        "daily_budget_exhausted": ingress_limiter.daily_budget_exhausted,
        "rpm_limit": ingress_limiter.rpm_limit,
        "tpm_limit": ingress_limiter.tpm_limit,
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Run Mock Stand-In Model Service")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port)
