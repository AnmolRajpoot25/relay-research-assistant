"""
The Relay — Production API Service.

Senior Engineering Perspective:
Exposes HTTP endpoints for interacting with the multi-step research paper Q&A pipeline,
inspecting real-time token budgets, and probing health telemetry.

Endpoints:
  POST /ask             Process inquiry through 4-stage pipeline with citations
  GET  /health          Kubernetes/liveness health check
  GET  /status          Model tier routing, quota budgets, and aggregate metrics
  POST /admin/snapshot  Trigger an atomic codebase and session state backup
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backup_system.snapshot_manager import SnapshotManager
from config import INDEX_DIRECTORY, settings
from indexing.passage_retriever import PassageRetriever
from models.model_client import ModelClient
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger("TheRelayServer")

# Global singleton handles
passage_retriever: Optional[PassageRetriever] = None
model_client: Optional[ModelClient] = None
orchestrator: Optional[PipelineOrchestrator] = None
snapshot_manager: Optional[SnapshotManager] = None


@asynccontextmanager
async def service_lifespan(app: FastAPI):
    global passage_retriever, model_client, orchestrator, snapshot_manager

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Initializing The Relay production services...")

    snapshot_manager = SnapshotManager()
    passage_retriever = PassageRetriever(index_directory=INDEX_DIRECTORY)

    if not passage_retriever.load_index():
        logger.warning(
            "Saved vector index not found. Building index or running mock setup may be required."
        )

    model_client = ModelClient()
    orchestrator = PipelineOrchestrator(
        model_client=model_client,
        passage_retriever=passage_retriever,
    )

    logger.info("The Relay API service is online and ready to accept traffic.")
    yield
    logger.info("Shutting down The Relay service cleanly.")


app = FastAPI(
    title="The Relay — Research Paper Assistant",
    description="Multi-step cited answer engine backed by Groq LPUs with resilient fallback",
    version="1.0.0",
    lifespan=service_lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QuestionQueryRequest(BaseModel):
    question: str = Field(..., description="Technical inquiry regarding the 3 arXiv papers")
    retrieval_setting: Optional[str] = Field(default="default", description="Chunk size & retrieval configuration")


class QuestionQueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    status: str
    first_token_ms: float
    total_ms: float
    tokens_used: int
    quality_loops_executed: int
    citations_count: int
    model_used: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=QuestionQueryResponse)
async def ask_question_endpoint(payload: QuestionQueryRequest):
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Pipeline orchestrator not initialized.")

    execution = await asyncio.to_thread(
        orchestrator.process_question,
        question=payload.question,
        retrieval_setting=payload.retrieval_setting or "default",
    )

    return QuestionQueryResponse(
        request_id=execution.request_id,
        question=execution.question,
        answer=execution.answer,
        status=execution.status,
        first_token_ms=round(execution.first_token_ms, 1),
        total_ms=round(execution.total_ms, 1),
        tokens_used=execution.total_tokens,
        quality_loops_executed=execution.quality_loops_executed,
        citations_count=execution.citations_count,
        model_used=execution.model_used,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "index_loaded": (passage_retriever.total_chunks > 0) if passage_retriever else False,
        "active_model_tier": model_client.current_tier if model_client else "unknown",
    }


@app.get("/status")
async def runtime_status():
    if not model_client:
        return {"status": "uninitialized"}

    return {
        "status": "operational",
        "model_routing": model_client.get_metrics_summary(),
    }


@app.post("/admin/snapshot")
async def create_system_snapshot():
    """Trigger zero-loss codebase and state snapshot."""
    if not snapshot_manager:
        raise HTTPException(status_code=503, detail="Snapshot manager uninitialized.")

    snapshot_dir = snapshot_manager.create_codebase_snapshot(label="api_triggered")
    return {"status": "snapshot_created", "snapshot_path": str(snapshot_dir)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)

