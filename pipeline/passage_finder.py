"""
Passage Finder — Step 2 of the multi-step pipeline (Tool Invocation).

Senior Engineering Perspective:
This step fulfills the "at least one tool call" requirement by querying the
vector database / semantic index of the 3 arXiv papers.
Key engineering practices:
1. Deterministic execution: Does not consume LLM tokens or Groq rate limits.
2. Latency tracking: Accurately records retrieval overhead in milliseconds.
3. Citation grounding: Pre-attaches precise provenance labels (Section, Page)
   so the downstream ResponseDrafter cannot hallucinate source locations.
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from indexing.passage_retriever import PassageRetriever
from indexing.text_chunker import TextChunk
from pipeline.question_analyzer import QuestionAnalysis

logger = logging.getLogger("PassageFinder")


@dataclass
class RetrievedPassage:
    """Passage chunk enriched with citation metadata and relevance scoring."""
    chunk_id: str
    text: str
    citation_label: str
    relevance_score: float
    paper_title: str
    section_name: str
    page_number: int

    def format_for_prompt_context(self) -> str:
        return (
            f"--- SOURCE CITATION: {self.citation_label} (Relevance: {self.relevance_score:.2f}) ---\n"
            f"{self.text.strip()}\n"
            f"--- END SOURCE ---"
        )


@dataclass
class RetrievalOutcome:
    """Complete summary of the vector search tool call."""
    passages: List[RetrievedPassage]
    query_used: str
    total_index_size: int
    duration_ms: float

    @property
    def has_sufficient_evidence(self) -> bool:
        return len(self.passages) > 0

    def compile_context_prompt(self) -> str:
        if not self.passages:
            return "No relevant passages were found in the library."
        return "\n\n".join(passage.format_for_prompt_context() for passage in self.passages)


class PassageFinder:
    """Tool wrapper that searches the indexing layer."""

    def __init__(
        self,
        retriever: PassageRetriever,
        default_top_k: int = 5,
        minimum_similarity: float = 0.20,
    ):
        self.retriever = retriever
        self.default_top_k = default_top_k
        self.minimum_similarity = minimum_similarity

    def retrieve(
        self,
        analysis: QuestionAnalysis,
        top_k: Optional[int] = None,
        minimum_similarity: Optional[float] = None,
        rerank: bool = False,
    ) -> RetrievalOutcome:
        """
        Executes semantic search against the pre-embedded paper passages.
        """
        search_query = analysis.composite_search_query
        limit = top_k or self.default_top_k
        min_sim = minimum_similarity if minimum_similarity is not None else self.minimum_similarity

        start_time = time.perf_counter()
        raw_matches = self.retriever.search(
            query=search_query,
            top_k=limit,
            minimum_similarity=min_sim,
            rerank=rerank,
            key_terms=analysis.key_technical_terms,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        enriched_passages: List[RetrievedPassage] = []
        for chunk, similarity_score in raw_matches:
            enriched_passages.append(
                RetrievedPassage(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    citation_label=chunk.citation_label,
                    relevance_score=similarity_score,
                    paper_title=chunk.paper_title,
                    section_name=chunk.section_name,
                    page_number=chunk.page_number,
                )
            )

        outcome = RetrievalOutcome(
            passages=enriched_passages,
            query_used=search_query,
            total_index_size=self.retriever.total_chunks,
            duration_ms=duration_ms,
        )

        logger.info(
            f"PassageFinder retrieved {len(enriched_passages)} passages "
            f"in {duration_ms:.2f}ms for query: '{search_query[:60]}...'"
        )
        return outcome

