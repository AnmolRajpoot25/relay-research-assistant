"""
Question Analyzer — Step 1 of the multi-step pipeline.

Senior Engineering Perspective:
Instead of sending raw user input directly to vector retrieval, we run a light,
structured analysis step. This achieves three essential goals:
1. Deconstructs complex user prompts into technical search terms suited for arXiv paper indexes.
2. Identifies targeted papers from our catalog, reducing noise.
3. Pre-emptively flags non-answerable or out-of-scope inquiries to avoid wasting expensive
   subsequent drafting tokens.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("QuestionAnalyzer")


@dataclass
class QuestionAnalysis:
    """Structured decomposition of an incoming user inquiry."""
    original_question: str
    reformulated_query: str
    key_technical_terms: List[str] = field(default_factory=list)
    relevant_papers: List[str] = field(default_factory=list)
    is_answerable_from_library: bool = True
    refusal_reason: str = ""
    estimated_complexity: str = "medium"

    @property
    def composite_search_query(self) -> str:
        """Query text passed to the passage retriever."""
        if self.key_technical_terms:
            return f"{self.reformulated_query} {' '.join(self.key_technical_terms)}"
        return self.reformulated_query


ANALYZER_SYSTEM_INSTRUCTIONS = """You are a technical research assistant analyzing inquiries against three specific papers:
1. "Attention Is All You Need" (arXiv:1706.03762v7) - Transformer architecture, self-attention, multi-head attention.
2. "Moshi: a speech-text foundation model for real-time dialogue" (arXiv:2410.00037v1) - Speech-text foundation model, Mimi neural codec, full-duplex conversational AI.
3. "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (arXiv:2507.07955v2) - Dynamic chunking, hierarchical sequence modeling, variable-length chunk boundaries.

Return ONLY a valid JSON object matching this schema:
{
    "reformulated_query": "search query optimized for embedding similarity",
    "key_technical_terms": ["term1", "term2"],
    "relevant_papers": ["paper name"],
    "is_answerable_from_library": true or false,
    "refusal_reason": "explanation if is_answerable_from_library is false, otherwise empty string",
    "estimated_complexity": "simple" or "medium" or "complex"
}

Grounding Rule: If the user asks about topics completely foreign to these three papers (e.g. stock prices, general news, unrelated models), set is_answerable_from_library to false.
"""


class QuestionAnalyzer:
    """Analyzes incoming questions using LLM reasoning to guide retrieval."""

    def __init__(self, model_client):
        self.model_client = model_client

    def analyze(self, question: str):
        """
        Processes the input question and returns a structured QuestionAnalysis
        along with the model call response for telemetry tracking.
        """
        messages = [
            {"role": "system", "content": ANALYZER_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": question.strip()},
        ]

        model_response = self.model_client.complete(
            messages=messages,
            temperature=0.1,
            max_tokens=350,
            stream=False,
        )

        if model_response.error_message:
            logger.warning(
                f"Analyzer LLM call encountered error: {model_response.error_message}. "
                "Engaging resilient fallback parsing."
            )
            return self._build_fallback_analysis(question), model_response

        analysis = self._parse_json_payload(question, model_response.content)
        return analysis, model_response

    def _parse_json_payload(self, question: str, response_text: str) -> QuestionAnalysis:
        cleaned_text = response_text.strip()
        if "```" in cleaned_text:
            segments = cleaned_text.split("```")
            for segment in segments:
                candidate = segment.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    parsed_dictionary = json.loads(candidate)
                    return self._convert_dict_to_analysis(question, parsed_dictionary)
                except Exception:
                    continue

        try:
            parsed_dictionary = json.loads(cleaned_text)
            return self._convert_dict_to_analysis(question, parsed_dictionary)
        except Exception as error:
            logger.warning(f"JSON parsing error in QuestionAnalyzer: {error}. Using heuristic fallback.")
            return self._build_fallback_analysis(question)

    def _convert_dict_to_analysis(self, original_question: str, data: dict) -> QuestionAnalysis:
        return QuestionAnalysis(
            original_question=original_question,
            reformulated_query=str(data.get("reformulated_query", original_question)),
            key_technical_terms=list(data.get("key_technical_terms", [])),
            relevant_papers=list(data.get("relevant_papers", [])),
            is_answerable_from_library=bool(data.get("is_answerable_from_library", True)),
            refusal_reason=str(data.get("refusal_reason", "")),
            estimated_complexity=str(data.get("estimated_complexity", "medium")),
        )

    def _build_fallback_analysis(self, question: str) -> QuestionAnalysis:
        """Deterministic heuristic extraction when LLM JSON output fails."""
        words = [w.strip(",.?!") for w in question.split() if len(w) > 3]
        return QuestionAnalysis(
            original_question=question,
            reformulated_query=question,
            key_technical_terms=words[:6],
            relevant_papers=[],
            is_answerable_from_library=True,
            refusal_reason="",
            estimated_complexity="medium",
        )

