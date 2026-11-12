"""
Response Drafter — Step 3 of the multi-step pipeline.

Senior Engineering Perspective:
The drafter combines prompt engineering, contextual grounding, and citation adherence.
Critical constraints from assignment brief:
- "Every answer must cite where in the papers it came from."
- "If the papers do not say, the assistant must say so instead of guessing."
- "A company never wants to send a wrong or embarrassing answer just to stay fast."

When quality feedback is provided from a previous iteration, this module integrates
that critique into an enhanced prompt to correct deficiencies during re-draft loops.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.passage_finder import RetrievalOutcome
from pipeline.question_analyzer import QuestionAnalysis

logger = logging.getLogger("ResponseDrafter")


@dataclass
class DraftResult:
    """Outcome of a drafting cycle."""
    answer_text: str
    is_refusal: bool
    refusal_reason: str = ""
    citations_identified: List[str] = field(default_factory=list)
    draft_iteration: int = 1


DRAFTER_SYSTEM_INSTRUCTIONS = """You are a senior research engineer answering questions from three specific peer-reviewed papers.
You must adhere strictly to these non-negotiable operational rules:
1. GROUNDING: Base your answer EXCLUSIVELY on the provided passage context. Do not extrapolate, hallucinate, or import outside training knowledge.
2. CITATION: Every factual assertion must be immediately followed by its bracketed citation label from the source passage (e.g., [Attention Is All You Need, §3.2, p.5]).
3. HONEST REFUSAL: If the provided passages do not explicitly provide the information needed to answer the question, you MUST refuse:
   "The papers in this library do not contain sufficient evidence to answer [topic]. I must refrain from guessing."
4. TONE & STRUCTURE: Concise, precise, professional technical language. Avoid conversational filler.
"""

REDRAFT_CRITIQUE_TEMPLATE = """
--- PREVIOUS DRAFT CRITIQUE ---
A rigorous quality inspection identified the following issues with your previous attempt:
{critique_feedback}

Please re-synthesize your response, correcting these flaws while maintaining strict citation fidelity.
"""


class ResponseDrafter:
    """Synthesizes cited answers from retrieved passage context."""

    def __init__(self, model_client):
        self.model_client = model_client

    def draft(
        self,
        analysis: QuestionAnalysis,
        retrieval: RetrievalOutcome,
        draft_iteration: int = 1,
        critique_feedback: str = "",
    ):
        """
        Generates an answer or honest refusal based on retrieved evidence.
        Returns a tuple of (DraftResult, ModelResponse).
        """
        # Scenario A: Question is definitively outside library scope
        if not analysis.is_answerable_from_library:
            reason = analysis.refusal_reason or "Inquiry falls outside the scope of the 3 library papers."
            refusal_text = (
                f"The papers in this library do not contain information regarding this inquiry. "
                f"Reason: {reason}"
            )
            draft = DraftResult(
                answer_text=refusal_text,
                is_refusal=True,
                refusal_reason=reason,
                draft_iteration=draft_iteration,
            )
            # Create a lightweight pseudo ModelResponse to preserve type signatures
            from models.groq_client import ModelResponse
            return draft, ModelResponse(content=refusal_text, model_used="deterministic_guardrail")

        # Scenario B: Vector retrieval returned zero matching passages
        if not retrieval.has_sufficient_evidence:
            refusal_text = (
                f"No relevant passages were discovered in 'Attention Is All You Need', 'Moshi', "
                f"or 'Dynamic Chunking' to substantiate an answer for: '{analysis.original_question}'. "
                "Per system policy, I cannot guess without supporting citations."
            )
            draft = DraftResult(
                answer_text=refusal_text,
                is_refusal=True,
                refusal_reason="Zero relevant passages retrieved from index.",
                draft_iteration=draft_iteration,
            )
            from models.groq_client import ModelResponse
            return draft, ModelResponse(content=refusal_text, model_used="deterministic_guardrail")

        # Scenario C: Compose prompt and invoke LLM
        system_prompt = DRAFTER_SYSTEM_INSTRUCTIONS
        if draft_iteration > 1 and critique_feedback:
            system_prompt += REDRAFT_CRITIQUE_TEMPLATE.format(critique_feedback=critique_feedback)

        user_content = (
            f"Question:\n{analysis.original_question}\n\n"
            f"Retrieved Passages:\n{retrieval.compile_context_prompt()}\n\n"
            "Formulate your complete, rigorously cited response now:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        model_response = self.model_client.complete(
            messages=messages,
            temperature=0.2,
            max_tokens=900,
            stream=True,
        )

        if model_response.error_message:
            logger.error(f"ResponseDrafter LLM failure: {model_response.error_message}")
            fallback_text = "Service temporarily unavailable to draft answer. Please retry shortly."
            return DraftResult(
                answer_text=fallback_text,
                is_refusal=True,
                refusal_reason=model_response.error_message,
                draft_iteration=draft_iteration,
            ), model_response

        is_refusal = self._detect_refusal_intent(model_response.content)
        citations = self._extract_citations(model_response.content)

        draft = DraftResult(
            answer_text=model_response.content,
            is_refusal=is_refusal,
            refusal_reason="Model elected to refuse based on insufficient passage evidence." if is_refusal else "",
            citations_identified=citations,
            draft_iteration=draft_iteration,
        )

        return draft, model_response

    def _extract_citations(self, text: str) -> List[str]:
        """Extracts bracketed citations matching our paper formats."""
        matches = re.findall(r'\[([^\]]+)\]', text)
        valid_citations = [
            m for m in matches
            if any(marker in m.lower() for marker in ["p.", "§", "section", "attention", "moshi", "chunking"])
        ]
        return valid_citations

    def _detect_refusal_intent(self, text: str) -> bool:
        """Heuristic detection of refusal or lack of evidence statements."""
        signatures = [
            "do not contain",
            "cannot answer",
            "no information",
            "not mentioned in",
            "does not mention",
            "insufficient evidence",
            "refrain from guessing",
            "outside the scope",
        ]
        text_lower = text.lower()
        return any(signature in text_lower for signature in signatures)

