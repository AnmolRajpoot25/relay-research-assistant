"""
Quality Checker — Step 4 of the multi-step pipeline (Branch and Loop Controller).

Senior Engineering Perspective:
To guarantee production safety ("never send a wrong or embarrassing answer just to stay fast"),
this module acts as an automated critic. It verifies:
1. Citation Authenticity: Ensures cited sections and pages actually exist in the retrieved passages.
2. Grounding Verification: Confirms no hallucinated claims crept into the answer.
3. Completeness: Checks if the user's specific core question was directly addressed.

Branch/Loop Mechanism:
If the quality check passes -> Pipeline advances to citation formatting and client delivery.
If the quality check fails -> Pipeline loops back to Step 3 (ResponseDrafter) with actionable
critique, until a ceiling of `max_loops` is reached.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from pipeline.passage_finder import RetrievalOutcome
from pipeline.response_drafter import DraftResult

logger = logging.getLogger("QualityChecker")


@dataclass
class QualityEvaluation:
    """Detailed audit verdict of a drafted response."""
    passes_audit: bool
    confidence_score: float
    detected_flaws: List[str] = field(default_factory=list)
    actionable_critique: str = ""
    citations_verified: int = 0
    citations_flagged: int = 0

    @property
    def requires_redraft(self) -> bool:
        return (not self.passes_audit) and len(self.detected_flaws) > 0


QUALITY_AUDIT_PROMPT = """You are an uncompromising technical auditor evaluating an AI assistant's draft answer.

Original Question:
{question}

Available Source Passages:
{passages_summary}

Draft Answer to Evaluate:
{draft_answer}

Perform a rigorous evaluation and return ONLY a JSON object with this exact structure:
{{
    "passes_audit": true or false,
    "confidence_score": 0.0 to 1.0,
    "detected_flaws": ["flaw 1", "flaw 2"],
    "citations_verified": <integer count of citations correctly matching source passages>,
    "citations_flagged": <integer count of citations that do NOT match any source passage>,
    "actionable_critique": "explicit instructions for the drafter to rectify errors"
}}

Evaluation Standards:
1. Passages Grounding: If an answer makes technical claims not found in the passages, FAIL it.
2. Citation Validity: If citations don't align with the provided sources, FAIL it.
3. If the answer is an honest and accurate refusal because the passages don't cover the topic, PASS it.
"""


class QualityChecker:
    """Verifies answer accuracy and determines whether to loop back for re-drafting."""

    def __init__(self, model_client):
        self.model_client = model_client

    def inspect(
        self,
        draft: DraftResult,
        retrieval: RetrievalOutcome,
        original_question: str,
    ):
        """
        Audits the draft against the retrieved source passages.
        Returns a tuple of (QualityEvaluation, Optional[ModelResponse]).
        """
        # Fast path 1: If the draft is an explicit refusal
        if draft.is_refusal:
            # Check if refusal was legitimate
            if not retrieval.has_sufficient_evidence:
                return QualityEvaluation(
                    passes_audit=True,
                    confidence_score=0.95,
                    detected_flaws=[],
                    actionable_critique="",
                    citations_verified=0,
                    citations_flagged=0,
                ), None
            # If high-relevance passages were retrieved but model refused without reason, flag it
            top_score = retrieval.passages[0].relevance_score if retrieval.passages else 0.0
            if top_score > 0.65 and len(retrieval.passages) >= 2:
                return QualityEvaluation(
                    passes_audit=False,
                    confidence_score=0.70,
                    detected_flaws=["Premature refusal despite strong semantic passage matches."],
                    actionable_critique=(
                        f"Relevant passages exist with high similarity ({top_score:.2f}). "
                        "Do not refuse; synthesize an answer directly from the supplied evidence."
                    ),
                    citations_verified=0,
                    citations_flagged=0,
                ), None

            return QualityEvaluation(
                passes_audit=True,
                confidence_score=0.85,
                detected_flaws=[],
                actionable_critique="",
                citations_verified=0,
                citations_flagged=0,
            ), None

        # Fast path 2: Check basic citation presence
        if not draft.citations_identified:
            return QualityEvaluation(
                passes_audit=False,
                confidence_score=0.90,
                detected_flaws=["Missing mandatory bracketed citations."],
                actionable_critique=(
                    "Your answer failed to include required source citations. "
                    "Ensure every key assertion ends with [Paper Title, Section, p.X]."
                ),
                citations_verified=0,
                citations_flagged=0,
            ), None

        # Full LLM-based audit
        passages_summary = "\n".join(
            f"- {p.citation_label}: {p.text[:180]}..."
            for p in retrieval.passages
        )

        formatted_prompt = QUALITY_AUDIT_PROMPT.format(
            question=original_question,
            passages_summary=passages_summary,
            draft_answer=draft.answer_text,
        )

        messages = [
            {"role": "system", "content": "You are a strict QA auditor. Return ONLY JSON."},
            {"role": "user", "content": formatted_prompt},
        ]

        model_response = self.model_client.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=400,
            stream=False,
        )

        if model_response.error_message:
            logger.warning(
                f"Quality audit LLM call encountered error: {model_response.error_message}. "
                "Failing closed per safety guarantee: unverified answers are rejected."
            )
            return QualityEvaluation(
                passes_audit=False,
                confidence_score=0.0,
                detected_flaws=[f"Audit verification failure: {model_response.error_message}"],
                actionable_critique="Auditor unavailable to substantiate factual claims.",
                citations_verified=0,
                citations_flagged=len(draft.citations_identified),
            ), model_response

        evaluation = self._parse_audit_json(model_response.content, draft)
        logger.info(
            f"Quality Audit Verdict: {'PASS' if evaluation.passes_audit else 'REJECT'} "
            f"(Confidence: {evaluation.confidence_score:.2f}, Flaws: {len(evaluation.detected_flaws)})"
        )
        return evaluation, model_response

    def _parse_audit_json(self, raw_text: str, draft: DraftResult) -> QualityEvaluation:
        cleaned = raw_text.strip()
        if "```" in cleaned:
            parts = cleaned.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    data = json.loads(candidate)
                    return self._dict_to_evaluation(data, draft)
                except Exception:
                    continue

        try:
            data = json.loads(cleaned)
            return self._dict_to_evaluation(data, draft)
        except Exception as error:
            logger.warning(f"Failed to parse audit JSON: {error}. Failing closed per safety policy.")
            return QualityEvaluation(
                passes_audit=False,
                confidence_score=0.0,
                detected_flaws=["Auditor output was unparseable; cannot certify claims."],
                actionable_critique="Re-synthesize answer adhering strictly to cited evidence.",
                citations_verified=0,
                citations_flagged=len(draft.citations_identified),
            )

    def _dict_to_evaluation(self, data: dict, draft: DraftResult) -> QualityEvaluation:
        # Strict validation: passes_audit must be explicitly boolean True
        raw_passes = data.get("passes_audit")
        if raw_passes is not True:
            passes = False
        else:
            passes = True

        # Strict validation of confidence score
        raw_conf = data.get("confidence_score")
        try:
            confidence = float(raw_conf)
            if not (0.0 <= confidence <= 1.0):
                confidence = 0.0
                passes = False
        except (TypeError, ValueError):
            confidence = 0.0
            passes = False

        flaws = data.get("detected_flaws")
        if not isinstance(flaws, list):
            flaws = [str(flaws)] if flaws else []

        if not passes and not flaws:
            flaws = ["Auditor failed draft verification without specifying flaws."]

        return QualityEvaluation(
            passes_audit=passes,
            confidence_score=confidence,
            detected_flaws=flaws,
            actionable_critique=str(data.get("actionable_critique", "")),
            citations_verified=int(data.get("citations_verified", 0) or 0),
            citations_flagged=int(data.get("citations_flagged", 0) or 0),
        )

