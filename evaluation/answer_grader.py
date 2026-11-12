"""
Answer Grader — Evaluates pipeline responses for results.csv compliance.

Senior Engineering Perspective:
Per the assignment brief (Page 4):
"Every figure in the tables must be computable from this file [results.csv] with columns:
 ... answer_grade (correct, wrong, refused_correctly, refused_wrongly)"

This grading module:
1. Detects refusals via robust heuristic and structural matching.
2. For answerable questions:
   - If properly answered with relevant evidence & citations -> "correct"
   - If refused when the paper answers it -> "refused_wrongly"
   - If factual inaccuracies or ungrounded assertions are detected -> "wrong"
3. For unanswerable questions:
   - If properly refused without hallucination -> "refused_correctly"
   - If model attempts to hallucinate or guess -> "wrong"
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("AnswerGrader")


@dataclass
class EvaluationGradeResult:
    """Individual grading outcome."""
    question_id: str
    question: str
    assigned_grade: str      # 'correct', 'wrong', 'refused_correctly', 'refused_wrongly'
    expected_grade: str
    is_satisfactory: bool
    rationale: str
    citations_present: bool
    matched_keywords: List[str] = field(default_factory=list)


class AnswerGrader:
    """Evaluates question answers against ground truth criteria."""

    def grade(
        self,
        question_data: Dict[str, Any],
        generated_answer: str,
    ) -> EvaluationGradeResult:
        question_id = question_data.get("id", "unknown")
        question_text = question_data["question"]
        expected_grade = question_data["expected_grade"]
        keywords = question_data.get("mandatory_keywords", [])
        target_paper = str(question_data.get("target_paper", ""))

        answer_clean = generated_answer.strip()
        answer_lower = answer_clean.lower()
        is_refusal = self._is_refusal_response(answer_lower)
        citations_valid, valid_citations = self._validate_citations(answer_clean, target_paper)

        matched_kw = [kw for kw in keywords if kw.lower() in answer_lower]

        # Case 1: The question was inherently unanswerable
        if expected_grade == "refused_correctly":
            if is_refusal:
                assigned = "refused_correctly"
                rationale = "System correctly recognized missing library evidence and refused to guess."
            else:
                assigned = "wrong"
                rationale = "System attempted to guess or hallucinate an unanswerable question."

        # Case 2: The question was answerable from the papers
        elif expected_grade == "correct":
            if is_refusal:
                assigned = "refused_wrongly"
                rationale = "System refused a question whose answer exists in the library."
            else:
                # Check keyword overlap and citations
                threshold = max(1, int(len(keywords) * 0.6)) if keywords else 0
                if len(matched_kw) >= threshold and citations_valid:
                    assigned = "correct"
                    rationale = f"Matched {len(matched_kw)}/{len(keywords)} concepts with {len(valid_citations)} verified citations."
                elif len(matched_kw) >= threshold and not citations_valid:
                    assigned = "wrong"
                    rationale = "Correct concepts present, but missing valid paper & section/page citations."
                else:
                    assigned = "wrong"
                    rationale = f"Insufficient concept match ({len(matched_kw)}/{len(keywords)}). Possible ungrounded claim."
        else:
            assigned = "wrong"
            rationale = f"Unrecognized expected_grade: {expected_grade}"

        is_satisfactory = (assigned == expected_grade)

        return EvaluationGradeResult(
            question_id=question_id,
            question=question_text,
            assigned_grade=assigned,
            expected_grade=expected_grade,
            is_satisfactory=is_satisfactory,
            rationale=rationale,
            citations_present=citations_valid,
            matched_keywords=matched_kw,
        )

    def _validate_citations(self, text: str, target_paper: str) -> Tuple[bool, List[str]]:
        """
        Validates that citations are not fabricated square brackets.
        Must contain known paper identifiers and section/page markers.
        """
        bracketed = re.findall(r'\[([^\]]+)\]', text)
        valid = []
        for b in bracketed:
            b_lower = b.lower()
            has_paper = any(
                k in b_lower
                for k in ["attention", "1706.03762", "vaswani", "moshi", "2410.00037", "défossez", "defossez", "chunking", "2507.07955"]
            )
            has_loc = any(k in b_lower for k in ["p.", "page", "§", "section"])
            if has_paper and has_loc:
                valid.append(b)

        if not valid:
            return False, []

        # Target paper alignment check if specified
        if target_paper and target_paper.lower() not in ("none", "", "multiple") and "&" not in target_paper:
            target_key = target_paper.split()[0].lower()
            paper_matched = any(target_key in c.lower() for c in valid)
            return paper_matched, valid

        return True, valid

    def _is_refusal_response(self, text_lower: str) -> bool:
        signatures = [
            "do not contain",
            "cannot answer",
            "no information",
            "not mentioned",
            "does not mention",
            "insufficient evidence",
            "refrain from guessing",
            "outside the scope",
            "cannot be answered",
        ]
        return any(sig in text_lower for sig in signatures)

