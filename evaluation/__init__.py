"""
Evaluation Package — Benchmark test dataset and automated answer grading.
"""

from evaluation.test_questions import EVALUATION_QUESTIONS
from evaluation.answer_grader import AnswerGrader, EvaluationGradeResult

__all__ = ["EVALUATION_QUESTIONS", "AnswerGrader", "EvaluationGradeResult"]

