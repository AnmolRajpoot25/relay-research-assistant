"""
Unit & Integration Tests for Answer Quality & Provenance Grading.
"""

import pytest
from evaluation.answer_grader import AnswerGrader


@pytest.fixture
def grader():
    return AnswerGrader()


def test_grader_approves_valid_grounded_answer(grader):
    question_data = {
        "id": "q01_test",
        "question": "What replaces recurrence in the Transformer?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["self-attention", "attention", "multi-head"],
    }
    answer = (
        "The Transformer replaces recurrence and convolutions entirely with multi-head "
        "self-attention mechanisms, enabling massive parallelization during training. "
        "[Attention Is All You Need, §3.1, p.3]"
    )
    result = grader.grade(question_data, answer)
    assert result.assigned_grade == "correct"
    assert result.is_satisfactory is True
    assert result.citations_present is True


def test_grader_rejects_hallucinated_answer_without_citations(grader):
    question_data = {
        "id": "q02_test",
        "question": "What replaces recurrence in the Transformer?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["self-attention", "attention", "multi-head"],
    }
    answer = "The Transformer replaces recurrence with self-attention and multi-head attention."
    result = grader.grade(question_data, answer)
    assert result.assigned_grade == "wrong"
    assert result.is_satisfactory is False
    assert result.citations_present is False


def test_grader_rejects_fake_bracketed_citations(grader):
    question_data = {
        "id": "q03_test",
        "question": "What replaces recurrence in the Transformer?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["self-attention", "attention"],
    }
    # Bracketed text with no paper name or page/section
    answer = "The Transformer uses self-attention. [Some Random Bracket, 2021]"
    result = grader.grade(question_data, answer)
    assert result.assigned_grade == "wrong"
    assert result.citations_present is False


def test_grader_approves_honest_refusal(grader):
    question_data = {
        "id": "q04_test",
        "question": "What is Google's current market cap?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "mandatory_keywords": [],
    }
    answer = "The papers in this library do not contain information regarding Google's market cap. I must refrain from guessing."
    result = grader.grade(question_data, answer)
    assert result.assigned_grade == "refused_correctly"
    assert result.is_satisfactory is True


def test_grader_flags_improper_refusal_as_refused_wrongly(grader):
    question_data = {
        "id": "q05_test",
        "question": "How many heads are in the Transformer?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["8"],
    }
    answer = "I do not contain information about the number of heads."
    result = grader.grade(question_data, answer)
    assert result.assigned_grade == "refused_wrongly"
    assert result.is_satisfactory is False

