"""
Curated Evaluation Dataset (18 Test Questions).

Senior Engineering Perspective:
The assignment mandates testing answer quality on at least 15 custom questions across our library:
- "Attention Is All You Need" (arXiv 1706.03762v7)
- "Moshi: a speech-text foundation model for real-time dialogue" (arXiv 2410.00037v1)
- "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling" (arXiv 2507.07955v2)

To rigorously challenge the system, the test suite includes:
1. Direct factual questions (expected: 'correct')
2. Cross-paper architectural comparisons (expected: 'correct')
3. Questions completely outside library scope (expected: 'refused_correctly')
4. Adversarial deceptive questions asking for unstated details (expected: 'refused_correctly')
"""

from typing import Any, Dict, List

EVALUATION_QUESTIONS: List[Dict[str, Any]] = [
    # ── Category 1: Attention Is All You Need (Vaswani et al., 2017) ─────────
    {
        "id": "q01_transformer_mechanism",
        "question": "What is the primary architectural mechanism replacing recurrence and convolutions in the Transformer?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["self-attention", "attention", "multi-head"],
    },
    {
        "id": "q02_transformer_heads",
        "question": "How many parallel attention heads are employed in the base Transformer model, and what is d_k?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["8", "64"],
    },
    {
        "id": "q03_positional_encoding",
        "question": "How does the Transformer inject sequence order information without recurrence or convolution?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["positional encoding", "sine", "cosine", "sinusoidal"],
    },
    {
        "id": "q04_transformer_optimizer",
        "question": "Which optimizer and learning rate schedule warmup strategy was used to train the base Transformer models?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["adam", "warmup", "learning rate"],
    },
    {
        "id": "q05_transformer_bleu",
        "question": "What BLEU score did the big Transformer model achieve on the WMT 2014 English-to-French translation task?",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need",
        "mandatory_keywords": ["41.0", "41.8", "bleu"],
    },

    # ── Category 2: Moshi Speech-Text Dialogue (Défossez et al., 2024) ───────
    {
        "id": "q06_moshi_codec",
        "question": "What neural audio codec does Moshi employ to discretize acoustic streams, and what is its frame rate?",
        "expected_grade": "correct",
        "target_paper": "Moshi",
        "mandatory_keywords": ["mimi", "codec", "neural"],
    },
    {
        "id": "q07_moshi_duplex",
        "question": "How does Moshi achieve full-duplex conversational interaction between two participants?",
        "expected_grade": "correct",
        "target_paper": "Moshi",
        "mandatory_keywords": ["full-duplex", "parallel", "streams", "inner monologue"],
    },
    {
        "id": "q08_moshi_latency",
        "question": "What is the theoretical latency of Moshi's streaming dialogue architecture?",
        "expected_grade": "correct",
        "target_paper": "Moshi",
        "mandatory_keywords": ["160ms", "latency", "real-time"],
    },
    {
        "id": "q09_moshi_dual_stream",
        "question": "What modalities are modeled jointly by Moshi's multi-stream transformer backbone?",
        "expected_grade": "correct",
        "target_paper": "Moshi",
        "mandatory_keywords": ["speech", "text", "audio"],
    },

    # ── Category 3: Dynamic Chunking (2025) ──────────────────────────────────
    {
        "id": "q10_dynamic_chunking_concept",
        "question": "What fundamental problem does Dynamic Chunking address in hierarchical sequence modeling?",
        "expected_grade": "correct",
        "target_paper": "Dynamic Chunking",
        "mandatory_keywords": ["variable-length", "chunks", "hierarchical", "boundaries"],
    },
    {
        "id": "q11_dynamic_chunking_boundaries",
        "question": "How does Dynamic Chunking establish segment boundaries in an end-to-end differentiable manner?",
        "expected_grade": "correct",
        "target_paper": "Dynamic Chunking",
        "mandatory_keywords": ["learned", "boundary", "differentiable", "end-to-end"],
    },
    {
        "id": "q12_dynamic_chunking_compression",
        "question": "How does dynamic chunking reduce computational complexity in long sequence transformers?",
        "expected_grade": "correct",
        "target_paper": "Dynamic Chunking",
        "mandatory_keywords": ["compression", "hierarchical", "reduction", "complexity"],
    },

    # ── Category 4: Cross-Paper Syntheses ────────────────────────────────────
    {
        "id": "q13_cross_attention_comparison",
        "question": "Compare how self-attention is applied in the original Transformer versus in the Moshi speech-text architecture.",
        "expected_grade": "correct",
        "target_paper": "Attention Is All You Need & Moshi",
        "mandatory_keywords": ["attention", "speech", "text"],
    },

    # ── Category 5: Questions that MUST be Refused (Unanswerable / Out of Library) ─
    {
        "id": "q14_unanswerable_stock_price",
        "question": "What was the closing stock price of Nvidia on October 15, 2024?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "mandatory_keywords": [],
    },
    {
        "id": "q15_unanswerable_gpt4_details",
        "question": "What is the exact parameter count and mixture-of-experts routing algorithm used by GPT-4?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "mandatory_keywords": [],
    },
    {
        "id": "q16_unanswerable_quantum_physics",
        "question": "Explain the experimental evidence for quantum gravity in black hole Hawking radiation.",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "mandatory_keywords": [],
    },
    {
        "id": "q17_unanswerable_author_birthplace",
        "question": "Where was Ashish Vaswani born, and what was his primary school?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "mandatory_keywords": [],
    },
    {
        "id": "q18_unanswerable_moshi_financials",
        "question": "What was Kyutai's total budget and compute expenditure to train Moshi in Euro currency?",
        "expected_grade": "refused_correctly",
        "target_paper": "None",
        "mandatory_keywords": [],
    },
]

