"""
Unit & Integration Tests for Indexing, Passage Retrieval, and Reranking.
"""

import pytest
from indexing.text_chunker import TextChunk
from indexing.passage_retriever import PassageRetriever


@pytest.fixture
def sample_chunks():
    return [
        TextChunk(
            chunk_id="chunk_01",
            text="The Transformer relies entirely on self-attention to compute representations of its input and output.",
            token_count=18,
            paper_title="Attention Is All You Need",
            arxiv_id="1706.03762",
            section_name="3.1 Multi-Head Attention",
            page_number=3,
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="chunk_02",
            text="Moshi uses the Mimi neural audio codec to compress speech at 12.5 Hz with 160ms latency.",
            token_count=21,
            paper_title="Moshi: a speech-text foundation model",
            arxiv_id="2410.00037",
            section_name="2.1 Architecture",
            page_number=4,
            chunk_index=1,
        ),
        TextChunk(
            chunk_id="chunk_03",
            text="Dynamic chunking learns variable-length boundary masks for hierarchical sequence modeling.",
            token_count=15,
            paper_title="Dynamic Chunking",
            arxiv_id="2507.07955",
            section_name="3 Method",
            page_number=2,
            chunk_index=2,
        ),
    ]


def test_retriever_build_and_search(tmp_path, sample_chunks):
    retriever = PassageRetriever(index_directory=tmp_path)
    retriever.build_index(sample_chunks)

    # Query for transformer
    results = retriever.search("What is the core mechanism of the Transformer?", top_k=2)
    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk.paper_title == "Attention Is All You Need"
    assert "self-attention" in top_chunk.text


def test_retriever_reranking_boosts_exact_keywords(tmp_path, sample_chunks):
    retriever = PassageRetriever(index_directory=tmp_path)
    retriever.build_index(sample_chunks)

    results_reranked = retriever.search(
        "Mimi neural codec latency",
        top_k=2,
        rerank=True,
        key_terms=["mimi", "latency"],
    )
    assert len(results_reranked) > 0
    top_chunk, score = results_reranked[0]
    assert top_chunk.paper_title == "Moshi: a speech-text foundation model"

