"""
Text Chunker — splits paper sections into retrieval-friendly chunks.

Why overlapping chunks? Without overlap, a relevant sentence that sits
right at a chunk boundary gets split across two chunks, and neither chunk
alone has enough context. We trade ~15% extra storage for much better
retrieval recall.

The assignment requires testing at least three chunking settings, so this
module is designed to be reconfigurable at runtime.
"""

import logging
import re
from dataclasses import dataclass

import tiktoken

from indexing.paper_parser import ExtractedSection

logger = logging.getLogger(__name__)

# Tokenizer wrapper that safely falls back to statistical word/character heuristics
# if tiktoken cannot access external resources
class ResilientTokenizer:
    def __init__(self):
        self._enc = None
        try:
            self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None

    def encode(self, text: str) -> list[int]:
        if self._enc is not None:
            try:
                return self._enc.encode(text)
            except Exception:
                pass
        # Fallback estimation: average word in technical English is ~1.3 tokens
        words = text.split()
        estimated = max(1, int(len(words) * 1.33))
        return list(range(estimated))

_tokenizer = ResilientTokenizer()


@dataclass
class TextChunk:
    """A retrieval-ready chunk of text with full provenance."""

    chunk_id: str  # Unique identifier (paper_section_chunk)
    text: str  # The actual chunk content
    token_count: int  # Number of tokens in this chunk
    paper_title: str  # Which paper this came from
    arxiv_id: str  # arXiv identifier
    section_name: str  # Section within the paper
    page_number: int  # Page where this section starts
    chunk_index: int  # Position within the section's chunks

    @property
    def citation_label(self) -> str:
        """Citation string for including in answers."""
        short_title = self.paper_title.split(":")[0][:30]
        return f"[{short_title}, {self.section_name}, p.{self.page_number}]"


@dataclass
class ChunkingConfig:
    """Configuration for a particular chunking experiment."""

    chunk_size_tokens: int = 512
    overlap_tokens: int = 64
    label: str = "512_tok_64_overlap"  # For results.csv


# Pre-defined settings for the retrieval experiment
CHUNKING_EXPERIMENTS: dict[str, ChunkingConfig] = {
    "small": ChunkingConfig(
        chunk_size_tokens=256, overlap_tokens=32, label="256_tok_32_overlap"
    ),
    "medium": ChunkingConfig(
        chunk_size_tokens=512, overlap_tokens=64, label="512_tok_64_overlap"
    ),
    "large": ChunkingConfig(
        chunk_size_tokens=1024, overlap_tokens=128, label="1024_tok_128_overlap"
    ),
}


class TextChunker:
    """Splits extracted paper sections into overlapping, token-counted chunks."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or CHUNKING_EXPERIMENTS["medium"]
        logger.info(
            f"TextChunker initialized: {self.config.chunk_size_tokens} tokens/chunk, "
            f"{self.config.overlap_tokens} token overlap"
        )

    def chunk_sections(self, sections: list[ExtractedSection]) -> list[TextChunk]:
        """Split a list of ExtractedSections into retrieval chunks."""
        all_chunks: list[TextChunk] = []

        for section in sections:
            section_chunks = self._split_section_into_chunks(section)
            all_chunks.extend(section_chunks)

        logger.info(
            f"Created {len(all_chunks)} chunks from {len(sections)} sections "
            f"(avg {sum(c.token_count for c in all_chunks) / max(len(all_chunks), 1):.0f} tokens/chunk)"
        )
        return all_chunks

    def _split_section_into_chunks(
        self, section: ExtractedSection
    ) -> list[TextChunk]:
        """Split a single section into overlapping chunks."""
        sentences = self._split_into_sentences(section.text)
        if not sentences:
            return []

        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_token_count = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_tokens = len(_tokenizer.encode(sentence))

            # If adding this sentence would exceed the chunk size, finalize current chunk
            if (
                current_token_count + sentence_tokens > self.config.chunk_size_tokens
                and current_sentences
            ):
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{section.arxiv_id}_{chunk_index}",
                        text=chunk_text,
                        token_count=current_token_count,
                        paper_title=section.paper_title,
                        arxiv_id=section.arxiv_id,
                        section_name=section.section_name,
                        page_number=section.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

                # Keep overlap: rewind to include some sentences from the end
                overlap_sentences, overlap_tokens = self._calculate_overlap(
                    current_sentences
                )
                current_sentences = overlap_sentences
                current_token_count = overlap_tokens

            current_sentences.append(sentence)
            current_token_count += sentence_tokens

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                TextChunk(
                    chunk_id=f"{section.arxiv_id}_{chunk_index}",
                    text=chunk_text,
                    token_count=len(_tokenizer.encode(chunk_text)),
                    paper_title=section.paper_title,
                    arxiv_id=section.arxiv_id,
                    section_name=section.section_name,
                    page_number=section.page_number,
                    chunk_index=chunk_index,
                )
            )

        return chunks

    def _calculate_overlap(
        self, sentences: list[str]
    ) -> tuple[list[str], int]:
        """
        Find the tail sentences that fit within the overlap budget.

        We walk backwards from the end, accumulating tokens until we hit
        the overlap limit.
        """
        overlap_sentences: list[str] = []
        accumulated_tokens = 0

        for sentence in reversed(sentences):
            sentence_tokens = len(_tokenizer.encode(sentence))
            if accumulated_tokens + sentence_tokens > self.config.overlap_tokens:
                break
            overlap_sentences.insert(0, sentence)
            accumulated_tokens += sentence_tokens

        return overlap_sentences, accumulated_tokens

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """
        Split text into sentences using regex.

        Academic papers have tricky punctuation (e.g., "Fig. 3", "et al."),
        so we use a pattern that handles common abbreviations.
        """
        # Split on period/question/exclamation followed by space and uppercase letter
        # but not after common abbreviations
        abbreviations = (
            r"(?<!\bFig)(?<!\bEq)(?<!\bRef)(?<!\bet al)"
            r"(?<!\bvs)(?<!\bDr)(?<!\bMr)(?<!\bMs)"
            r"(?<!\bi\.e)(?<!\be\.g)"
        )
        pattern = abbreviations + r"[.!?]\s+"

        raw_splits = re.split(pattern, text)
        # Clean up and filter empty strings
        sentences = [s.strip() for s in raw_splits if s.strip()]
        return sentences
