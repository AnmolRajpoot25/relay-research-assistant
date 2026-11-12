"""
Paper Parser — extracts structured text from arXiv PDF files.

We use PyMuPDF (fitz) for extraction because it handles the two-column
layouts common in academic papers reasonably well. We also track which
page and approximate section each piece of text came from, because every
answer must cite WHERE in the papers it found its evidence.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)


@dataclass
class PaperMetadata:
    """Identifying information for a research paper."""

    title: str
    arxiv_id: str
    version: int
    filename: str


@dataclass
class ExtractedSection:
    """A contiguous block of text from a paper, with its source info."""

    paper_title: str
    arxiv_id: str
    section_name: str  # e.g., "Abstract", "3.2 Multi-Head Attention"
    page_number: int
    text: str

    @property
    def citation_label(self) -> str:
        """Human-readable citation like '[Attention, §3.2, p.5]'."""
        short_title = self.paper_title.split(":")[0][:30]
        return f"[{short_title}, {self.section_name}, p.{self.page_number}]"


# Papers we need to process
PAPER_CATALOG = [
    PaperMetadata(
        title="Attention Is All You Need",
        arxiv_id="1706.03762",
        version=7,
        filename="attention_is_all_you_need_v7.pdf",
    ),
    PaperMetadata(
        title="Moshi: a speech-text foundation model for real-time dialogue",
        arxiv_id="2410.00037",
        version=1,
        filename="moshi_speech_text_v1.pdf",
    ),
    PaperMetadata(
        title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        arxiv_id="2507.07955",
        version=2,
        filename="dynamic_chunking_v2.pdf",
    ),
]


class PaperParser:
    """Extracts structured text from research paper PDFs."""

    def __init__(self, papers_directory: Path):
        self.papers_directory = papers_directory

    def parse_all_papers(self) -> list[ExtractedSection]:
        """Parse all papers in the catalog and return sections."""
        all_sections: list[ExtractedSection] = []

        for paper_info in PAPER_CATALOG:
            pdf_path = self.papers_directory / paper_info.filename
            if not pdf_path.exists():
                logger.warning(
                    f"Paper not found: {pdf_path}. Run download_papers.py first."
                )
                continue
            sections = self._extract_sections_from_pdf(pdf_path, paper_info)
            all_sections.extend(sections)
            logger.info(
                f"Extracted {len(sections)} sections from '{paper_info.title}'"
            )

        return all_sections

    def _extract_sections_from_pdf(
        self, pdf_path: Path, paper_info: PaperMetadata
    ) -> list[ExtractedSection]:
        """Extract text sections from a single PDF."""
        document = pymupdf.open(str(pdf_path))
        sections: list[ExtractedSection] = []
        current_section_name = "Preamble"
        current_section_text = ""
        current_page = 1

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            # Extract text blocks sorted by position (handles columns better)
            text_blocks = page.get_text("blocks", sort=True)

            for block in text_blocks:
                # block format: (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] != 0:  # Skip image blocks (type != 0)
                    continue

                block_text = block[4].strip()
                if not block_text:
                    continue

                # Detect section headings: short lines, often numbered, often bold
                # Heuristic: lines under 80 chars that start with a number or known heading
                is_likely_heading = self._looks_like_section_heading(block_text)

                if is_likely_heading:
                    # Save the previous section if it has content
                    if current_section_text.strip():
                        sections.append(
                            ExtractedSection(
                                paper_title=paper_info.title,
                                arxiv_id=paper_info.arxiv_id,
                                section_name=current_section_name,
                                page_number=current_page,
                                text=current_section_text.strip(),
                            )
                        )
                    current_section_name = block_text[:100]  # Cap heading length
                    current_section_text = ""
                    current_page = page_number
                else:
                    current_section_text += block_text + "\n"

        # Don't forget the last section
        if current_section_text.strip():
            sections.append(
                ExtractedSection(
                    paper_title=paper_info.title,
                    arxiv_id=paper_info.arxiv_id,
                    section_name=current_section_name,
                    page_number=current_page,
                    text=current_section_text.strip(),
                )
            )

        document.close()
        return sections

    @staticmethod
    def _looks_like_section_heading(text: str) -> bool:
        """Heuristic to detect section headings in academic papers."""
        text = text.strip()

        if len(text) > 100 or len(text) < 2:
            return False
        if "\n" in text:
            return False

        # Numbered sections like "3.2 Multi-Head Attention"
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", text):
            return True

        # Known heading keywords
        heading_keywords = [
            "Abstract",
            "Introduction",
            "Related Work",
            "Background",
            "Method",
            "Approach",
            "Architecture",
            "Experiment",
            "Results",
            "Discussion",
            "Conclusion",
            "Acknowledgment",
            "References",
            "Appendix",
            "Model",
            "Training",
            "Evaluation",
        ]
        for keyword in heading_keywords:
            if text.lower().startswith(keyword.lower()):
                return True

        return False
