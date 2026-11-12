"""
Download the three exact arXiv papers required by The Relay pipeline.

This script fetches specific versions of each paper so that every developer
and CI run works against identical source material. We pin versions because
arXiv papers can be revised — pinning eliminates a class of "works on my
machine" issues.

Usage:
    python papers/download_papers.py

The PDFs are saved into the same directory as this script (the `papers/` folder).
"""

import logging
import sys
import time
from pathlib import Path
from typing import NamedTuple

import httpx

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paper definitions ────────────────────────────────────────────────────────

PAPERS_DIRECTORY = Path(__file__).parent.resolve()


class PaperSpec(NamedTuple):
    """Immutable specification for a paper to download."""

    title: str
    url: str
    filename: str


# We use exact versioned URLs so the content never changes between runs.
REQUIRED_PAPERS: list[PaperSpec] = [
    PaperSpec(
        title="Attention Is All You Need (v7)",
        url="https://arxiv.org/pdf/1706.03762v7",
        filename="attention_is_all_you_need_v7.pdf",
    ),
    PaperSpec(
        title="Moshi: a speech-text foundation model for real-time dialogue (v1)",
        url="https://arxiv.org/pdf/2410.00037v1",
        filename="moshi_speech_text_v1.pdf",
    ),
    PaperSpec(
        title="Dynamic Chunking for End-to-End Hierarchical Sequence Modeling (v2)",
        url="https://arxiv.org/pdf/2507.07955v2",
        filename="dynamic_chunking_v2.pdf",
    ),
]

# ── Download logic ───────────────────────────────────────────────────────────

# arXiv occasionally rate-limits bots; a polite user-agent and delay help.
HTTP_HEADERS = {
    "User-Agent": "TheRelay-PaperDownloader/1.0 (academic research pipeline)",
}
DOWNLOAD_TIMEOUT_SECONDS = 120.0
DELAY_BETWEEN_DOWNLOADS_SECONDS = 2.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0


def download_single_paper(
    paper: PaperSpec,
    output_directory: Path,
    client: httpx.Client,
) -> Path:
    """
    Download a single paper PDF, with retries on transient failures.

    Args:
        paper: The paper specification (title, URL, filename).
        output_directory: Directory to save the PDF into.
        client: A reusable httpx client for connection pooling.

    Returns:
        The path to the saved PDF file.

    Raises:
        httpx.HTTPStatusError: If the server returns a non-2xx status after
            all retries are exhausted.
        httpx.TimeoutException: If the download times out after all retries.
    """
    destination = output_directory / paper.filename

    if destination.exists():
        file_size_kb = destination.stat().st_size / 1024
        logger.info(
            "Skipping '%s' — already exists (%.1f KB)",
            paper.title,
            file_size_kb,
        )
        return destination

    last_exception: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Downloading '%s' (attempt %d/%d) from %s",
                paper.title,
                attempt,
                MAX_RETRIES,
                paper.url,
            )

            # Use stream=True to handle large PDFs without loading them
            # entirely into memory (some papers can be 20+ MB).
            with client.stream("GET", paper.url, follow_redirects=True) as response:
                response.raise_for_status()

                with open(destination, "wb") as pdf_file:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        pdf_file.write(chunk)

            file_size_kb = destination.stat().st_size / 1024
            logger.info(
                "Saved '%s' → %s (%.1f KB)",
                paper.title,
                destination.name,
                file_size_kb,
            )
            return destination

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
            last_exception = exc
            logger.warning(
                "Attempt %d failed for '%s': %s",
                attempt,
                paper.title,
                exc,
            )

            # Clean up partial downloads so we don't leave corrupt files.
            if destination.exists():
                destination.unlink()

            if attempt < MAX_RETRIES:
                backoff_delay = RETRY_BACKOFF_SECONDS * attempt
                logger.info("Retrying in %.1f seconds...", backoff_delay)
                time.sleep(backoff_delay)

    # All retries exhausted — surface the last error.
    raise RuntimeError(
        f"Failed to download '{paper.title}' after {MAX_RETRIES} attempts"
    ) from last_exception


def download_all_papers(output_directory: Path | None = None) -> list[Path]:
    """
    Download all required papers into the specified directory.

    Args:
        output_directory: Where to save PDFs. Defaults to the `papers/` folder.

    Returns:
        List of paths to all downloaded PDF files.
    """
    if output_directory is None:
        output_directory = PAPERS_DIRECTORY

    output_directory.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %d papers to %s", len(REQUIRED_PAPERS), output_directory)

    downloaded_paths: list[Path] = []

    with httpx.Client(
        headers=HTTP_HEADERS,
        timeout=httpx.Timeout(DOWNLOAD_TIMEOUT_SECONDS),
    ) as client:
        for index, paper in enumerate(REQUIRED_PAPERS):
            path = download_single_paper(paper, output_directory, client)
            downloaded_paths.append(path)

            # Be polite to arXiv — don't hammer their servers.
            if index < len(REQUIRED_PAPERS) - 1:
                time.sleep(DELAY_BETWEEN_DOWNLOADS_SECONDS)

    logger.info(
        "All %d papers downloaded successfully.", len(downloaded_paths)
    )
    return downloaded_paths


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        paths = download_all_papers()
        print(f"\n[OK] {len(paths)} papers ready in {PAPERS_DIRECTORY}")
        for path in paths:
            print(f"  * {path.name}")
    except Exception as exc:
        logger.error("Paper download failed: %s", exc)
        sys.exit(1)
