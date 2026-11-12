"""
Run Pipeline CLI — Command-line interface for The Relay.

Senior Engineering Perspective:
Provides direct engineer ergonomics for:
1. Re-indexing papers under distinct chunking parameters (small, medium, large).
2. Querying individual questions with full telemetry (TTFT, step breakdown, citations).
3. Interactive terminal session with live quota inspection.
"""

import argparse
import logging
import sys
from pathlib import Path

from config import INDEX_DIRECTORY, PAPERS_DIRECTORY, settings
from indexing.paper_parser import PaperParser
from indexing.passage_retriever import PassageRetriever
from indexing.text_chunker import CHUNKING_EXPERIMENTS, TextChunker
from models.model_client import ModelClient
from pipeline.orchestrator import PipelineOrchestrator


def configure_logging():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def initialize_or_load_index(chunk_setting: str = "medium") -> PassageRetriever:
    retriever = PassageRetriever(index_directory=INDEX_DIRECTORY)
    if retriever.load_index():
        return retriever

    print(f"\n[!] No existing index found in {INDEX_DIRECTORY}. Building now with '{chunk_setting}' setting...")
    return build_index(chunk_setting)


def build_index(chunk_setting: str = "medium") -> PassageRetriever:
    parser = PaperParser(papers_directory=PAPERS_DIRECTORY)
    sections = parser.parse_all_papers()

    if not sections:
        print(f"\n[X] Error: No sections parsed! Ensure PDFs exist in {PAPERS_DIRECTORY}.")
        print("    Download them by running: python papers/download_papers.py")
        sys.exit(1)

    config = CHUNKING_EXPERIMENTS.get(chunk_setting, CHUNKING_EXPERIMENTS["medium"])
    chunker = TextChunker(config=config)
    chunks = chunker.chunk_sections(sections)

    retriever = PassageRetriever(index_directory=INDEX_DIRECTORY)
    retriever.build_index(chunks)
    print(f"[OK] Successfully built index containing {len(chunks)} chunks using setting '{chunk_setting}'.")
    return retriever


def execute_single_query(question: str, chunk_setting: str = "medium"):
    retriever = initialize_or_load_index(chunk_setting)
    model_client = ModelClient()
    pipeline = PipelineOrchestrator(model_client=model_client, passage_retriever=retriever)

    print("\n" + "=" * 70)
    print(f"QUERY: {question}")
    print("=" * 70 + "\n")

    execution = pipeline.process_question(question=question, retrieval_setting=chunk_setting)

    print(execution.answer)
    print("\n" + "-" * 70)
    print(f"Status:               {execution.status.upper()}")
    print(f"Time to First Token:  {execution.first_token_ms:.1f} ms")
    print(f"Total Latency:        {execution.total_ms:.1f} ms")
    print(f"Tokens Consumed:      {execution.total_tokens_in} prompt + {execution.total_tokens_out} completion = {execution.total_tokens}")
    print(f"Quality Loops:        {execution.quality_loops_executed}")
    print(f"Citations Extracted:  {execution.citations_count}")
    print(f"Model Employed:       {execution.model_used}")
    print("-" * 70 + "\n")


def run_interactive_shell():
    print("\n" + "=" * 70)
    print("   The Relay — Interactive Research Paper Assistant")
    print("   Type your technical question or type 'status' / 'exit'.")
    print("=" * 70 + "\n")

    retriever = initialize_or_load_index("medium")
    model_client = ModelClient()
    pipeline = PipelineOrchestrator(model_client=model_client, passage_retriever=retriever)

    while True:
        try:
            user_input = input("\nInquiry > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting session.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Session terminated.")
            break
        if user_input.lower() == "status":
            import json
            print(json.dumps(model_client.get_metrics_summary(), indent=2))
            continue

        result = pipeline.process_question(user_input)
        print("\n" + result.answer + "\n")
        print(f"[{result.model_used} | {result.total_ms:.0f}ms | {result.total_tokens} tokens]")


def main():
    parser = argparse.ArgumentParser(description="The Relay — Research Assistant CLI")
    parser.add_argument("question", nargs="?", help="Direct question to answer")
    parser.add_argument("--build-index", action="store_true", help="Parse and index PDFs")
    parser.add_argument(
        "--chunk-setting",
        choices=["small", "medium", "large"],
        default="medium",
        help="Chunk size preset (small=256, medium=512, large=1024)",
    )
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive terminal mode")

    args = parser.parse_args()
    configure_logging()

    if args.build_index:
        build_index(args.chunk_setting)
        return

    if args.interactive:
        run_interactive_shell()
        return

    if args.question:
        execute_single_query(args.question, args.chunk_setting)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
