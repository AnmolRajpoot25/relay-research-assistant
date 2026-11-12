"""
Indexing module — transforms research papers into searchable passage chunks.

Pipeline: PDF → raw text → overlapping chunks → embeddings → similarity search.
"""
from indexing.paper_parser import PaperParser
from indexing.text_chunker import TextChunker
from indexing.passage_retriever import PassageRetriever
