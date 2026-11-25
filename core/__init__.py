"""RogeX Core Pipeline Package - Unified with rag.py."""

from rag import (
    PDFProcessor,
    ProcessedDocument,
    ExtractedPage,
    DocumentTextSplitter,
    DocumentChunk,
    EmbeddingManager,
    get_embedding_manager,
    FAISSVectorStore,
    SearchResult,
    RAGEngine,
    RAGResponse,
    SourceCitation,
)

__all__ = [
    "PDFProcessor",
    "ProcessedDocument",
    "ExtractedPage",
    "DocumentTextSplitter",
    "DocumentChunk",
    "EmbeddingManager",
    "get_embedding_manager",
    "FAISSVectorStore",
    "SearchResult",
    "RAGEngine",
    "RAGResponse",
    "SourceCitation",
]
