"""RogeX FAISS Vector Store and Similarity Search Engine."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import faiss
import numpy as np

from core.embeddings import EmbeddingManager, get_embedding_manager
from core.text_splitter import DocumentChunk


@dataclass
class SearchResult:
    """Similarity search hit with score and metadata."""

    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    metadata: Dict[str, Any]


class FAISSVectorStore:
    """In-memory and persisted FAISS vector index with rich metadata storage."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Use Inner Product (IP) index. With unit-normalized vectors, IP == Cosine Similarity.
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Return number of vectors currently indexed."""
        return self.index.ntotal if self.index is not None else 0

    def is_empty(self) -> bool:
        """Check if vector index contains zero vectors."""
        return self.total_vectors == 0

    def clear(self) -> None:
        """Reset index and metadata."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> int:
        """Embed document chunks and add them into the FAISS index."""
        if not chunks:
            return 0

        embedder = embedding_manager or get_embedding_manager()
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.embed_texts(texts)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata for retrieval
        for chunk in chunks:
            self.chunks_metadata.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    "char_count": chunk.char_count,
                    "metadata": chunk.metadata,
                }
            )

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 4,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> List[SearchResult]:
        """Perform cosine similarity search against indexed document chunks."""
        if self.is_empty():
            return []

        embedder = embedding_manager or get_embedding_manager()
        query_vector = embedder.embed_query(query)

        k = min(top_k, self.total_vectors)
        distances, indices = self.index.search(query_vector, k)

        results: List[SearchResult] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue

            meta = self.chunks_metadata[idx]
            # Score is cosine similarity in range [-1.0, 1.0], clip to [0, 1] for clean percentage
            score = float(np.clip(dist, 0.0, 1.0))

            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    source=meta["source"],
                    page=meta["page"],
                    score=score,
                    metadata=meta.get("metadata", {}),
                )
            )

        return results

    def remove_document(self, filename: str, embedding_manager: Optional[EmbeddingManager] = None) -> int:
        """Remove all chunks associated with a document and rebuild index."""
        remaining_chunks: List[DocumentChunk] = []

        for m in self.chunks_metadata:
            if m["source"] != filename:
                remaining_chunks.append(
                    DocumentChunk(
                        chunk_id=m["chunk_id"],
                        text=m["text"],
                        source=m["source"],
                        page=m["page"],
                        char_count=m["char_count"],
                        metadata=m["metadata"],
                    )
                )

        removed_count = len(self.chunks_metadata) - len(remaining_chunks)
        self.clear()

        if remaining_chunks:
            self.add_chunks(remaining_chunks, embedding_manager)

        return removed_count

    def save(self, directory: Path) -> bool:
        """Persist FAISS index and metadata to disk."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            index_file = directory / "index.faiss"
            meta_file = directory / "metadata.json"

            faiss.write_index(self.index, str(index_file))
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)

            return True
        except Exception:
            return False

    def load(self, directory: Path) -> bool:
        """Load persisted FAISS index and metadata from disk."""
        try:
            index_file = directory / "index.faiss"
            meta_file = directory / "metadata.json"

            if not (index_file.exists() and meta_file.exists()):
                return False

            self.index = faiss.read_index(str(index_file))
            with open(meta_file, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)

            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Return vector database metrics."""
        distinct_sources = set(m["source"] for m in self.chunks_metadata)
        return {
            "total_vectors": self.total_vectors,
            "total_documents": len(distinct_sources),
            "documents": list(distinct_sources),
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
        }
