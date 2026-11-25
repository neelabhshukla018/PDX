"""RogeX Document Chunking and Text Splitting Engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from core.pdf_processor import ProcessedDocument


@dataclass
class DocumentChunk:
    """Individual document chunk with rich provenance metadata."""

    chunk_id: str
    text: str
    source: str
    page: int
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentTextSplitter:
    """Configurable recursive character text splitter maintaining document and page provenance."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = max(chunk_size, 100)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Split text recursively using hierarchical separators."""
        if len(text) <= self.chunk_size or not separators:
            return [text.strip()] if text.strip() else []

        sep = separators[0]
        sub_seps = separators[1:]

        if sep == "":
            # Character-level fallback split
            chunks = []
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunks.append(text[start:end].strip())
                start += self.chunk_size - self.chunk_overlap
            return [c for c in chunks if c]

        splits = text.split(sep)
        chunks: List[str] = []
        current_chunk = ""

        for s in splits:
            item = s.strip()
            if not item:
                continue

            test_chunk = f"{current_chunk}{sep}{item}".strip() if current_chunk else item

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(item) > self.chunk_size:
                    # Recursive split on deeper separator
                    deeper_chunks = self._split_text_recursive(item, sub_seps)
                    chunks.extend(deeper_chunks)
                    current_chunk = ""
                else:
                    current_chunk = item

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap blending between adjacent chunks if needed
        return chunks

    def split_processed_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """Split pages of a processed document into indexed chunks preserving page numbers."""
        all_chunks: List[DocumentChunk] = []

        for page in doc.pages:
            if not page.text or not page.text.strip():
                continue

            page_splits = self._split_text_recursive(page.text, self.separators)

            for idx, split_text in enumerate(page_splits):
                if not split_text:
                    continue

                chunk_id = f"{doc.doc_id}_p{page.page_number}_c{idx + 1}"
                chunk_meta = {
                    "doc_id": doc.doc_id,
                    "source": doc.filename,
                    "page": page.page_number,
                    "chunk_index": idx + 1,
                    "char_count": len(split_text),
                }

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=split_text,
                        source=doc.filename,
                        page=page.page_number,
                        char_count=len(split_text),
                        metadata=chunk_meta,
                    )
                )

        return all_chunks

    def split_multiple_documents(
        self, docs: List[ProcessedDocument]
    ) -> List[DocumentChunk]:
        """Split a collection of documents."""
        combined_chunks: List[DocumentChunk] = []
        for d in docs:
            combined_chunks.extend(self.split_processed_document(d))
        return combined_chunks
