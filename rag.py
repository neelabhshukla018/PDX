"""PDX – AI-Powered Document Intelligence & RAG Core Engine
Complete Retrieval-Augmented Generation Pipeline with Strict Grounding and Google Gemini.
"""

import hashlib
import io
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import faiss
import numpy as np
import pypdf
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

# System Defaults
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "4"))
STRICT_REFUSAL_MESSAGE = (
    "I couldn't find sufficient information in your uploaded documents."
)


# ==============================================================================
# 1. DATA STRUCTURES
# ==============================================================================


@dataclass
class ExtractedPage:
    """Individual page extraction record with metadata."""

    page_number: int
    text: str
    char_count: int
    source: str


@dataclass
class ProcessedDocument:
    """Complete document extraction summary."""

    doc_id: str
    filename: str
    file_size: int
    total_pages: int
    non_empty_pages: int
    total_characters: int
    pages: List[ExtractedPage] = field(default_factory=list)
    is_empty: bool = False
    error: Optional[str] = None


@dataclass
class DocumentChunk:
    """Individual document chunk with rich provenance metadata."""

    chunk_id: str
    text: str
    source: str
    page: int
    char_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Similarity search hit with score and metadata."""

    chunk_id: str
    text: str
    source: str
    page: int
    score: float
    metadata: Dict[str, Any]


@dataclass
class SourceCitation:
    """Formatted citation for an answer."""

    source: str
    page: int
    score: float
    snippet: str


@dataclass
class RAGResponse:
    """Complete RAG response payload with answer, citations, and status."""

    query: str
    answer: str
    sources: List[SourceCitation] = field(default_factory=list)
    status: str = "success"  # "success", "no_context", "insufficient_info", "error", "api_key_missing"
    model_name: str = ""
    error_message: Optional[str] = None


# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================


def compute_file_hash(data: bytes) -> str:
    """Generate SHA-256 hash of file content for deduplication."""
    return hashlib.sha256(data).hexdigest()[:16]


def sanitize_text(text: str) -> str:
    """Normalize whitespace and strip control characters from extracted text."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple horizontal whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def format_file_size(num_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


def truncate_snippet(text: str, max_chars: int = 240) -> str:
    """Truncate text for preview cards with clean ellipsis."""
    clean = sanitize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0] + "..."


# ==============================================================================
# 3. PDF INGESTION & TEXT EXTRACTION
# ==============================================================================


class PDFProcessor:
    """Robust PDF document processor handling diverse encodings and edge cases."""

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str) -> ProcessedDocument:
        """Extract text and metadata from raw PDF bytes."""
        file_size = len(file_bytes)
        doc_hash = compute_file_hash(file_bytes)

        if file_size == 0:
            return ProcessedDocument(
                doc_id=doc_hash,
                filename=filename,
                file_size=0,
                total_pages=0,
                non_empty_pages=0,
                total_characters=0,
                is_empty=True,
                error="Uploaded file is empty (0 bytes).",
            )

        stream = io.BytesIO(file_bytes)

        try:
            reader = pypdf.PdfReader(stream)

            # Check if PDF is encrypted
            if reader.is_encrypted:
                try:
                    decrypted = reader.decrypt("")
                    if not decrypted:
                        return ProcessedDocument(
                            doc_id=doc_hash,
                            filename=filename,
                            file_size=file_size,
                            total_pages=0,
                            non_empty_pages=0,
                            total_characters=0,
                            error="The PDF is password-protected and cannot be decrypted without a password.",
                        )
                except Exception as e:
                    return ProcessedDocument(
                        doc_id=doc_hash,
                        filename=filename,
                        file_size=file_size,
                        total_pages=0,
                        non_empty_pages=0,
                        total_characters=0,
                        error=f"Could not decrypt PDF: {str(e)}",
                    )

            total_pages = len(reader.pages)
            if total_pages == 0:
                return ProcessedDocument(
                    doc_id=doc_hash,
                    filename=filename,
                    file_size=file_size,
                    total_pages=0,
                    non_empty_pages=0,
                    total_characters=0,
                    is_empty=True,
                    error="PDF document contains zero pages.",
                )

            extracted_pages: List[ExtractedPage] = []
            total_chars = 0
            non_empty_count = 0

            for idx, page in enumerate(reader.pages):
                page_num = idx + 1
                try:
                    raw_text = page.extract_text() or ""
                except Exception:
                    raw_text = ""

                cleaned_text = sanitize_text(raw_text)
                char_count = len(cleaned_text)

                if char_count > 0:
                    non_empty_count += 1
                    total_chars += char_count

                extracted_pages.append(
                    ExtractedPage(
                        page_number=page_num,
                        text=cleaned_text,
                        char_count=char_count,
                        source=filename,
                    )
                )

            is_empty = total_chars == 0
            error_msg = None
            if is_empty:
                error_msg = "No selectable text found in PDF. This document may be scanned, image-only, or empty."

            return ProcessedDocument(
                doc_id=doc_hash,
                filename=filename,
                file_size=file_size,
                total_pages=total_pages,
                non_empty_pages=non_empty_count,
                total_characters=total_chars,
                pages=extracted_pages,
                is_empty=is_empty,
                error=error_msg,
            )

        except Exception as e:
            return ProcessedDocument(
                doc_id=doc_hash,
                filename=filename,
                file_size=file_size,
                total_pages=0,
                non_empty_pages=0,
                total_characters=0,
                error=f"Corrupted or invalid PDF file: {str(e)}",
            )

    @classmethod
    def extract_from_path(cls, file_path: Union[str, Path]) -> ProcessedDocument:
        """Extract text and metadata from a local PDF file path."""
        path = Path(file_path)
        if not path.exists():
            return ProcessedDocument(
                doc_id="missing",
                filename=path.name,
                file_size=0,
                total_pages=0,
                non_empty_pages=0,
                total_characters=0,
                error=f"File not found: {path}",
            )
        file_bytes = path.read_bytes()
        return cls.extract_from_bytes(file_bytes, path.name)


# ==============================================================================
# 4. DOCUMENT CHUNKING (LANGCHAIN TEXT SPLITTER)
# ==============================================================================


class DocumentTextSplitter:
    """Document chunking using LangChain RecursiveCharacterTextSplitter with provenance preservation."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.chunk_size = max(chunk_size, 100)
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            length_function=len,
        )

    def split_processed_document(self, doc: ProcessedDocument) -> List[DocumentChunk]:
        """Split pages of a processed document into indexed chunks preserving page numbers."""
        all_chunks: List[DocumentChunk] = []

        for page in doc.pages:
            if not page.text or not page.text.strip():
                continue

            # Split page text using LangChain's recursive character text splitter
            raw_splits = self.splitter.split_text(page.text)

            for idx, split_text in enumerate(raw_splits):
                cleaned_split = split_text.strip()
                if not cleaned_split:
                    continue

                chunk_id = f"{doc.doc_id}_p{page.page_number}_c{idx + 1}"
                chunk_meta = {
                    "doc_id": doc.doc_id,
                    "source": doc.filename,
                    "page": page.page_number,
                    "chunk_index": idx + 1,
                    "char_count": len(cleaned_split),
                }

                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        text=cleaned_split,
                        source=doc.filename,
                        page=page.page_number,
                        char_count=len(cleaned_split),
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


# ==============================================================================
# 5. EMBEDDINGS ENGINE (SENTENCE TRANSFORMERS)
# ==============================================================================


class EmbeddingManager:
    """Manages HuggingFace Sentence Transformers embedding model with normalization."""

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        # Lazy loading on first use to ensure instant startup

    def _load_model(self) -> None:
        """Lazy load or reuse loaded model instance."""
        if (
            EmbeddingManager._model is None
            or EmbeddingManager._model_name != self.model_name
        ):
            EmbeddingManager._model = SentenceTransformer(self.model_name)
            EmbeddingManager._model_name = self.model_name

    @property
    def model(self) -> SentenceTransformer:
        """Return the active SentenceTransformer instance."""
        if EmbeddingManager._model is None:
            self._load_model()
        return EmbeddingManager._model

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        if EmbeddingManager._model is not None:
            if hasattr(self.model, "get_embedding_dimension"):
                return self.model.get_embedding_dimension()
            return self.model.get_sentence_embedding_dimension()
        return 384

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized embedding vectors for a list of strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a single normalized embedding vector for a query."""
        emb = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32).reshape(1, -1)


_cached_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingManager:
    """Singleton getter for embedding manager."""
    global _cached_embedding_manager
    if (
        _cached_embedding_manager is None
        or _cached_embedding_manager.model_name != model_name
    ):
        _cached_embedding_manager = EmbeddingManager(model_name)
    return _cached_embedding_manager


# ==============================================================================
# 6. FAISS VECTOR DATABASE
# ==============================================================================


class FAISSVectorStore:
    """In-memory and persisted FAISS vector index with rich metadata storage."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        # Inner Product on L2-normalized embeddings equals Cosine Similarity
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

        # Add vectors to FAISS index
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
        top_k: int = DEFAULT_TOP_K,
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
            # Score is cosine similarity clipped to [0.0, 1.0]
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

    def remove_document(
        self, filename: str, embedding_manager: Optional[EmbeddingManager] = None
    ) -> int:
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
        distinct_sources = sorted(list(set(m["source"] for m in self.chunks_metadata)))
        return {
            "total_vectors": self.total_vectors,
            "total_documents": len(distinct_sources),
            "documents": distinct_sources,
            "dimension": self.dimension,
            "index_type": "FAISS IndexFlatIP (Cosine Similarity)",
        }


# ==============================================================================
# 7. RAG ORCHESTRATION ENGINE & GEMINI LLM
# ==============================================================================


class RAGEngine:
    """End-to-End Retrieval Augmented Generation Engine with Strict Grounding and Google Gemini."""

    STRICT_SYSTEM_INSTRUCTION = (
        "You are PDX, an advanced academic and professional AI Document Intelligence assistant.\n"
        "Your task is to answer the user's question accurately, conversationally, and objectively, relying EXCLUSIVELY on the provided document excerpts.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Answer using ONLY the provided document context excerpts.\n"
        "2. If the user's question cannot be answered directly from the provided excerpts, you MUST reply with this exact phrase:\n"
        f'   "{STRICT_REFUSAL_MESSAGE}"\n'
        "3. Do NOT extrapolate, hallucinate, assume, or draw from external knowledge.\n"
        "4. When information is available, write a well-structured, clear, and comprehensive answer using Markdown headings, bullet points, and code blocks where helpful.\n"
        "5. Explicitly cite the document name and page number for every key fact mentioned.\n"
        "6. Never fabricate or invent document references."
    )

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.vector_store = vector_store

        # If api_key is explicitly passed as empty string, treat as unconfigured
        if api_key == "":
            self.api_key = ""
            self.gemini_api_key = ""
            self.provider = "gemini"
            self.endpoint_url = None
            self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            self._genai_client = None
            self._last_error = None
            return

        # Multi-provider support: GroqCloud (gsk_...), xAI Grok (xai-...), or Google Gemini
        groq_candidate = (
            os.getenv("GROQ_API_KEY")
            or os.getenv("GROK_API_KEY")
            or os.getenv("XAI_API_KEY", "")
        ).strip()
        self.gemini_api_key = (
            api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip()

        if groq_candidate:
            self.api_key = groq_candidate
            if groq_candidate.startswith("gsk_"):
                self.provider = "groq"
                self.endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
                user_m = (
                    model_name
                    or os.getenv("GROQ_MODEL")
                    or os.getenv("GROK_MODEL", "")
                )
                if not user_m or "grok" in user_m.lower() or "gemini" in user_m.lower():
                    self.model_name = "qwen/qwen3.8-27b"
                else:
                    self.model_name = user_m
            else:
                self.provider = "grok"
                self.endpoint_url = "https://api.x.ai/v1/chat/completions"
                self.model_name = (
                    model_name
                    or os.getenv("GROK_MODEL")
                    or os.getenv("GROQ_MODEL", "grok-beta")
                )
        else:
            self.provider = "gemini"
            self.api_key = self.gemini_api_key
            self.endpoint_url = None
            self.model_name = (
                model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
            )

        self._genai_client = None
        self._last_error = None

        if self.provider == "gemini" and self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str) -> None:
        """Update API key dynamically."""
        clean_key = (api_key or "").strip()
        self.api_key = clean_key
        if clean_key.startswith("gsk_"):
            self.provider = "groq"
            self.endpoint_url = "https://api.groq.com/openai/v1/chat/completions"
            self.model_name = "qwen/qwen3.8-27b"
        elif clean_key.startswith("xai-"):
            self.provider = "grok"
            self.endpoint_url = "https://api.x.ai/v1/chat/completions"
            self.model_name = "grok-beta"
        else:
            self.provider = "gemini"
            self.gemini_api_key = clean_key
            self.endpoint_url = None
            self._init_client()
        self._last_error = None

    def set_model(self, model_name: str) -> None:
        """Update model dynamically."""
        self.model_name = model_name.strip()

    def _init_client(self) -> None:
        """Initialize Google GenAI client using official google-genai SDK."""
        if not self.api_key:
            self._genai_client = None
            return

        try:
            from google import genai

            self._genai_client = genai.Client(api_key=self.api_key)
            self._last_error = None
        except Exception as e:
            self._genai_client = None
            self._last_error = str(e)

    def is_configured(self) -> bool:
        """Check if active LLM client is initialized."""
        if self.provider in ("groq", "grok"):
            return bool(self.api_key)
        return bool(self.api_key and self._genai_client is not None)

    def test_connection(self) -> Tuple[bool, str]:
        """Test API key connection with a lightweight ping call."""
        if self.provider in ("groq", "grok"):
            if not self.api_key:
                return False, f"{self.provider.capitalize()} API key is not configured in .env."
            try:
                import requests

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                }
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 5,
                }
                resp = requests.post(
                    self.endpoint_url, headers=headers, json=payload, timeout=10
                )
                if resp.status_code == 200:
                    return (
                        True,
                        f"Successfully connected to {self.provider.capitalize()} ({self.model_name})",
                    )
                return (
                    False,
                    f"{self.provider.capitalize()} connection failed (HTTP {resp.status_code}): {resp.text}",
                )
            except Exception as e:
                return False, f"{self.provider.capitalize()} connection error: {str(e)}"

        if not self.api_key:
            return False, "GEMINI_API_KEY is not configured in .env."

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            resp = client.models.generate_content(
                model=self.model_name,
                contents="ping",
                config=types.GenerateContentConfig(
                    max_output_tokens=5,
                    temperature=0.0,
                ),
            )
            return True, f"Successfully connected to Gemini ({self.model_name})"
        except Exception as e:
            err = str(e)
            if "403" in err or "Forbidden" in err:
                return (
                    False,
                    "Access forbidden: API key invalid or lacks permission for this model.",
                )
            elif "429" in err or "ResourceExhausted" in err:
                return False, "Rate limit or quota exceeded on Google Gemini."
            elif "400" in err or "INVALID_ARGUMENT" in err:
                return False, f"Invalid request or model name: {self.model_name}"
            return False, f"Connection failed: {err}"

    def _query_openai_compatible(self, prompt: str) -> str:
        """Non-streaming query to OpenAI-compatible API (Groq or xAI) with failover."""
        import requests

        candidate_models = [self.model_name]
        if self.provider == "groq":
            for m in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                if m not in candidate_models:
                    candidate_models.append(m)

        last_error = None
        for model in candidate_models:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.STRICT_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "stream": False,
            }
            try:
                resp = requests.post(
                    self.endpoint_url, headers=headers, json=payload, timeout=25
                )
                if resp.status_code != 200:
                    last_error = Exception(f"HTTP {resp.status_code}: {resp.text}")
                    continue
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content:
                    return content
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error
        return ""

    def _query_openai_compatible_stream(
        self, prompt: str
    ) -> Generator[str, None, None]:
        """Streaming token generator from OpenAI-compatible API (Groq or xAI) with failover."""
        import json
        import requests

        candidate_models = [self.model_name]
        if self.provider == "groq":
            for m in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
                if m not in candidate_models:
                    candidate_models.append(m)

        last_error = None
        for model in candidate_models:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": self.STRICT_SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "stream": True,
            }
            try:
                resp = requests.post(
                    self.endpoint_url, headers=headers, json=payload, stream=True, timeout=25
                )
                if resp.status_code != 200:
                    last_error = Exception(f"HTTP {resp.status_code}: {resp.text}")
                    continue

                yielded_any = False
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                                    yielded_any = True
                            except Exception:
                                continue
                if yielded_any:
                    return
            except Exception as e:
                last_error = e
                continue

        if last_error:
            raise last_error

    def _format_context(self, search_results: List[SearchResult]) -> str:
        """Format retrieved chunks into clean, numbered context blocks."""
        blocks: List[str] = []
        for idx, res in enumerate(search_results, start=1):
            block = (
                f"--- DOCUMENT EXCERPT [{idx}] ---\n"
                f"Document: {res.source}\n"
                f"Page: {res.page}\n"
                f"Relevance Score: {res.score:.2f}\n"
                f"Content:\n{res.text.strip()}\n"
                f"--------------------------------"
            )
            blocks.append(block)
        return "\n\n".join(blocks)

    def _build_prompt(self, question: str, context_str: str) -> str:
        """Construct grounded prompt."""
        return (
            f"You are given excerpts from the user's uploaded documents. Answer the question relying ONLY on these excerpts.\n\n"
            f"CONTEXT EXCERPTS:\n{context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"ANSWER:"
        )

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = 0.0,
    ) -> Tuple[List[SearchResult], List[SourceCitation], Optional[str]]:
        """Retrieve relevant chunks and citations for a question with summary query support."""
        if self.vector_store.is_empty():
            return (
                [],
                [],
                "No documents are currently indexed. Please upload one or more PDF documents first.",
            )

        # Detect summary, overview, or conceptual exploration queries
        summary_keywords = (
            "summarize",
            "summary",
            "overview",
            "main concept",
            "important concept",
            "key point",
            "generate questions",
            "about the document",
            "explain the document",
            "explain document",
            "what is this",
            "tell me about",
        )
        is_summary = any(kw in question.lower() for kw in summary_keywords)

        if is_summary:
            # Query FAISS with semantic expansion for document overview
            search_results = self.vector_store.similarity_search(
                f"{question} summary overview introduction abstract conclusion main topics concepts",
                top_k=top_k,
            )
            # Ensure early introductory chunks from documents are represented
            if len(search_results) < top_k and self.vector_store.chunks_metadata:
                seen_ids = {r.chunk_id for r in search_results}
                for meta in self.vector_store.chunks_metadata:
                    if meta["chunk_id"] not in seen_ids:
                        search_results.append(
                            SearchResult(
                                chunk_id=meta["chunk_id"],
                                text=meta["text"],
                                source=meta["source"],
                                page=meta["page"],
                                score=0.85,
                                metadata=meta.get("metadata", {}),
                            )
                        )
                        seen_ids.add(meta["chunk_id"])
                        if len(search_results) >= top_k:
                            break
        else:
            search_results = self.vector_store.similarity_search(question, top_k=top_k)

        # Filter by threshold if explicitly set > 0, otherwise retain all retrieved top_k chunks
        if similarity_threshold > 0.0:
            relevant_results = [
                r for r in search_results if r.score >= similarity_threshold
            ]
        else:
            relevant_results = search_results

        if not relevant_results:
            return (
                [],
                [],
                STRICT_REFUSAL_MESSAGE,
            )

        citations = [
            SourceCitation(
                source=r.source,
                page=r.page,
                score=r.score,
                snippet=r.text,
            )
            for r in relevant_results
        ]

        return relevant_results, citations, None

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = 0.0,
    ) -> RAGResponse:
        """Execute complete RAG pipeline: retrieve context, prompt Gemini, enforce grounding."""
        relevant_results, citations, direct_answer = self.retrieve(
            question, top_k=top_k, similarity_threshold=similarity_threshold
        )

        if direct_answer:
            status = "no_context" if self.vector_store.is_empty() else "insufficient_info"
            return RAGResponse(
                query=question,
                answer=direct_answer,
                sources=citations,
                status=status,
                model_name=self.model_name,
            )

        # Check API key configuration
        if not self.is_configured():
            provider_name = "Grok" if self.provider == "grok" else "Google Gemini"
            key_name = "GROK_API_KEY" if self.provider == "grok" else "GEMINI_API_KEY"
            return RAGResponse(
                query=question,
                answer=(
                    f"**{provider_name} API Key Required**\n\n"
                    f"Context was successfully retrieved from your uploaded documents, but a valid API Key is required to synthesize the answer.\n\n"
                    f"Please configure your `{key_name}` in the local `.env` file."
                ),
                sources=citations,
                status="api_key_missing",
                model_name=self.model_name,
                error_message=f"Missing or uninitialized {key_name}",
            )

        context_str = self._format_context(relevant_results)
        prompt = self._build_prompt(question, context_str)

        # OpenAI-compatible execution branch (Groq or xAI Grok)
        if self.provider in ("groq", "grok"):
            try:
                answer_text = self._query_openai_compatible(prompt)
                if not answer_text:
                    answer_text = STRICT_REFUSAL_MESSAGE
                return RAGResponse(
                    query=question,
                    answer=answer_text,
                    sources=citations,
                    status="success",
                    model_name=self.model_name,
                )
            except Exception as e:
                provider_title = self.provider.capitalize()
                return RAGResponse(
                    query=question,
                    answer=f"**{provider_title} Generation Error**\n\n{str(e)}",
                    sources=citations,
                    status="error",
                    model_name=self.model_name,
                    error_message=str(e),
                )

        # Automatic failover candidates for high-traffic spikes (HTTP 503 / 429)
        candidate_models = [self.model_name]
        for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model_candidate in candidate_models:
            for attempt in range(2):
                try:
                    from google.genai import types

                    response = self._genai_client.models.generate_content(
                        model=model_candidate,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=self.STRICT_SYSTEM_INSTRUCTION,
                            temperature=0.1,
                            max_output_tokens=2048,
                        ),
                    )
                    answer_text = response.text.strip() if response.text else ""
                    if not answer_text:
                        answer_text = STRICT_REFUSAL_MESSAGE

                    return RAGResponse(
                        query=question,
                        answer=answer_text,
                        sources=citations,
                        status="success",
                        model_name=model_candidate,
                    )
                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if (
                        "503" in err_str
                        or "UNAVAILABLE" in err_str
                        or "429" in err_str
                        or "ResourceExhausted" in err_str
                    ):
                        import time
                        time.sleep(1.0)
                        continue
                    else:
                        break

        error_msg = self._humanize_error(last_error or Exception("Synthesis failed"))
        return RAGResponse(
            query=question,
            answer=f"**Gemini Generation Error**\n\n{error_msg}",
            sources=citations,
            status="error",
            model_name=self.model_name,
            error_message=str(last_error),
        )

    def query_stream(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        similarity_threshold: float = 0.0,
    ) -> Generator[str, None, List[SourceCitation]]:
        """Streaming version of RAG query yielding token chunks for real-time ChatGPT feel."""
        relevant_results, citations, direct_answer = self.retrieve(
            question, top_k=top_k, similarity_threshold=similarity_threshold
        )

        if direct_answer:
            yield direct_answer
            return citations

        if not self.is_configured():
            provider_name = "Grok" if self.provider == "grok" else "Google Gemini"
            key_name = "GROK_API_KEY" if self.provider == "grok" else "GEMINI_API_KEY"
            yield (
                f"**{provider_name} API Key Required**\n\n"
                f"Context was successfully retrieved from your uploaded documents, but a valid API Key is required to synthesize the answer.\n\n"
                f"Please configure your `{key_name}` in the local `.env` file."
            )
            return citations

        context_str = self._format_context(relevant_results)
        prompt = self._build_prompt(question, context_str)

        # OpenAI-compatible streaming branch (Groq or xAI Grok)
        if self.provider in ("groq", "grok"):
            try:
                has_text = False
                for chunk in self._query_openai_compatible_stream(prompt):
                    has_text = True
                    yield chunk
                if not has_text:
                    yield STRICT_REFUSAL_MESSAGE
                return citations
            except Exception as e:
                provider_title = self.provider.capitalize()
                yield f"**{provider_title} Generation Error**\n\n{str(e)}"
                return citations

        # Automatic failover candidates for high-traffic spikes (HTTP 503 / 429)
        candidate_models = [self.model_name]
        for fallback in ["gemini-2.0-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        last_error = None
        for model_candidate in candidate_models:
            for attempt in range(2):
                try:
                    from google.genai import types

                    stream = self._genai_client.models.generate_content_stream(
                        model=model_candidate,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=self.STRICT_SYSTEM_INSTRUCTION,
                            temperature=0.1,
                            max_output_tokens=2048,
                        ),
                    )

                    has_text = False
                    for chunk in stream:
                        if chunk.text:
                            has_text = True
                            yield chunk.text

                    if not has_text:
                        yield STRICT_REFUSAL_MESSAGE

                    return citations

                except Exception as e:
                    last_error = e
                    err_str = str(e)
                    if (
                        "503" in err_str
                        or "UNAVAILABLE" in err_str
                        or "429" in err_str
                        or "ResourceExhausted" in err_str
                    ):
                        import time
                        time.sleep(1.0)
                        continue
                    else:
                        break

        yield f"**Gemini Generation Error**\n\n{self._humanize_error(last_error or Exception('Generation failed'))}"
        return citations

    def _humanize_error(self, e: Exception) -> str:
        """Format raw exceptions into clear, friendly explanations."""
        err_str = str(e)
        if "403" in err_str or "Forbidden" in err_str:
            return (
                "Access forbidden (HTTP 403). Your Google Gemini API Key may be invalid, expired, "
                "or restricted from accessing the specified model. Please verify your key at Google AI Studio."
            )
        elif "429" in err_str or "ResourceExhausted" in err_str:
            return (
                "Google Gemini rate limit or quota exceeded (HTTP 429). Please wait a moment and try again, "
                "or check your AI Studio quota limits."
            )
        elif "400" in err_str or "INVALID_ARGUMENT" in err_str:
            return (
                f"Invalid request configuration (HTTP 400). Please check that model '{self.model_name}' "
                "is supported in your region."
            )
        elif "connection" in err_str.lower() or "timeout" in err_str.lower():
            return "Network connection error while contacting Google Gemini. Please check your internet connection."
        return f"An unexpected error occurred: {err_str}"


# Self-test block when executed directly
if __name__ == "__main__":
    print("PDX RAG Core Engine Loaded Successfully.")
    mgr = get_embedding_manager()
    print(f"Embedding Model: {mgr.model_name} (dim={mgr.dimension})")
    vstore = FAISSVectorStore(dimension=mgr.dimension)
    engine = RAGEngine(vector_store=vstore)
    print(f"RAG Engine Initialized (Model={engine.model_name}, Configured={engine.is_configured()})")
