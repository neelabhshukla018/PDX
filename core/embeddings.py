"""RogeX HuggingFace Sentence Transformers Local Embedding Engine."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manages local Sentence Transformers embedding model with caching and normalization."""

    _instance: Optional["EmbeddingManager"] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: Optional[str] = None

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        """Lazy load or reuse loaded model instance."""
        if EmbeddingManager._model is None or EmbeddingManager._model_name != self.model_name:
            # Load HuggingFace model locally
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
        """Return embedding vector dimension."""
        # all-MiniLM-L6-v2 is 384-dimensional
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate normalized embedding vectors for a list of strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # normalize_embeddings=True allows using Inner Product (IP) as exact Cosine Similarity
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate a single normalized embedding vector for a user query."""
        emb = self.model.encode(
            query,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32).reshape(1, -1)


_cached_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> EmbeddingManager:
    """Singleton getter for embedding manager."""
    global _cached_manager
    if _cached_manager is None or _cached_manager.model_name != model_name:
        _cached_manager = EmbeddingManager(model_name)
    return _cached_manager
