"""RogeX System Configuration and Environment Settings."""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent.parent

# Automatically load environment variables from .env if present
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()


@dataclass
class AppConfig:
    """Central configuration class for RogeX RAG application."""

    # Project Metadata
    APP_NAME: str = "RogeX"
    APP_TAGLINE: str = "AI-Powered Document Intelligence & RAG Chatbot"
    APP_VERSION: str = "1.0.0"

    # Filesystem Paths
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    VECTORSTORE_DIR: Path = BASE_DIR / "data" / "vectorstore"
    ASSETS_DIR: Path = BASE_DIR / "assets"

    # LLM Settings (Google Gemini)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    DEFAULT_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    AVAILABLE_GEMINI_MODELS: tuple = (
        "gemini-2.5-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
    )

    # Embedding Settings (Local HuggingFace Sentence Transformers)
    DEFAULT_EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = 384

    # Chunking Defaults
    DEFAULT_CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    DEFAULT_CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MIN_CHUNK_SIZE: int = 200
    MAX_CHUNK_SIZE: int = 3000

    # Vector Retrieval Defaults
    DEFAULT_TOP_K: int = int(os.getenv("TOP_K", "4"))
    MAX_TOP_K: int = 10

    # RAG Strict Grounding Instruction
    STRICT_RAG_NO_ANSWER_MSG: str = (
        "I couldn't find sufficient information about this in your uploaded documents."
    )

    def ensure_directories(self) -> None:
        """Create required runtime directories if they do not exist."""
        for path in [self.DATA_DIR, self.UPLOADS_DIR, self.VECTORSTORE_DIR, self.ASSETS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> AppConfig:
    """Retrieve singleton configuration instance with directories ensured."""
    cfg = AppConfig()
    cfg.ensure_directories()
    return cfg
