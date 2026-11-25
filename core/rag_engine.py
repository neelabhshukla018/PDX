"""RogeX RAG Orchestration Engine with Strict Grounding and Gemini LLM."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import get_config
from core.vector_store import FAISSVectorStore, SearchResult


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


class RAGEngine:
    """End-to-End Retrieval Augmented Generation Engine with Strict Document Grounding."""

    STRICT_SYSTEM_INSTRUCTION = (
        "You are RogeX, an advanced academic and professional AI Document Intelligence assistant.\n"
        "Your task is to answer the user's question accurately and objectively, relying EXCLUSIVELY on the provided document excerpts.\n\n"
        "STRICT GROUNDING RULES:\n"
        "1. Answer using ONLY the provided document context excerpts.\n"
        "2. If the user's question cannot be answered directly from the provided excerpts, you MUST reply with this exact phrase:\n"
        '   "I couldn\'t find sufficient information about this in your uploaded documents."\n'
        "3. Do NOT extrapolate, hallucinate, assume, or draw from external knowledge.\n"
        "4. When information is available, write a well-structured, clear, and academic answer.\n"
        "5. Explicitly cite the document name and page number for every key fact mentioned.\n"
        "6. Never fabricate or invent document references."
    )

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.config = get_config()
        self.vector_store = vector_store
        self.api_key = api_key or self.config.GEMINI_API_KEY
        self.model_name = model_name or self.config.DEFAULT_GEMINI_MODEL
        self._genai_client = None
        self._legacy_configured = False

        if self.api_key:
            self._init_client()

    def set_api_key(self, api_key: str) -> None:
        """Update API key dynamically."""
        self.api_key = api_key.strip()
        self._init_client()

    def set_model(self, model_name: str) -> None:
        """Update model dynamically."""
        self.model_name = model_name

    def _init_client(self) -> None:
        """Initialize Google GenAI client (preferring google.genai, fallback to google.generativeai)."""
        if not self.api_key:
            self._genai_client = None
            self._legacy_configured = False
            return

        # Attempt modern google-genai SDK
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            self._legacy_configured = True
            return
        except Exception:
            self._genai_client = None

        # Fallback to google.generativeai
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self._legacy_configured = True
        except Exception:
            self._legacy_configured = False

    def is_configured(self) -> bool:
        """Check if Gemini client has a valid API key configured."""
        return bool(self.api_key and (self._genai_client is not None or self._legacy_configured))

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

    def query(
        self,
        question: str,
        top_k: int = 4,
        similarity_threshold: float = 0.20,
    ) -> RAGResponse:
        """Execute complete RAG pipeline: retrieve context, prompt Gemini, enforce grounding."""
        # 1. Validation: Verify Vector Store status
        if self.vector_store.is_empty():
            return RAGResponse(
                query=question,
                answer="No documents are currently indexed. Please upload one or more PDF documents first.",
                sources=[],
                status="no_context",
                model_name=self.model_name,
            )

        # 2. Retrieval: FAISS similarity search
        search_results = self.vector_store.similarity_search(question, top_k=top_k)

        # Filter by minimum similarity threshold to avoid irrelevant matches
        relevant_results = [r for r in search_results if r.score >= similarity_threshold]

        if not relevant_results:
            return RAGResponse(
                query=question,
                answer=self.config.STRICT_RAG_NO_ANSWER_MSG,
                sources=[],
                status="insufficient_info",
                model_name=self.model_name,
            )

        # Format citations
        citations = [
            SourceCitation(
                source=r.source,
                page=r.page,
                score=r.score,
                snippet=r.text,
            )
            for r in relevant_results
        ]

        # 3. Validation: Verify Gemini API key
        if not self.is_configured():
            return RAGResponse(
                query=question,
                answer=(
                    "⚠️ **Gemini API Key Required**\n\n"
                    "Context was successfully retrieved from your uploaded documents, but a valid Google Gemini API Key is required to generate the answer.\n\n"
                    "Please configure your `GEMINI_API_KEY` in the **Settings** tab or in the `.env` file."
                ),
                sources=citations,
                status="api_key_missing",
                model_name=self.model_name,
                error_message="Missing GEMINI_API_KEY",
            )

        # 4. Prompt Assembly & Gemini Generation
        context_str = self._format_context(relevant_results)

        prompt = (
            f"You are given excerpts from the user's uploaded documents. Answer the question relying ONLY on these excerpts.\n\n"
            f"CONTEXT EXCERPTS:\n{context_str}\n\n"
            f"USER QUESTION: {question}\n\n"
            f"ANSWER:"
        )

        try:
            answer_text = ""

            # Use modern google.genai client if available
            if self._genai_client is not None:
                from google.genai import types
                response = self._genai_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.STRICT_SYSTEM_INSTRUCTION,
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                answer_text = response.text.strip() if response.text else ""
            else:
                # Fallback to legacy google.generativeai
                import google.generativeai as legacy_genai
                model = legacy_genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=self.STRICT_SYSTEM_INSTRUCTION,
                )
                response = model.generate_content(
                    prompt,
                    generation_config=legacy_genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=2048,
                    ),
                )
                answer_text = response.text.strip() if response.text else ""

            if not answer_text:
                answer_text = self.config.STRICT_RAG_NO_ANSWER_MSG

            return RAGResponse(
                query=question,
                answer=answer_text,
                sources=citations,
                status="success",
                model_name=self.model_name,
            )

        except Exception as e:
            error_str = str(e)
            return RAGResponse(
                query=question,
                answer=f"Error communicating with Gemini LLM: {error_str}",
                sources=citations,
                status="error",
                model_name=self.model_name,
                error_message=error_str,
            )
