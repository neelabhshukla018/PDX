"""RogeX PDF Ingestion and Text Extraction Engine."""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import pypdf

from utils.helpers import compute_file_hash, sanitize_text


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


class PDFProcessor:
    """Robust PDF document processor handling diverse formats, encodings, and edge cases."""

    @staticmethod
    def extract_from_bytes(file_bytes: bytes, filename: str) -> ProcessedDocument:
        """Extract text and metadata from raw PDF bytes (e.g. from Streamlit file uploader)."""
        file_size = len(file_bytes)
        doc_hash = compute_file_hash(file_bytes)
        stream = io.BytesIO(file_bytes)

        try:
            reader = pypdf.PdfReader(stream)

            # Check if PDF is encrypted
            if reader.is_encrypted:
                try:
                    # Attempt decrypting with empty password (common for readable PDFs)
                    decrypted = reader.decrypt("")
                    if not decrypted:
                        return ProcessedDocument(
                            doc_id=doc_hash,
                            filename=filename,
                            file_size=file_size,
                            total_pages=0,
                            non_empty_pages=0,
                            total_characters=0,
                            error="The PDF is password-protected and cannot be read without a password.",
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

            is_empty = (total_chars == 0)
            error_msg = None
            if is_empty:
                error_msg = "No selectable text found in PDF. This document may be scanned or image-only."

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
