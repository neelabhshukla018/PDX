"""RogeX Document Manager Component: Multi-PDF Ingestion, Chunking, and Indexing."""

import streamlit as st
from config.settings import AppConfig
from core.pdf_processor import PDFProcessor
from core.text_splitter import DocumentTextSplitter
from core.vector_store import FAISSVectorStore
from utils.helpers import format_file_size, compute_file_hash


def render_document_manager(
    config: AppConfig,
    vector_store: FAISSVectorStore,
    rag_engine,
) -> None:
    """Render the document upload, extraction, and management interface."""
    st.markdown("## 📁 Document Management")
    st.markdown(
        "Upload one or multiple PDF documents. RogeX parses each document, segments it into chunks with rich page metadata, and builds a local FAISS vector index."
    )

    # 1. Multi-PDF Upload Section
    st.markdown("### 📤 Upload PDF Documents")
    uploaded_files = st.file_uploader(
        "Select PDF files to ingest",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload standard PDF files. Scanned PDFs without text layers will be detected and flagged.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected for processing.**")

        # Process and Index Button
        if st.button("⚡ Process & Index Documents", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Starting document ingestion...")
            status_text = st.empty()

            chunk_size = st.session_state.get("chunk_size", config.DEFAULT_CHUNK_SIZE)
            chunk_overlap = st.session_state.get("chunk_overlap", config.DEFAULT_CHUNK_OVERLAP)
            splitter = DocumentTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            total_files = len(uploaded_files)
            new_docs_indexed = 0
            new_chunks_count = 0
            errors: list = []

            for i, file_obj in enumerate(uploaded_files):
                pct = int((i / total_files) * 80)
                progress_bar.progress(pct, text=f"Processing [{i + 1}/{total_files}]: {file_obj.name}")

                file_bytes = file_obj.read()
                file_hash = compute_file_hash(file_bytes)

                # Step 1: Extract Text
                processed_doc = PDFProcessor.extract_from_bytes(file_bytes, file_obj.name)

                if processed_doc.error:
                    errors.append(f"**{file_obj.name}**: {processed_doc.error}")
                    if processed_doc.is_empty:
                        # Continue processing other documents
                        continue

                # Step 2: Split Text into Provenance Chunks
                status_text.text(f"Chunking {file_obj.name} with size {chunk_size} and overlap {chunk_overlap}...")
                chunks = splitter.split_processed_document(processed_doc)

                if not chunks:
                    errors.append(f"**{file_obj.name}**: No valid text chunks could be formed.")
                    continue

                # Step 3: Embed & Index in FAISS
                status_text.text(f"Generating HuggingFace embeddings & indexing {len(chunks)} chunks...")
                # If document was already indexed under same hash, remove previous version
                if file_obj.name in [d.get("filename") for d in st.session_state.uploaded_docs.values()]:
                    vector_store.remove_document(file_obj.name)

                vector_store.add_chunks(chunks)

                # Persist document metadata in session
                st.session_state.uploaded_docs[file_hash] = {
                    "filename": file_obj.name,
                    "size": len(file_bytes),
                    "pages": processed_doc.total_pages,
                    "non_empty_pages": processed_doc.non_empty_pages,
                    "characters": processed_doc.total_characters,
                    "chunks": len(chunks),
                    "hash": file_hash,
                }

                new_docs_indexed += 1
                new_chunks_count += len(chunks)

            # Step 4: Finalize & Save
            progress_bar.progress(100, text="Ingestion complete!")
            status_text.empty()

            # Recalculate total chunks
            st.session_state.total_chunks_indexed = vector_store.total_vectors
            st.session_state.vector_store_ready = vector_store.total_vectors > 0

            # Persist vector store to disk
            vector_store.save(config.VECTORSTORE_DIR)

            if new_docs_indexed > 0:
                st.success(
                    f"🎉 Successfully processed {new_docs_indexed} document(s) into {new_chunks_count} indexed chunks! The knowledge base is ready for questions."
                )

            if errors:
                for err in errors:
                    st.warning(err)

            st.rerun()

    st.markdown("---")

    # 2. Currently Managed Documents Section
    st.markdown("### 📚 Active Documents in Knowledge Base")
    uploaded_docs = st.session_state.get("uploaded_docs", {})

    if not uploaded_docs:
        st.info("No documents are currently indexed. Upload PDF files above to populate your knowledge base.")
        return

    # List each active document with detailed stats and remove capability
    for doc_hash, doc in list(uploaded_docs.items()):
        filename = doc.get("filename", "Unknown Document")
        size = format_file_size(doc.get("size", 0))
        pages = doc.get("pages", 0)
        chunks = doc.get("chunks", 0)
        chars = doc.get("characters", 0)

        card_col, action_col = st.columns([5, 1])

        with card_col:
            st.markdown(
                f"""
                <div class="doc-item-card">
                    <div>
                        <div class="doc-name">📄 {filename}</div>
                        <div class="doc-meta">
                            <span><b>{pages}</b> Pages</span>
                            <span>•</span>
                            <span><b>{chunks}</b> Chunks</span>
                            <span>•</span>
                            <span><b>{chars:,}</b> Characters</span>
                            <span>•</span>
                            <span>{size}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with action_col:
            # Document Removal Functionality
            if st.button("🗑️ Remove", key=f"del_{doc_hash}", use_container_width=True):
                # Remove from FAISS index
                vector_store.remove_document(filename)
                # Remove from session
                del st.session_state.uploaded_docs[doc_hash]
                # Update vector count
                st.session_state.total_chunks_indexed = vector_store.total_vectors
                st.session_state.vector_store_ready = vector_store.total_vectors > 0
                # Persist updated vector store
                vector_store.save(config.VECTORSTORE_DIR)
                st.success(f"Removed '{filename}' from the knowledge base.")
                st.rerun()
