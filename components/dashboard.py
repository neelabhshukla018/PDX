"""RogeX Dashboard Component: Metrics, Quick Start Guide, and Knowledge Base Overview."""

import streamlit as st
from config.settings import AppConfig
from utils.helpers import format_file_size


def render_dashboard(config: AppConfig, rag_engine) -> None:
    """Render the main landing dashboard for RogeX."""
    # 1. Hero / Welcome Section
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">Your documents. Your intelligence.</div>
            <div class="hero-subtitle">
                Upload your knowledge and start asking questions. RogeX indexes your PDF documents into a local FAISS vector store and generates strictly grounded answers using Google Gemini.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Key Metrics Cards
    uploaded_docs = st.session_state.get("uploaded_docs", {})
    total_docs = len(uploaded_docs)
    total_chunks = st.session_state.get("total_chunks_indexed", 0)
    total_queries = st.session_state.get("total_queries_asked", 0)
    is_ready = st.session_state.get("vector_store_ready", False) and total_chunks > 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Uploaded Documents</div>
                <div class="stat-value">{total_docs}</div>
                <div class="stat-subtext">Active PDFs in Session</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Indexed Chunks</div>
                <div class="stat-value">{total_chunks}</div>
                <div class="stat-subtext">FAISS Vector Vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Queries Processed</div>
                <div class="stat-value">{total_queries}</div>
                <div class="stat-subtext">Session Interactions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        db_status = "Online" if is_ready else "Empty"
        db_color = "var(--accent-emerald)" if is_ready else "var(--text-muted)"
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Vector Store Status</div>
                <div class="stat-value" style="color: {db_color};">{db_status}</div>
                <div class="stat-subtext">HuggingFace Embeddings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 3. Quick-Start Guide Cards
    st.markdown("### 🚀 Quick-Start Guide")
    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-step-num">1</div>
                <div class="feature-card-title">Upload Documents</div>
                <div class="feature-card-desc">
                    Navigate to <b>Documents</b> and upload one or multiple PDF files (reports, research papers, lecture notes, contracts).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with g_col2:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-step-num">2</div>
                <div class="feature-card-title">Vector Indexing</div>
                <div class="feature-card-desc">
                    RogeX splits text into overlapping chunks, generates 384-d local HuggingFace embeddings, and indexes them in FAISS.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with g_col3:
        st.markdown(
            """
            <div class="feature-card">
                <div class="feature-step-num">3</div>
                <div class="feature-card-title">Strict RAG Chat</div>
                <div class="feature-card-desc">
                    Ask questions naturally. RogeX retrieves relevant excerpts and provides answers strictly grounded in your documents with page citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 4. Suggested Questions Section
    st.markdown("### 💡 Quick Suggested Questions")
    st.caption("Click any question below to immediately ask it in the AI Chat:")

    sample_questions = [
        "What is the main topic and purpose of this document?",
        "Summarize the key findings and conclusions.",
        "What specific methodology or approach was used?",
        "Are there any significant limitations or constraints highlighted?",
    ]

    q_cols = st.columns(2)
    for idx, question in enumerate(sample_questions):
        col = q_cols[idx % 2]
        with col:
            if col.button(f"👉 {question}", key=f"dash_sug_{idx}", use_container_width=True):
                st.session_state.suggested_question = question
                st.session_state.active_view = "chat"
                st.rerun()

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    # 5. Recent Documents Overview Table
    st.markdown("### 📄 Indexed Knowledge Base")
    if total_docs > 0:
        doc_rows = []
        for doc_id, doc in uploaded_docs.items():
            doc_rows.append(
                {
                    "Document Name": doc.get("filename", "Unknown"),
                    "Total Pages": doc.get("pages", 0),
                    "Indexed Chunks": doc.get("chunks", 0),
                    "File Size": format_file_size(doc.get("size", 0)),
                    "Status": "Indexed & Searchable",
                }
            )
        st.dataframe(
            doc_rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No documents have been uploaded yet. Click below to upload your first PDF document."
        )
        if st.button("📁 Go to Document Upload", type="primary"):
            st.session_state.active_view = "documents"
            st.rerun()
