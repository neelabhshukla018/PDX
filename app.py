"""PDX – AI-Powered Document Intelligence & RAG Chatbot
"Your Documents. Your AI. Your Answers."

A minimalist, ChatGPT-style conversational AI chatbot with strict document grounding,
built with Streamlit, Google Gemini (official google-genai SDK), LangChain,
Sentence Transformers, and FAISS.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from rag import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TOP_K,
    STRICT_REFUSAL_MESSAGE,
    DocumentTextSplitter,
    EmbeddingManager,
    FAISSVectorStore,
    PDFProcessor,
    ProcessedDocument,
    RAGEngine,
    format_file_size,
    get_embedding_manager,
    truncate_snippet,
)

# ------------------------------------------------------------------------------
# 1. Page Configuration & Environment Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="PDX – AI Document Chatbot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()

ENV_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or os.getenv("GROK_API_KEY")
    or os.getenv("GEMINI_API_KEY", "")
).strip()

ENV_MODEL = (
    os.getenv("GROQ_MODEL")
    or os.getenv("GROK_MODEL")
    or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
).strip()


# ------------------------------------------------------------------------------
# 2. ChatGPT-Inspired Custom CSS Theme
# ------------------------------------------------------------------------------
def inject_custom_css():
    """Inject warm off-white, minimalist, ChatGPT-inspired styling."""
    st.markdown(
        """
        <style>
        /* -------------------------------------------------------------
           RogeX ChatGPT-Inspired Design System
           Background: #FAFAF8 (Warm off-white)
           Sidebar:    #F3F3F0 (Neutral off-white)
           Text:       #202020 (Charcoal)
           Secondary:  #777777 (Muted gray)
           Borders:    #E5E5E0 (Subtle border)
           Accent:     #315C48 (Forest / Sage Green)
           Cards:      #FFFFFF (Clean white)
           ------------------------------------------------------------- */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        :root {
            --bg-main: #FAFAF8;
            --bg-sidebar: #F3F3F0;
            --bg-card: #FFFFFF;
            --text-primary: #202020;
            --text-secondary: #777777;
            --border-color: #E5E5E0;
            --accent-primary: #202020;
            --accent-light: #F0F0EE;
            --accent-hover: #000000;
        }

        /* Global App Styling */
        .stApp {
            background-color: var(--bg-main) !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: var(--text-primary) !important;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        section[data-testid="stSidebar"] div.block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }

        /* Clean Header */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* RogeX Brand Header */
        .rogex-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .rogex-logo {
            width: 38px;
            height: 38px;
            background: var(--accent-primary);
            color: #FFFFFF;
            border-radius: 9px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.15rem;
            letter-spacing: -0.5px;
            box-shadow: 0 2px 4px rgba(49, 92, 72, 0.15);
        }

        .rogex-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.1;
            letter-spacing: -0.5px;
        }

        .rogex-tagline {
            font-size: 0.73rem;
            color: var(--text-secondary);
            font-weight: 500;
            letter-spacing: 0.01em;
            margin-top: 2px;
        }

        /* Action Buttons */
        .stButton button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.88rem !important;
            transition: all 0.15s ease-in-out !important;
            border: 1px solid var(--border-color) !important;
        }

        .stButton button[kind="primary"] {
            background-color: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            color: #FFFFFF !important;
        }

        .stButton button[kind="primary"]:hover {
            background-color: var(--accent-hover) !important;
            border-color: var(--accent-hover) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18) !important;
        }

        .stButton button[kind="secondary"] {
            background-color: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }

        .stButton button[kind="secondary"]:hover {
            border-color: var(--accent-primary) !important;
            color: var(--accent-primary) !important;
            background-color: var(--accent-light) !important;
        }

        /* Document Item Card in Sidebar */
        .doc-item-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }

        .doc-name {
            font-weight: 600;
            color: var(--text-primary);
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            max-width: 160px;
        }

        .doc-meta {
            font-size: 0.72rem;
            color: var(--text-secondary);
            margin-top: 2px;
        }

        /* Status Badge */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid transparent;
        }

        .status-badge.connected {
            background-color: #EBF8F2;
            color: #1B7A4E;
            border-color: #BBEBD3;
        }

        .status-badge.warning {
            background-color: #FFF8E6;
            color: #B45309;
            border-color: #FDE68A;
        }

        .status-badge.error {
            background-color: #FEF2F2;
            color: #B91C1C;
            border-color: #FECACA;
        }

        /* Welcome Screen Hero */
        .chat-welcome-container {
            max-width: 780px;
            margin: 2.5rem auto 1.5rem auto;
            text-align: center;
        }

        .chat-welcome-greeting {
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.03em;
            margin-bottom: 0.4rem;
        }

        .chat-welcome-sub {
            font-size: 1.1rem;
            color: var(--text-secondary);
            font-weight: 400;
            margin-bottom: 2rem;
        }

        /* Suggested Prompt Cards (21st.dev / Aceternity style) */
        .prompt-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px 18px;
            text-align: left;
            cursor: pointer;
            transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
            height: 100%;
        }

        .prompt-card:hover {
            border-color: var(--accent-primary);
            background: #FCFCFB;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }

        .prompt-card-icon {
            font-size: 1.25rem;
            margin-bottom: 8px;
            display: inline-block;
        }

        .prompt-card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 4px;
        }

        .prompt-card-desc {
            font-size: 0.8rem;
            color: var(--text-secondary);
            line-height: 1.35;
        }

        /* Status Banner for Uploaded Docs */
        .doc-status-banner {
            max-width: 780px;
            margin: 0 auto 1.5rem auto;
            background: #FFFFFF;
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent-primary);
            border-radius: 10px;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.88rem;
        }

        /* Chat Message Bubbles */
        .user-message-container {
            display: flex;
            justify-content: flex-end;
            margin: 1.2rem 0;
        }

        .user-message-bubble {
            background-color: var(--accent-light);
            color: var(--text-primary);
            border: 1px solid #E2E2DE;
            border-radius: 16px 16px 4px 16px;
            padding: 12px 18px;
            max-width: 78%;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            word-wrap: break-word;
        }

        .assistant-message-container {
            display: flex;
            gap: 14px;
            margin: 1.4rem 0;
            max-width: 88%;
        }

        .assistant-avatar {
            width: 32px;
            height: 32px;
            background: var(--accent-primary);
            color: #FFFFFF;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 2px;
        }

        .assistant-message-content {
            background: transparent;
            color: var(--text-primary);
            font-size: 0.96rem;
            line-height: 1.65;
            flex-grow: 1;
        }

        /* Source Citation Box */
        .citation-container {
            margin-top: 0.9rem;
            padding: 10px 14px;
            background: #FFFFFF;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            font-size: 0.82rem;
        }

        .citation-header {
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 6px;
            margin-bottom: 8px;
        }

        .citation-pill {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 3px 8px;
            background: #F3F3F0;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 0.76rem;
            font-weight: 600;
            color: var(--accent-primary);
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .citation-snippet {
            font-size: 0.78rem;
            color: var(--text-secondary);
            font-style: italic;
            border-left: 2px solid var(--accent-primary);
            padding-left: 8px;
            margin: 4px 0 10px 0;
            line-height: 1.4;
        }

        /* Streamlit Native Overrides for Clean ChatGPT Feel */
        div[data-testid="stChatMessage"] {
            background-color: transparent !important;
            padding: 1rem 0 !important;
        }

        div[data-testid="stChatInput"] {
            border-radius: 14px !important;
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.04) !important;
            background: #FFFFFF !important;
        }

        div[data-testid="stChatInput"]:focus-within {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.10) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_css()


# ------------------------------------------------------------------------------
# 3. Cached Resource Singletons
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_embedder(model_name: str) -> EmbeddingManager:
    """Cache the SentenceTransformers model in memory."""
    return get_embedding_manager(model_name)


# ------------------------------------------------------------------------------
# 4. Session State Initialization
# ------------------------------------------------------------------------------
def init_session():
    """Initialize robust session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []  # [{role: "user"|"assistant", content: str, sources: list}]

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = {}  # doc_id -> {filename, size, pages, chunks, hash}

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = FAISSVectorStore(dimension=384)

    if "custom_api_key" not in st.session_state:
        st.session_state.custom_api_key = ENV_API_KEY

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = ENV_MODEL

    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K

    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = DEFAULT_CHUNK_SIZE

    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = DEFAULT_CHUNK_OVERLAP

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    if "upload_success_banner" not in st.session_state:
        st.session_state.upload_success_banner = None


init_session()

embedder = load_cached_embedder("sentence-transformers/all-MiniLM-L6-v2")
vector_store: FAISSVectorStore = st.session_state.vector_store

# Instantiate RAG Engine with current session credentials
rag_engine = RAGEngine(
    vector_store=vector_store,
    api_key=st.session_state.custom_api_key or ENV_API_KEY,
    model_name=st.session_state.selected_model or ENV_MODEL,
)


# ------------------------------------------------------------------------------
# 5. Sidebar Component (Branding, PDF Upload, Docs List, Settings)
# ------------------------------------------------------------------------------
with st.sidebar:
    # 5.1 PDX Branding
    st.markdown(
        """
        <div class="rogex-brand">
            <div>
                <div class="rogex-title">PDX</div>
                <div class="rogex-tagline">Your Documents. Your AI. Your Answers.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5.2 Primary Action: New Chat Button
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 5.3 PDF Document Ingestion
    st.markdown(
        "<span style='font-size: 0.78rem; font-weight: 700; color: #777777; text-transform: uppercase; letter-spacing: 0.05em;'>UPLOAD DOCUMENTS</span>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Upload one or multiple PDFs to index into your personal knowledge base.",
    )

    if uploaded_files:
        new_files_processed = 0
        splitter = DocumentTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
        )

        for file_obj in uploaded_files:
            file_bytes = file_obj.getvalue()
            filename = file_obj.name

            # Check if this exact file is already indexed
            is_already_indexed = any(
                doc.get("filename") == filename
                and doc.get("size") == len(file_bytes)
                for doc in st.session_state.uploaded_docs.values()
            )

            if not is_already_indexed:
                with st.spinner(f"Processing and indexing '{filename}'..."):
                    processed_doc: ProcessedDocument = (
                        PDFProcessor.extract_from_bytes(file_bytes, filename)
                    )

                    if processed_doc.error:
                        st.error(f"{filename}: {processed_doc.error}")
                        continue

                    if processed_doc.is_empty:
                        st.warning(
                            f"{filename}: No selectable text detected in PDF."
                        )
                        continue

                    # Chunk and Embed
                    chunks = splitter.split_processed_document(processed_doc)
                    if not chunks:
                        st.warning(
                            f"{filename}: Could not extract readable chunks."
                        )
                        continue

                    indexed_count = vector_store.add_chunks(chunks, embedder)

                    # Save metadata
                    st.session_state.uploaded_docs[processed_doc.doc_id] = {
                        "filename": filename,
                        "size": len(file_bytes),
                        "pages": processed_doc.total_pages,
                        "non_empty_pages": processed_doc.non_empty_pages,
                        "characters": processed_doc.total_characters,
                        "chunks": indexed_count,
                        "doc_id": processed_doc.doc_id,
                    }
                    new_files_processed += 1

        if new_files_processed > 0:
            st.session_state.upload_success_banner = (
                f"Your document{'s are' if new_files_processed > 1 else ' is'} ready. Start asking questions!"
            )
            st.rerun()

    # 5.4 Uploaded Document List & Remove Functionality
    total_docs = len(st.session_state.uploaded_docs)
    total_chunks = vector_store.total_vectors

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"<span style='font-size: 0.78rem; font-weight: 700; color: #777777; text-transform: uppercase; letter-spacing: 0.05em;'>KNOWLEDGE BASE ({total_docs})</span>",
        unsafe_allow_html=True,
    )

    if total_docs > 0:
        docs_to_remove = []
        for doc_id, doc_meta in list(st.session_state.uploaded_docs.items()):
            col_doc, col_del = st.columns([5, 1])
            with col_doc:
                fname = doc_meta["filename"]
                pgs = doc_meta["pages"]
                chks = doc_meta["chunks"]
                sz = format_file_size(doc_meta["size"])
                st.markdown(
                    f"""
                    <div class="doc-item-card">
                        <div style="overflow: hidden;">
                            <div class="doc-name" title="{fname}">{fname}</div>
                            <div class="doc-meta">{pgs} pages • {chks} chunks • {sz}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_del:
                st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
                if st.button("x", key=f"del_{doc_id}", help=f"Remove '{fname}'"):
                    docs_to_remove.append((doc_id, doc_meta["filename"]))

        if docs_to_remove:
            for d_id, d_name in docs_to_remove:
                vector_store.remove_document(d_name, embedder)
                st.session_state.uploaded_docs.pop(d_id, None)
            st.success("Document removed.")
            st.rerun()
    else:
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #888888; padding: 6px 0;">
                No documents uploaded yet.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5.5 Clear Conversation Button
    if len(st.session_state.messages) > 0:
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        if st.button("Clear Conversation", use_container_width=True, type="secondary"):
            st.session_state.messages = []
            st.session_state.pending_query = None
            st.rerun()

    # 5.8 Footer
    st.markdown(
        """
        <div style="font-size: 0.72rem; color: #999999; text-align: center; margin-top: 2rem;">
            PDX • Academic & Enterprise RAG
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 6. Main Chat Area
# ------------------------------------------------------------------------------

# 6.1 Upload Success Notification Banner
if st.session_state.upload_success_banner:
    st.success(f"{st.session_state.upload_success_banner}")
    st.session_state.upload_success_banner = None

# 6.2 Status Banner when documents are indexed
total_docs = len(st.session_state.uploaded_docs)
if total_docs > 0 and len(st.session_state.messages) == 0:
    doc_names_preview = ", ".join(
        [d["filename"] for d in list(st.session_state.uploaded_docs.values())[:3]]
    )
    if total_docs > 3:
        doc_names_preview += f" and {total_docs - 3} more"

    st.markdown(
        f"""
        <div class="doc-status-banner">
            <div>
                <span style="font-weight: 600; color: var(--accent-primary);">Indexed Documents:</span>
                <span style="color: #202020; margin-left: 4px;">{doc_names_preview}</span>
                <span style="color: #777777; font-size: 0.82rem; margin-left: 8px;">({total_chunks} chunks ready)</span>
            </div>
            <div class="status-badge connected" style="font-size: 0.72rem; padding: 2px 8px;">
                Ready
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# 6.3 Empty State / Welcome Screen with 4 Suggested Prompts
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="chat-welcome-container">
            <div class="chat-welcome-greeting">Hello, I'm PDX.</div>
            <div class="chat-welcome-sub">What would you like to know about your documents?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 4 Suggested Prompts Grid (Aceternity / 21st.dev inspired minimal cards)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="prompt-card-title">Summarize my document</div>
                <div class="prompt-card-desc">Get an executive synthesis of core conclusions, thesis, and results.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Summarize my document.", key="sug_btn_1", use_container_width=True):
            st.session_state.pending_query = "Summarize my document."
            st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="prompt-card">
                <div class="prompt-card-title">Explain important concepts</div>
                <div class="prompt-card-desc">Break down technical definitions, frameworks, and methodologies.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Explain the important concepts.", key="sug_btn_2", use_container_width=True):
            st.session_state.pending_query = "Explain the important concepts."
            st.rerun()

    with col2:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="prompt-card-title">Find specific information</div>
                <div class="prompt-card-desc">Retrieve precise metrics, formulas, dates, or citations.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Find specific information.", key="sug_btn_3", use_container_width=True):
            st.session_state.pending_query = "Find specific information."
            st.rerun()

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="prompt-card">
                <div class="prompt-card-title">Generate questions from document</div>
                <div class="prompt-card-desc">Formulate study questions and review prompts to test comprehension.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate questions from my document.", key="sug_btn_4", use_container_width=True):
            st.session_state.pending_query = "Generate questions from my document."
            st.rerun()

    if total_docs == 0:
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        st.info("**Getting Started**: Upload one or more PDF documents in the sidebar, then ask any question.")

st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

# 6.4 Render Conversation History
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])

    if role == "user":
        st.markdown(
            f"""
            <div class="user-message-container">
                <div class="user-message-bubble">
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col_avatar, col_body = st.columns([1, 19])
        with col_avatar:
            st.markdown(
                '<div class="assistant-avatar">P</div>',
                unsafe_allow_html=True,
            )
        with col_body:
            st.markdown(
                f'<div class="assistant-message-content">{content}</div>',
                unsafe_allow_html=True,
            )

            # Render Grounded Sources & Citations
            if sources:
                with st.expander(f"Sources & Document Citations ({len(sources)})", expanded=False):
                    for idx, src in enumerate(sources, start=1):
                        source_doc = src.get("source", "Document")
                        page_num = src.get("page", 1)
                        score = src.get("score", 0.0)
                        snippet = src.get("snippet", "")
                        pct = int(score * 100) if score <= 1.0 else 100

                        st.markdown(
                            f"""
                            <div class="citation-pill">
                                {source_doc} &bull; Page {page_num} &bull; {pct}% Match
                            </div>
                            <div class="citation-snippet">
                                "{truncate_snippet(snippet, 260)}"
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ------------------------------------------------------------------------------
# 7. Chat Input & RAG Execution
# ------------------------------------------------------------------------------

# Check for suggested prompt button click
query_to_submit = None
if st.session_state.pending_query:
    query_to_submit = st.session_state.pending_query
    st.session_state.pending_query = None

user_input = st.chat_input("Ask a question about your uploaded documents...")
final_query = query_to_submit or user_input

if final_query:
    clean_query = final_query.strip()

    # 1. Append User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_query,
            "sources": [],
        }
    )

    # Immediately render user bubble
    st.markdown(
        f"""
        <div class="user-message-container">
            <div class="user-message-bubble">
                {clean_query}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Execute RAG Query
    col_av, col_resp = st.columns([1, 19])
    with col_av:
        st.markdown('<div class="assistant-avatar">P</div>', unsafe_allow_html=True)

    with col_resp:
        # Check if documents are indexed
        if vector_store.is_empty():
            no_docs_reply = (
                "**No documents are currently indexed.**\n\n"
                "Please upload one or more PDF documents in the sidebar so I can answer questions about their content."
            )
            st.markdown(no_docs_reply)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": no_docs_reply,
                    "sources": [],
                }
            )
        else:
            # Check if LLM API Key is configured
            if not rag_engine.is_configured():
                # Perform retrieval to confirm context exists
                relevant_results, citations, direct_ans = rag_engine.retrieve(
                    clean_query, top_k=st.session_state.top_k
                )
                if direct_ans:
                    answer_text = direct_ans
                else:
                    active_p = getattr(rag_engine, "provider", "gemini")
                    p_name = "Groq" if active_p == "groq" else ("Grok" if active_p == "grok" else "Google Gemini")
                    k_name = "GROQ_API_KEY" if active_p == "groq" else ("GROK_API_KEY" if active_p == "grok" else "GEMINI_API_KEY")
                    answer_text = (
                        f"**{p_name} API Key Required**\n\n"
                        f"I successfully retrieved relevant excerpts from your documents, but an API Key "
                        f"is required to synthesize an intelligent answer.\n\n"
                        f"Please configure your `{k_name}` in the local `.env` file."
                    )
                st.markdown(answer_text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer_text,
                        "sources": [
                            {"source": c.source, "page": c.page, "score": c.score, "snippet": c.snippet}
                            for c in citations
                        ],
                    }
                )
            else:
                # Stream real-time response from Gemini
                with st.spinner("Searching document index and synthesizing answer..."):
                    token_stream = rag_engine.query_stream(
                        clean_query, top_k=st.session_state.top_k
                    )
                    full_answer = st.write_stream(token_stream)

                # Retrieve citations for provenance display
                relevant_results, citations, _ = rag_engine.retrieve(
                    clean_query, top_k=st.session_state.top_k
                )

                citations_data = [
                    {
                        "source": c.source,
                        "page": c.page,
                        "score": c.score,
                        "snippet": c.snippet,
                    }
                    for c in citations
                ]

                # Show sources immediately under streaming response
                if citations_data:
                    with st.expander(f"Sources & Document Citations ({len(citations_data)})", expanded=False):
                        for c in citations_data:
                            pct = int(c["score"] * 100) if c["score"] <= 1.0 else 100
                            st.markdown(
                                f"""
                                <div class="citation-pill">
                                    {c['source']} &bull; Page {c['page']} &bull; {pct}% Match
                                </div>
                                <div class="citation-snippet">
                                    "{truncate_snippet(c['snippet'], 260)}"
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": full_answer,
                        "sources": citations_data,
                    }
                )

    st.rerun()
