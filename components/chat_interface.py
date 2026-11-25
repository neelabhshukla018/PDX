"""RogeX Conversational Chat Interface with Verified Source Citations."""

import streamlit as st
from config.settings import AppConfig
from core.rag_engine import RAGEngine
from core.vector_store import FAISSVectorStore
from utils.helpers import truncate_snippet


def render_chat_interface(
    config: AppConfig,
    vector_store: FAISSVectorStore,
    rag_engine: RAGEngine,
) -> None:
    """Render the AI Chat interface with conversation history and grounded source references."""
    # Top Action Bar
    header_col, actions_col = st.columns([4, 2])

    with header_col:
        st.markdown("## 💬 AI Document Intelligence Chat")
        st.caption(
            "Ask questions about your uploaded documents. Answers are strictly synthesized from retrieved context excerpts."
        )

    with actions_col:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        col_new, col_clear = st.columns(2)
        with col_new:
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

    # Empty State Notice if no documents indexed
    if vector_store.is_empty():
        st.warning(
            "⚠️ **No documents are currently indexed.**\n\n"
            "Please go to the **📁 Documents** tab and upload one or more PDF files so RogeX can answer your questions."
        )

    # Empty State / Quick Suggested Questions
    if len(st.session_state.chat_history) == 0:
        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("#### 💡 Quick Questions to Try:")
        suggestions = [
            "What is the main objective or thesis of the document?",
            "Summarize the key conclusions and practical findings.",
            "Explain the technical methodology used in this work.",
            "What limitations or future improvements are mentioned?",
        ]
        s_cols = st.columns(2)
        for idx, prompt_text in enumerate(suggestions):
            with s_cols[idx % 2]:
                if st.button(f"🔍 {prompt_text}", key=f"chat_sug_{idx}", use_container_width=True):
                    st.session_state.suggested_question = prompt_text
                    st.rerun()

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    # Render Conversation History
    for msg in st.session_state.chat_history:
        role = msg.get("role")
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(
                f"""
                <div class="chat-bubble-user">
                    <b>You</b><br>
                    {content}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="chat-bubble-assistant">
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
                        <span style="font-weight:700; color:#10B981;">⚡ RogeX Intelligence</span>
                    </div>
                    {content}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display Grounded Sources if available
            if sources:
                with st.expander(f"📚 Verified Sources ({len(sources)} citations)", expanded=False):
                    for idx, src in enumerate(sources, start=1):
                        source_doc = src.get("source", "Document")
                        page_num = src.get("page", 1)
                        score = src.get("score", 0.0)
                        snippet = src.get("snippet", "")

                        pct = int(score * 100) if score <= 1.0 else 100

                        st.markdown(
                            f"""
                            <div class="source-box">
                                <span class="source-badge">
                                    📄 {source_doc} &bull; Page {page_num} &bull; {pct}% Match
                                </span>
                                <div class="source-snippet">
                                    "{truncate_snippet(snippet, 280)}"
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Check for suggested question to trigger
    auto_query = None
    if st.session_state.get("suggested_question"):
        auto_query = st.session_state.suggested_question
        st.session_state.suggested_question = None

    # Chat Input Box
    user_input = st.chat_input("Ask a question about your uploaded documents...")
    query_to_process = auto_query or user_input

    if query_to_process:
        query_text = query_to_process.strip()

        # 1. Record User Message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query_text,
                "sources": [],
            }
        )
        st.session_state.total_queries_asked += 1

        # 2. Execute RAG Query with Loading Indicator
        top_k = st.session_state.get("top_k", config.DEFAULT_TOP_K)

        with st.spinner("Searching FAISS index and generating grounded answer with Gemini..."):
            response = rag_engine.query(query_text, top_k=top_k)

        # Convert citations to dictionaries
        citations_data = [
            {
                "source": c.source,
                "page": c.page,
                "score": c.score,
                "snippet": c.snippet,
            }
            for c in response.sources
        ]

        # 3. Record Assistant Response
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": citations_data,
            }
        )

        st.rerun()
