"""RogeX Sidebar Navigation and Global Actions Component."""

import streamlit as st
from config.settings import AppConfig


def render_sidebar(config: AppConfig, rag_engine) -> str:
    """Render the application sidebar with branding, navigation, and system status."""
    with st.sidebar:
        # Branding Header
        st.markdown(
            """
            <div class="rogex-brand">
                <div class="rogex-logo-icon">R</div>
                <div>
                    <div class="rogex-brand-title">RogeX</div>
                    <span class="rogex-brand-badge">DOCUMENT INTELLIGENCE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Primary Navigation
        nav_options = {
            "dashboard": "📊 Dashboard",
            "chat": "💬 AI Chat",
            "documents": "📁 Documents",
            "settings": "⚙️ Settings",
        }

        current_view = st.session_state.get("active_view", "dashboard")
        active_index = list(nav_options.keys()).index(current_view) if current_view in nav_options else 0

        selected_nav = st.radio(
            "Navigation",
            options=list(nav_options.keys()),
            format_func=lambda k: nav_options[k],
            index=active_index,
            label_visibility="collapsed",
        )

        if selected_nav != current_view:
            st.session_state.active_view = selected_nav
            st.rerun()

        st.markdown("---")

        # Quick Action: New Chat
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.chat_history = []
            st.session_state.active_view = "chat"
            st.rerun()

        # Documents Overview in Sidebar
        total_docs = len(st.session_state.get("uploaded_docs", {}))
        total_chunks = st.session_state.get("total_chunks_indexed", 0)

        st.markdown(
            f"""
            <div style="margin: 1.2rem 0 0.5rem 0;">
                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8E9296; font-weight: 700;">
                    KNOWLEDGE BASE ({total_docs})
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if total_docs > 0:
            for doc_id, doc in list(st.session_state.uploaded_docs.items())[:5]:
                filename = doc.get("filename", "Document")
                pages = doc.get("pages", 1)
                st.markdown(
                    f"""
                    <div style="font-size: 0.82rem; padding: 5px 0; color: #2B2C2E; display: flex; align-items: center; justify-content: space-between;">
                        <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 170px;" title="{filename}">
                            📄 {filename}
                        </span>
                        <span style="font-size: 0.72rem; color: #8E9296;">{pages}p</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if total_docs > 5:
                st.caption(f"+ {total_docs - 5} more documents")
        else:
            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #8E9296; padding: 4px 0;">
                    No documents indexed yet.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # System Status Indicators
        st.markdown(
            """
            <div style="margin-bottom: 0.5rem;">
                <span style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #8E9296; font-weight: 700;">
                    SYSTEM STATUS
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gemini API Status
        is_gemini_connected = rag_engine.is_configured()
        if is_gemini_connected:
            st.markdown(
                """
                <div class="status-pill status-active" style="width: 100%; margin-bottom: 6px;">
                    <span style="display:inline-block; width:7px; height:7px; background:#10B981; border-radius:50%;"></span>
                    Gemini API Connected
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-pill status-empty" style="width: 100%; margin-bottom: 6px;">
                    <span style="display:inline-block; width:7px; height:7px; background:#F59E0B; border-radius:50%;"></span>
                    Gemini Key Needed
                </div>
                """,
                unsafe_allow_html=True,
            )

        # FAISS Index Status
        vector_store_ready = st.session_state.get("vector_store_ready", False) and total_chunks > 0
        if vector_store_ready:
            st.markdown(
                f"""
                <div class="status-pill status-active" style="width: 100%; margin-bottom: 6px;">
                    <span style="display:inline-block; width:7px; height:7px; background:#10B981; border-radius:50%;"></span>
                    FAISS: {total_chunks} Chunks
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="status-pill status-empty" style="width: 100%; margin-bottom: 6px;">
                    <span style="display:inline-block; width:7px; height:7px; background:#9CA3AF; border-radius:50%;"></span>
                    FAISS Index: Empty
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Clear Chat Button
        if len(st.session_state.get("chat_history", [])) > 0:
            if st.button("🗑️ Clear Conversation", use_container_width=True, kind="secondary"):
                st.session_state.chat_history = []
                st.rerun()

        # Footer
        st.markdown(
            f"""
            <div style="font-size: 0.72rem; color: #A1A5A9; text-align: center; margin-top: 1.5rem;">
                RogeX v{config.APP_VERSION} • Academic RAG Project
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.active_view
