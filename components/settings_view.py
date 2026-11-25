"""RogeX Settings and Configuration Component."""

import streamlit as st
from config.settings import AppConfig
from core.rag_engine import RAGEngine
from core.vector_store import FAISSVectorStore


def render_settings(
    config: AppConfig,
    vector_store: FAISSVectorStore,
    rag_engine: RAGEngine,
) -> None:
    """Render the configuration panel for Gemini API, RAG hyperparameters, and model telemetry."""
    st.markdown("## ⚙️ Application Settings")
    st.markdown("Fine-tune retrieval parameters, manage API authentication, and inspect system telemetry.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        # 1. Gemini LLM Authentication
        st.markdown("### 🔑 Google Gemini Configuration")

        current_key = st.session_state.get("custom_api_key", "") or config.GEMINI_API_KEY
        api_key_input = st.text_input(
            "Gemini API Key",
            value=current_key,
            type="password",
            help="Get your free API key at https://aistudio.google.com/ and paste it here, or set GEMINI_API_KEY in .env",
        )

        if api_key_input != current_key:
            st.session_state.custom_api_key = api_key_input
            rag_engine.set_api_key(api_key_input)
            st.success("Gemini API key updated!")
            st.rerun()

        # Model Selection
        selected_model = st.selectbox(
            "Gemini Model",
            options=config.AVAILABLE_GEMINI_MODELS,
            index=config.AVAILABLE_GEMINI_MODELS.index(
                st.session_state.get("selected_model", config.DEFAULT_GEMINI_MODEL)
            )
            if st.session_state.get("selected_model") in config.AVAILABLE_GEMINI_MODELS
            else 0,
            help="Select the Gemini model variant used to synthesize answers from context.",
        )

        if selected_model != st.session_state.get("selected_model"):
            st.session_state.selected_model = selected_model
            rag_engine.set_model(selected_model)
            st.success(f"Model switched to {selected_model}")

        st.markdown("---")

        # 2. RAG Hyperparameter Tuning
        st.markdown("### 🎛️ RAG Ingestion & Retrieval Settings")

        top_k = st.slider(
            "Retrieval Top-K (Document Chunks)",
            min_value=1,
            max_value=config.MAX_TOP_K,
            value=st.session_state.get("top_k", config.DEFAULT_TOP_K),
            help="Number of most relevant context chunks retrieved from FAISS for each question.",
        )
        st.session_state.top_k = top_k

        chunk_size = st.slider(
            "Chunk Size (Characters)",
            min_value=config.MIN_CHUNK_SIZE,
            max_value=config.MAX_CHUNK_SIZE,
            value=st.session_state.get("chunk_size", config.DEFAULT_CHUNK_SIZE),
            step=100,
            help="Maximum character length of each document excerpt chunk.",
        )
        st.session_state.chunk_size = chunk_size

        chunk_overlap = st.slider(
            "Chunk Overlap (Characters)",
            min_value=0,
            max_value=min(chunk_size // 2, 500),
            value=st.session_state.get("chunk_overlap", config.DEFAULT_CHUNK_OVERLAP),
            step=25,
            help="Number of overlapping characters between consecutive chunks to preserve context.",
        )
        st.session_state.chunk_overlap = chunk_overlap

    with col_right:
        # 3. Embedding Model Information Card
        st.markdown("### 🧠 Embedding Engine")
        st.markdown(
            f"""
            <div class="stat-card" style="margin-bottom: 1rem;">
                <div class="stat-label">Model Name</div>
                <div style="font-weight:700; font-size:1rem; color:#1C1D1F;">all-MiniLM-L6-v2</div>
                <div style="font-size:0.8rem; color:#5E6266; margin-top:4px;">HuggingFace Sentence Transformers</div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #E7E7E0;">
                <div style="font-size:0.8rem; line-height:1.6;">
                    • <b>Dimensions:</b> {config.EMBEDDING_DIMENSION}<br>
                    • <b>Distance Metric:</b> Normalized Inner Product (Cosine)<br>
                    • <b>Execution:</b> 100% Local (CPU/PyTorch)<br>
                    • <b>Cost:</b> Free & Unlimited
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4. Vector Database Status Card
        st.markdown("### 🗄️ FAISS Vector Index")
        stats = vector_store.get_stats()
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">Vector Store Status</div>
                <div style="font-weight:700; font-size:1.1rem; color:#10B981;">Active & Ready</div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #E7E7E0;">
                <div style="font-size:0.8rem; line-height:1.6;">
                    • <b>Indexed Vectors:</b> {stats['total_vectors']}<br>
                    • <b>Unique Documents:</b> {stats['total_documents']}<br>
                    • <b>Index Algorithm:</b> FAISS IndexFlatIP<br>
                    • <b>Storage Path:</b> <code>data/vectorstore/</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        # Reset Vector Database
        if st.button("🗑️ Reset Vector Database", use_container_width=True, type="secondary"):
            vector_store.clear()
            st.session_state.uploaded_docs = {}
            st.session_state.total_chunks_indexed = 0
            st.session_state.vector_store_ready = False
            vector_store.save(config.VECTORSTORE_DIR)
            st.warning("Vector database and document cache have been reset.")
            st.rerun()
