"""RogeX Helper Functions, UI Styling, and Session State Management."""

import hashlib
import re
from typing import Dict, Any
import streamlit as st


def format_file_size(num_bytes: int) -> str:
    """Convert bytes to a human-readable file size string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


def compute_file_hash(data: bytes) -> str:
    """Generate SHA-256 hash of file content for deduplication."""
    return hashlib.sha256(data).hexdigest()[:16]


def sanitize_text(text: str) -> str:
    """Normalize whitespace and strip spurious characters from extracted text."""
    if not text:
        return ""
    # Replace null bytes or control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    # Replace multiple newlines with at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple horizontal spaces with single space
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def truncate_snippet(text: str, max_chars: int = 240) -> str:
    """Truncate text for preview cards with clean ellipsis."""
    clean = sanitize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0] + "..."


def init_session_state() -> None:
    """Initialize Streamlit session state keys with robust default structures."""
    defaults: Dict[str, Any] = {
        "active_view": "dashboard",  # "dashboard", "chat", "documents", "settings"
        "uploaded_docs": {},  # doc_id -> {filename, size, pages, chunks, hash, raw_text}
        "chat_history": [],  # list of {role: "user"|"assistant", content: str, sources: list, timestamp: str}
        "vector_store_ready": False,
        "total_chunks_indexed": 0,
        "total_queries_asked": 0,
        "suggested_question": None,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 4,
        "selected_model": "gemini-2.5-flash",
        "custom_api_key": "",
    }

    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def load_custom_css() -> str:
    """Return modern, premium AI SaaS styling with off-white backgrounds, charcoal text, and sage green accents."""
    return """
    <style>
    /* -------------------------------------------------------------
       ROGEX MODERN PREMIUM AI SAAS DESIGN SYSTEM
       Theme: Cream/Off-white background (#FBFBF9),
              Deep charcoal typography (#1C1D1F / #2B2C2E),
              Subtle emerald/sage green accents (#10B981 / #059669)
       ------------------------------------------------------------- */

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-main: #FBFBF9;
        --bg-card: #FFFFFF;
        --bg-card-subtle: #F6F6F2;
        --border-color: #E7E7E0;
        --border-focus: #10B981;
        --text-primary: #1C1D1F;
        --text-secondary: #5E6266;
        --text-muted: #8E9296;
        --accent-emerald: #10B981;
        --accent-dark-green: #047857;
        --accent-light-green: #ECFDF5;
        --shadow-subtle: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.02);
        --shadow-card: 0 4px 12px rgba(0,0,0,0.03), 0 1px 3px rgba(0,0,0,0.02);
        --shadow-lift: 0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.03);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    /* Base App Styling */
    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Global Headings */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-primary) !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Header styling */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F7F7F4 !important;
        border-right: 1px solid var(--border-color) !important;
    }

    section[data-testid="stSidebar"] div.block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* RogeX Brand Header in Sidebar */
    .rogex-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0.5rem 0.2rem 1.5rem 0.2rem;
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 1.5rem;
    }

    .rogex-logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #10B981 0%, #047857 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 800;
        font-size: 20px;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.25);
    }

    .rogex-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        line-height: 1.1;
    }

    .rogex-brand-badge {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        color: var(--accent-emerald);
        background: var(--accent-light-green);
        padding: 2px 6px;
        border-radius: 4px;
        display: inline-block;
        margin-top: 2px;
    }

    /* Hero / Header Section */
    .hero-container {
        padding: 2.2rem 2.5rem;
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAF8 100%);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-card);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 320px;
        height: 100%;
        background: radial-gradient(circle at 100% 0%, rgba(16, 185, 129, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.035em;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        font-weight: 400;
        max-width: 620px;
        line-height: 1.5;
        margin-bottom: 0;
    }

    /* Stat Cards */
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.25rem 1.4rem;
        box-shadow: var(--shadow-subtle);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lift);
        border-color: #D6D6CC;
    }

    .stat-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        line-height: 1;
    }

    .stat-subtext {
        font-size: 0.78rem;
        color: var(--accent-emerald);
        font-weight: 600;
        margin-top: 0.4rem;
    }

    /* Feature Cards */
    .feature-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.4rem;
        height: 100%;
        box-shadow: var(--shadow-subtle);
    }

    .feature-step-num {
        width: 28px;
        height: 28px;
        background: var(--accent-light-green);
        color: var(--accent-dark-green);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-bottom: 0.8rem;
    }

    .feature-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }

    .feature-card-desc {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }

    /* Document Item Card in Documents view */
    .doc-item-card {
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: var(--shadow-subtle);
    }

    .doc-name {
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text-primary);
    }

    .doc-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
        display: flex;
        gap: 12px;
        margin-top: 4px;
    }

    /* Chat Messages Styling */
    .chat-bubble-user {
        background-color: #1F2421;
        color: #FFFFFF !important;
        padding: 1rem 1.3rem;
        border-radius: 18px 18px 4px 18px;
        max-width: 82%;
        margin-left: auto;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(31, 36, 33, 0.12);
        font-size: 0.96rem;
        line-height: 1.55;
    }

    .chat-bubble-user p, .chat-bubble-user span {
        color: #FFFFFF !important;
    }

    .chat-bubble-assistant {
        background-color: #FFFFFF;
        color: var(--text-primary);
        border: 1px solid var(--border-color);
        padding: 1.25rem 1.45rem;
        border-radius: 18px 18px 18px 4px;
        max-width: 88%;
        margin-right: auto;
        margin-bottom: 1.25rem;
        box-shadow: var(--shadow-card);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    /* Source Citation Accordion / Box */
    .source-box {
        margin-top: 1rem;
        padding: 0.9rem 1.1rem;
        background: #F8FAF8;
        border-left: 3px solid var(--accent-emerald);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        font-size: 0.85rem;
    }

    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #FFFFFF;
        border: 1px solid #D1E7DD;
        color: var(--accent-dark-green);
        font-weight: 600;
        font-size: 0.76rem;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    .source-snippet {
        color: var(--text-secondary);
        font-size: 0.82rem;
        line-height: 1.45;
        font-style: italic;
    }

    /* Custom Buttons */
    button[kind="primary"] {
        background-color: var(--accent-emerald) !important;
        border-color: var(--accent-emerald) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="primary"]:hover {
        background-color: var(--accent-dark-green) !important;
        border-color: var(--accent-dark-green) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.35) !important;
    }

    button[kind="secondary"] {
        background-color: #FFFFFF !important;
        border-color: var(--border-color) !important;
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        border-radius: var(--radius-sm) !important;
        transition: all 0.15s ease !important;
    }

    button[kind="secondary"]:hover {
        background-color: #F6F6F2 !important;
        border-color: #D6D6CC !important;
        color: var(--text-primary) !important;
    }

    /* Status Pills */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }

    .status-active {
        background-color: var(--accent-light-green);
        color: var(--accent-dark-green);
        border: 1px solid #A7F3D0;
    }

    .status-empty {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }

    /* Suggested Question Button */
    .suggested-btn {
        background-color: #FFFFFF;
        border: 1px solid var(--border-color);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        display: inline-block;
        margin: 4px 4px 4px 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .suggested-btn:hover {
        border-color: var(--accent-emerald);
        color: var(--accent-emerald);
        background-color: var(--accent-light-green);
    }
    </style>
    """
