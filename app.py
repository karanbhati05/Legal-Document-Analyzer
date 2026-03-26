from __future__ import annotations

import os
from typing import Iterable, Tuple

import streamlit as st
from dotenv import load_dotenv

from rag import LegalRAGAnalyzer, Settings


load_dotenv()


def _sync_env_from_streamlit_secrets() -> None:
    # Streamlit Cloud provides runtime secrets via st.secrets.
    secret_keys = [
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "MODEL_PROVIDER",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "TOP_K",
    ]
    try:
        for key in secret_keys:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        # Local runs may not have a Streamlit secrets file.
        return


_sync_env_from_streamlit_secrets()

st.set_page_config(page_title="Legal Document Analyzer (RAG + GenAI)", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal Document Analyzer")
st.caption("Upload legal documents, get explanation + summary, and ask specific questions using a RAG pipeline.")

os.environ.setdefault("MODEL_PROVIDER", "gemini")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("Missing GOOGLE_API_KEY. Add it in Streamlit Cloud secrets or local .env.")
    st.stop()


def _upload_signature(files: Iterable) -> Tuple[Tuple[str, int], ...]:
    return tuple(sorted((f.name, int(f.size)) for f in files))


if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "stats" not in st.session_state:
    st.session_state.stats = None
if "upload_signature" not in st.session_state:
    st.session_state.upload_signature = None

uploaded_files = st.file_uploader(
    "Upload legal documents",
    type=None,
    accept_multiple_files=True,
    help="Supported directly: PDF, DOCX, TXT, MD, RTF, CSV, JSON, HTML, XML. Other types are best-effort text decode.",
)

if st.button("Reset Session"):
    st.session_state.analyzer = None
    st.session_state.stats = None
    st.session_state.upload_signature = None
    st.success("Session reset.")

if uploaded_files:
    current_signature = _upload_signature(uploaded_files)
    if current_signature != st.session_state.upload_signature:
        try:
            with st.status("Building knowledge base...", expanded=True) as status:
                status.write("Extracting text from uploaded legal documents...")
                settings = Settings()
                analyzer = LegalRAGAnalyzer(settings=settings)

                status.write("Splitting document content into chunks...")
                status.write("Creating vector embeddings and indexing for retrieval...")
                stats = analyzer.ingest(uploaded_files)

                st.session_state.analyzer = analyzer
                st.session_state.stats = stats
                st.session_state.upload_signature = current_signature
                status.update(label="Knowledge base ready", state="complete", expanded=False)

            st.success(
                f"Knowledge base built automatically. Documents: {stats['raw_documents']} | Chunks: {stats['chunks']}"
            )
        except Exception as exc:
            st.session_state.analyzer = None
            st.session_state.stats = None
            st.session_state.upload_signature = None
            st.error(f"Failed to build knowledge base: {exc}")

if st.session_state.stats:
    st.info(
        f"Active KB -> Documents: {st.session_state.stats['raw_documents']} | "
        f"Chunks: {st.session_state.stats['chunks']}"
    )

if st.session_state.analyzer is not None:
    tab1, tab2, tab3 = st.tabs(["Explanation", "Summary", "Q&A"])

    with tab1:
        if st.button("Generate Explanation", use_container_width=True):
            with st.spinner("Analyzing legal content..."):
                try:
                    explanation = st.session_state.analyzer.explain()
                    st.markdown(explanation)
                except Exception as exc:
                    st.error(f"Explanation failed: {exc}")

    with tab2:
        if st.button("Generate Summary", use_container_width=True):
            with st.spinner("Summarizing key legal points..."):
                try:
                    summary = st.session_state.analyzer.summarize()
                    st.markdown(summary)
                except Exception as exc:
                    st.error(f"Summary failed: {exc}")

    with tab3:
        user_query = st.text_area(
            "Ask a specific query about your legal document(s)",
            placeholder="Example: What are the termination conditions and notice period?",
            height=120,
        )
        if st.button("Get Answer", use_container_width=True):
            if not user_query.strip():
                st.warning("Enter a query first.")
            else:
                with st.spinner("Retrieving context and generating answer..."):
                    try:
                        answer = st.session_state.analyzer.answer_query(user_query)
                        st.markdown(answer)
                    except Exception as exc:
                        st.error(f"Q&A failed: {exc}")
else:
    st.write("Upload documents to automatically build the RAG knowledge base.")

st.divider()
st.caption("Informational analysis only. This app does not provide legal advice.")
