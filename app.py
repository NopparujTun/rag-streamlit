"""Enterprise Smart Knowledge-Base — Streamlit Application.

Main entry point for the RAG-powered AI assistant.  Orchestrates:

1. PDF upload and ingestion (chunking + indexing).
2. Hybrid search (Pinecone vector + BM25 keyword).
3. LLM answer generation with faithfulness evaluation.
"""

import logging
import os
import time
import shutil
from typing import Any, Dict, List

import streamlit as st
import yaml
from dotenv import load_dotenv

# MUST be the first Streamlit command
st.set_page_config(page_title="Enterprise Smart KB", page_icon="🧠", layout="wide")

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import process_uploaded_pdf, process_uploaded_pdf_path
from src.rag.embeddings import get_embedding_model
from src.rag.evaluator import evaluate_faithfulness
from src.rag.generator import generate_answer
from src.rag.vectorstore import (
    clear_hybrid_store,
    load_hybrid_store,
    perform_hybrid_search,
    save_hybrid_store,
)
from src.ui.chat import (
    add_user_message,
    display_sources,
    init_chat_history,
    render_chat_history,
    render_evaluation_metrics,
)
from src.ui.sidebar import render_sidebar, show_ingestion_toast

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration or halt the Streamlit app on failure.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dictionary.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception as exc:
        st.error(f"🚨 Config Error: {exc}")
        st.stop()
        return {}  # Unreachable, satisfies type checker.


config = _load_config()

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.title("🧠 Enterprise Smart Knowledge-Base")
st.markdown("ระบบ AI Assistant ค้นหาและสรุปข้อมูลองค์กร (Hybrid Search Edition)")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "ingest_time" not in st.session_state:
    st.session_state.ingest_time = 0.0
if "processed_files" not in st.session_state:
    st.session_state.processed_files: List[str] = []

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_and_chunk_paths(
    file_paths: tuple, chunk_size: int, chunk_overlap: int
) -> list:
    """Load all PDFs from *file_paths* from disk and split them into chunks (cached).

    Args:
        file_paths: Tuple of absolute paths to PDF files on disk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Combined list of document chunks across all files.
    """
    all_docs = []
    for path in file_paths:
        all_docs.extend(process_uploaded_pdf_path(path))
    return chunk_documents(all_docs, chunk_size, chunk_overlap)


@st.cache_data(show_spinner=False)
def _load_and_chunk(
    file_bytes: bytes, chunk_size: int, chunk_overlap: int
) -> list:
    """Load a PDF from bytes and split it into chunks (cached)."""
    docs = process_uploaded_pdf(file_bytes)
    return chunk_documents(docs, chunk_size, chunk_overlap)


@st.cache_resource(show_spinner=False)
def _load_model(model_name: str, device: str):
    """Load the embedding model (cached as a resource)."""
    return get_embedding_model(model_name, device)


@st.cache_resource(show_spinner=False)
def _init_vectorstore():
    """Connect to the existing vector store if one is persisted."""
    if os.path.exists(config["vector_db"]["persist_directory"]):
        return load_hybrid_store(
            embedding_model=embedding_model,
            persist_dir=config["vector_db"]["persist_directory"],
            index_name=config["vector_db"]["index_name"],
        )
    return None, None


with st.spinner("⏳ ยกฐานข้อมูล AI... (กำลังโหลดโมเดล หรือดาวน์โหลดครั้งแรกอาจใช้เวลาสักครู่)"):
    embedding_model = _load_model(
        config["embedding"]["model_name"],
        config["embedding"]["device"],
    )
    vectorstore, bm25_retriever = _init_vectorstore()

is_db_ready: bool = vectorstore is not None

# =========================================================================
# 1. File upload (Sidebar)
# =========================================================================

uploaded_file_paths = render_sidebar(
    is_db_ready,
    st.session_state.total_chunks,
    st.session_state.ingest_time,
)

if uploaded_file_paths:
    # Build a stable key from the sorted file names to detect new uploads.
    current_file_key = sorted(os.path.basename(p) for p in uploaded_file_paths)
    if current_file_key != st.session_state.processed_files:
        show_ingestion_toast("processing")
        try:
            # Read all PDFs from the uploaded_docs directory on disk.
            from src.ui.sidebar import UPLOAD_DIR
            disk_paths = tuple(
                sorted(
                    os.path.join(UPLOAD_DIR, f)
                    for f in os.listdir(UPLOAD_DIR)
                    if f.lower().endswith(".pdf")
                )
            )

            chunks = _load_and_chunk_paths(
                disk_paths,
                config["ingestion"]["chunk_size"],
                config["ingestion"]["chunk_overlap"],
            )
            if not chunks:
                raise ValueError("ไม่พบตัวอักษรในไฟล์เหล่านี้")

            st.cache_resource.clear()
            start_time = time.time()

            vectorstore, bm25_retriever = save_hybrid_store(
                chunks=chunks,
                embedding_model=embedding_model,
                persist_dir=config["vector_db"]["persist_directory"],
                index_name=config["vector_db"]["index_name"],
            )

            st.session_state.total_chunks = len(chunks)
            st.session_state.ingest_time = time.time() - start_time
            st.session_state.processed_files = current_file_key

            logger.info(
                "Ingestion complete: %d chunks from %d files in %.2fs",
                len(chunks),
                len(disk_paths),
                st.session_state.ingest_time,
            )

            show_ingestion_toast("success")
            time.sleep(1)
            st.rerun()

        except Exception as exc:
            logger.error("Ingestion failed: %s", exc)
            show_ingestion_toast("error")
            st.sidebar.error(f"❌ {exc}")

# =========================================================================
# 2. Logic: Clear Knowledge Base
# =========================================================================

if st.session_state.get("execute_kb_clear"):
    st.session_state.execute_kb_clear = False
    
    if vectorstore is not None:
        clear_hybrid_store(vectorstore, config["vector_db"]["persist_directory"])
        
    from src.ui.sidebar import UPLOAD_DIR
    if os.path.exists(UPLOAD_DIR):
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass

    st.session_state.total_chunks = 0
    st.session_state.ingest_time = 0.0
    st.session_state.processed_files = []
    st.session_state.messages = []
    
    if "uploader_key" in st.session_state:
        st.session_state.uploader_key += 1
        
    st.cache_resource.clear()
    st.cache_data.clear()
    
    st.success("Knowledge Base cleared successfully!")
    time.sleep(1)
    st.rerun()

# =========================================================================
# 3. Chat interface (Main area)
# =========================================================================

init_chat_history()
render_chat_history()

if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    if not is_db_ready:
        st.warning("⚠️ กรุณาอัปโหลดเอกสาร Knowledge Base ก่อนเริ่มใช้งานครับ")
        st.stop()

    add_user_message(prompt)

    with st.chat_message("assistant"):
        with st.spinner("กำลังประมวลผล..."):
            # Build recent chat context
            history_text = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.messages[-3:-1]
            )

            # Hybrid retrieval
            retrieved_docs = perform_hybrid_search(
                prompt, vectorstore, bm25_retriever, k=3
            )

            # LLM generation
            start_response_time = time.time()
            answer, sources = generate_answer(
                prompt, retrieved_docs, chat_history=history_text
            )
            response_time = time.time() - start_response_time

            st.write(answer)

            # Faithfulness evaluation
            eval_result = evaluate_faithfulness(sources, answer)

            # Render metrics & sources
            render_evaluation_metrics(response_time, eval_result)
            display_sources(sources)

            # Persist to history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )