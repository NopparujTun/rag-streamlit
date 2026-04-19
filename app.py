"""Enterprise Smart Knowledge-Base — Streamlit Application."""

import logging
import os
import shutil
import time
from typing import List

import streamlit as st
from dotenv import load_dotenv

# MUST be the first Streamlit command
st.set_page_config(page_title="Enterprise Smart KB", page_icon="🧠", layout="wide")

from src.ingestion.pipeline import run_ingestion_pipeline
from src.rag.embeddings import get_embedding_model
from src.rag.evaluator import evaluate_faithfulness
from src.rag.generator import generate_answer
from src.rag.vectorstore import (
    clear_hybrid_store,
    load_hybrid_store,
)
from src.ui.chat import (
    add_user_message,
    display_sources,
    display_agent_steps,
    init_chat_history,
    render_chat_history,
    render_evaluation_metrics,
)
from src.ui.sidebar import render_sidebar, show_ingestion_toast, UPLOAD_DIR
from src.utils.config import load_config

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

config = load_config()

def init_session_state():
    """Initialize default values in session state."""
    defaults = {
        "total_chunks": 0,
        "ingest_time": 0.0,
        "processed_files": [],
        "execute_kb_clear": False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

@st.cache_resource(show_spinner=False)
def get_models():
    """Load and cache all heavy models."""
    from src.rag.vectorstore import get_reranker
    
    embedding_model = get_embedding_model(
        config["embedding"]["model_name"],
        config["embedding"]["device"],
    )
    
    vectorstore, bm25_retriever = load_hybrid_store(
        embedding_model=embedding_model,
        persist_dir=config["vector_db"]["persist_directory"],
        index_name=config["vector_db"]["index_name"],
    )
    
    reranker = get_reranker()
    
    return embedding_model, vectorstore, bm25_retriever, reranker

def handle_ingestion(uploaded_file_paths, embedding_model):
    """Orchestrate the ingestion of new documents."""
    current_file_key = sorted(os.path.basename(p) for p in uploaded_file_paths)
    
    if current_file_key != st.session_state.processed_files:
        show_ingestion_toast("processing")
        try:
            disk_paths = sorted([
                os.path.join(UPLOAD_DIR, f) 
                for f in os.listdir(UPLOAD_DIR) 
                if f.lower().endswith(".pdf")
            ])
            
            vectorstore, bm25_retriever, total_chunks, ingest_time = run_ingestion_pipeline(
                file_paths=disk_paths,
                embedding_model=embedding_model,
                index_name=config["vector_db"]["index_name"],
                persist_dir=config["vector_db"]["persist_directory"],
                chunk_size=config["ingestion"]["chunk_size"],
                chunk_overlap=config["ingestion"]["chunk_overlap"]
            )
            
            # Update state and refresh
            st.session_state.total_chunks = total_chunks
            st.session_state.ingest_time = ingest_time
            st.session_state.processed_files = current_file_key
            st.cache_resource.clear()
            
            show_ingestion_toast("success")
            time.sleep(1)
            st.rerun()
            
        except Exception as exc:
            logger.error(f"Ingestion failed: {exc}")
            show_ingestion_toast("error")
            st.sidebar.error(f"❌ {exc}")

def handle_clear_kb(vectorstore):
    """Clear all data from the system."""
    st.session_state.execute_kb_clear = False
    
    if vectorstore is not None:
        clear_hybrid_store(vectorstore, config["vector_db"]["persist_directory"])
        
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

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

# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------

def main():
    init_session_state()
    
    st.title("🧠 Enterprise Smart Knowledge-Base")
    st.markdown("ระบบ AI Assistant ค้นหาและสรุปข้อมูลองค์กร (Hybrid Search Edition)")

    # Load Resources
    with st.spinner("⏳ ยกฐานข้อมูล AI... (กำลังโหลดโมเดล หรือดาวน์โหลดครั้งแรกอาจใช้เวลาสักครู่)"):
        embedding_model, vectorstore, bm25_retriever, reranker = get_models()
        st.session_state.reranker = reranker

    is_db_ready = vectorstore is not None

    # Sidebar
    uploaded_file_paths = render_sidebar(
        is_db_ready,
        st.session_state.total_chunks,
        st.session_state.ingest_time
    )

    # Ingestion Logic
    if uploaded_file_paths:
        handle_ingestion(uploaded_file_paths, embedding_model)

    # Clear Logic
    if st.session_state.get("execute_kb_clear"):
        handle_clear_kb(vectorstore)

    # Chat UI
    init_chat_history()
    render_chat_history()

    if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
        if not is_db_ready:
            st.warning("⚠️ กรุณาอัปโหลดเอกสาร Knowledge Base ก่อนเริ่มใช้งานครับ")
            st.stop()

        add_user_message(prompt)

        with st.chat_message("assistant"):
            with st.spinner("กำลังประมวลผล..."):
                history_text = "\n".join(
                    f"{msg['role']}: {msg['content']}"
                    for msg in st.session_state.messages[-3:-1]
                )

                start_time = time.time()
                answer, sources, steps = generate_answer(
                    query=prompt,
                    vectorstore=vectorstore,
                    bm25_retriever=bm25_retriever,
                    chat_history=history_text,
                    reranker=st.session_state.reranker
                )
                response_time = time.time() - start_time

                display_agent_steps(steps)
                st.write(answer)

                eval_result = evaluate_faithfulness(sources, answer)
                render_evaluation_metrics(response_time, eval_result)
                display_sources(sources)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "steps": steps}
                )

if __name__ == "__main__":
    main()
