"""Sidebar UI components for the Streamlit application.

Provides the file-upload widget, system-status display, and ingestion
toast notifications.
"""

import os
from typing import List

import streamlit as st

UPLOAD_DIR = "uploaded_docs"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(
    is_db_ready: bool,
    total_chunks: int = 0,
    ingest_time: float = 0.0,
) -> List[str]:
    """Render the sidebar with file uploader and system status indicators.

    Uploaded PDFs are persisted to the ``uploaded_docs/`` directory so they
    survive Streamlit reruns.  Returns the list of *saved file paths* so the
    ingestion pipeline can read them from disk.

    Args:
        is_db_ready: Whether the vector database has been initialised.
        total_chunks: Number of document chunks currently indexed.
        ingest_time: Time taken for the last ingestion run (seconds).

    Returns:
        A list of absolute paths to the saved PDF files. Empty list when no
        files are uploaded.
    """
    st.sidebar.title("⚙️ Configuration")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_files = st.sidebar.file_uploader(
        "📥 Upload Knowledge Base (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="อัปโหลดคู่มือหรือกฎระเบียบขององค์กร (รองรับหลายไฟล์)",
    )

    saved_paths: List[str] = []
    if uploaded_files:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        for uf in uploaded_files:
            dest = os.path.join(UPLOAD_DIR, uf.name)
            with open(dest, "wb") as fh:
                fh.write(uf.getvalue())
            saved_paths.append(dest)

    st.sidebar.divider()
    
    st.sidebar.markdown("### 📁 Uploaded Files")
    if os.path.exists(UPLOAD_DIR):
        existing_files = [f for f in os.listdir(UPLOAD_DIR) if f.lower().endswith(".pdf")]
    else:
        existing_files = []
        
    if not existing_files:
        st.sidebar.caption("ยังไม่มีไฟล์ในระบบ")
    else:
        for f in existing_files:
            col1, col2 = st.sidebar.columns([5, 1])
            col1.caption(f"📄 {f}")
            if col2.button("✖", key=f"del_{f}", help=f"ลบไฟล์ {f}"):
                os.remove(os.path.join(UPLOAD_DIR, f))
                st.session_state.uploader_key += 1
                st.rerun()

    st.sidebar.divider()

    st.sidebar.markdown("### 📊 System Status")
    if is_db_ready:
        st.sidebar.success("✅ Database Ready")
        st.sidebar.caption(f"🧩 ข้อมูลในระบบ: {total_chunks} Chunks")
        if ingest_time > 0:
            st.sidebar.caption(f"⏱️ Ingestion Time: {ingest_time:.2f} วินาที")
    else:
        st.sidebar.warning("⚠️ No Data in System")
        st.sidebar.caption("กรุณาอัปโหลดเอกสารเพื่อเริ่มต้น")

    st.sidebar.divider()
    st.sidebar.markdown("### 🛠️ Danger Zone")
    if st.sidebar.button("🗑️ Clear Knowledge Base", type="primary", use_container_width=True):
        st.session_state.execute_kb_clear = True
        st.rerun()

    return saved_paths


# ---------------------------------------------------------------------------
# Toast notifications
# ---------------------------------------------------------------------------

def show_ingestion_toast(status: str = "processing") -> None:
    """Show a toast notification reflecting the ingestion pipeline status.

    Args:
        status: One of ``"processing"``, ``"success"``, or ``"error"``.
    """
    toast_map = {
        "processing": ("กำลังประมวลผลและเรียนรู้เอกสาร...", "⏳"),
        "success": ("อัปโหลดและสร้าง Database สำเร็จ!", "🎉"),
        "error": ("เกิดข้อผิดพลาดในการอ่านไฟล์", "🚨"),
    }

    message, icon = toast_map.get(status, ("Unknown status", "❓"))
    st.toast(message, icon=icon)