"""Hybrid vector + keyword store with Reciprocal Rank Fusion (RRF) and Re-ranking.

Manages a dual-retrieval system:
- **Pinecone** for dense vector similarity search (cloud).
- **BM25Retriever** for sparse keyword search (persisted locally as pickle).

Results are combined using RRF scoring, over-fetched, and optionally 
re-ranked using a Cross-Encoder for maximum precision.
"""

import logging
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple
import concurrent.futures

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# RRF smoothing constant (standard value from the original RRF paper).
_RRF_K = 60
_BM25_FILENAME = "bm25_index.pkl"


# ---------------------------------------------------------------------------
# Store persistence
# ---------------------------------------------------------------------------

def save_hybrid_store(
    chunks: List[Document],
    embedding_model: Embeddings,
    persist_dir: str,
    index_name: str,
) -> Tuple[PineconeVectorStore, BM25Retriever]:
    """Index document chunks into Pinecone and save a local BM25 index."""
    os.makedirs(persist_dir, exist_ok=True)

    # Dense vector store (cloud)
    logger.info("Uploading %d chunks to Pinecone index '%s'", len(chunks), index_name)
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
    )

    # Sparse keyword store (local)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_path = os.path.join(persist_dir, _BM25_FILENAME)
    with open(bm25_path, "wb") as fh:
        pickle.dump(bm25_retriever, fh)

    logger.info("Saved BM25 index to %s", bm25_path)
    return vectorstore, bm25_retriever


def load_hybrid_store(
    embedding_model: Embeddings,
    persist_dir: str,
    index_name: str,
) -> Tuple[PineconeVectorStore, Optional[BM25Retriever]]:
    """Connect to an existing Pinecone index and load the local BM25 index."""
    logger.info("Connecting to Pinecone index '%s'", index_name)
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model,
    )

    bm25_path = os.path.join(persist_dir, _BM25_FILENAME)
    bm25_retriever: Optional[BM25Retriever] = None
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as fh:
            bm25_retriever = pickle.load(fh)
        logger.info("Loaded BM25 index from %s", bm25_path)
    else:
        logger.warning("BM25 index not found at %s — keyword search disabled", bm25_path)

    return vectorstore, bm25_retriever


def clear_hybrid_store(
    vectorstore: PineconeVectorStore,
    persist_dir: str,
) -> None:
    """Clear all vectors from Pinecone and delete local BM25 directory."""
    logger.info("Clearing Knowledge Base: Vector database and BM25.")
    
    # 1. Clear Pinecone (Cloud)
    try:
        vectorstore.delete(delete_all=True)
        logger.info("Deleted all vectors from Pinecone.")
    except Exception as exc:
        logger.error("Failed to delete from Pinecone: %s", exc)

    # 2. Clear BM25 (Local)
    if os.path.exists(persist_dir):
        for filename in os.listdir(persist_dir):
            file_path = os.path.join(persist_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error("Failed to delete %s: %s", file_path, e)
        logger.info("Cleared contents of BM25 index directory: %s", persist_dir)


# ---------------------------------------------------------------------------
# Hybrid search & Re-ranking
# ---------------------------------------------------------------------------

def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> CrossEncoder:
    """Load a Cross-Encoder model for precision re-ranking."""
    logger.info("Loading Re-ranker model: %s (device=%s)", model_name, device)
    return CrossEncoder(model_name, max_length=512, device=device)


def _compute_rrf_scores(
    ranked_docs: List[Document],
    scores: Dict[str, float],
    mapping: Dict[str, Document],
) -> None:
    """Accumulate RRF scores for a ranked list of documents (in-place)."""
    for rank, doc in enumerate(ranked_docs):
        content = doc.page_content
        mapping[content] = doc
        scores[content] = scores.get(content, 0) + (1.0 / (rank + 1 + _RRF_K))


def perform_hybrid_search(
    query: str,
    vectorstore: PineconeVectorStore,
    bm25_retriever: Optional[BM25Retriever],
    k: int = 3,
    fetch_k: int = 8,  # ลดจาก 15 เหลือ 8 เพื่อลดภาระ Re-ranker
    reranker: Optional[CrossEncoder] = None,
) -> List[Document]:
    """Retrieve documents using Parallel RRF and optional Cross-Encoder Re-ranking."""
    
    # 1. ค้นหาแบบคู่ขนาน (Parallel Fetching) เพื่อประหยัดเวลา
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # สั่ง Thread 1 ไปหา Pinecone
        future_vector = executor.submit(vectorstore.similarity_search, query, k=fetch_k)
        
        # สั่ง Thread 2 ไปหา BM25 (ถ้ามี)
        if bm25_retriever is not None:
            bm25_retriever.k = fetch_k
            future_bm25 = executor.submit(bm25_retriever.invoke, query)
        
        # รอรับผลลัพธ์จากทั้งคู่พร้อมกัน
        vector_docs = future_vector.result()
        bm25_docs = future_bm25.result() if bm25_retriever is not None else []

    # 2. รวมคะแนนด้วย RRF
    if bm25_retriever is None:
        fused_docs = vector_docs
    else:
        scores: Dict[str, float] = {}
        mapping: Dict[str, Document] = {}

        _compute_rrf_scores(vector_docs, scores, mapping)
        _compute_rrf_scores(bm25_docs, scores, mapping)

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        fused_docs = [mapping[content] for content, _ in sorted_docs[:fetch_k]]

    # 3. Precision Re-ranking
    if reranker is not None and fused_docs:
        pairs = [[query, doc.page_content] for doc in fused_docs]
        rerank_scores = reranker.predict(pairs)
        
        scored_docs = list(zip(fused_docs, rerank_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        fused_docs = [doc for doc, score in scored_docs]

    # ตัดส่งคืนให้ LLM แค่ k ก้อนที่ดีที่สุด
    return fused_docs[:k]