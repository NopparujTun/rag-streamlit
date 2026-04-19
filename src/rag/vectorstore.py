"""Hybrid vector + keyword store with Reciprocal Rank Fusion (RRF) and Re-ranking.

Manages a dual-retrieval system:
- **Pinecone** for dense vector similarity search (cloud).
- **BM25Retriever** for sparse keyword search (persisted locally as pickle).

Results are combined using RRF scoring, over-fetched, and optionally 
re-ranked using a Cross-Encoder for maximum precision.
"""

import concurrent.futures
import logging
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# RRF smoothing constant (standard value from the original RRF paper).
DEFAULT_RRF_K = 60
BM25_FILENAME = "bm25_index.pkl"


class HybridRetriever:
    """A managed hybrid retriever combining dense and sparse search."""

    def __init__(
        self,
        vectorstore: PineconeVectorStore,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoder] = None,
        rrf_k: int = DEFAULT_RRF_K,
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        k: int = 3,
        fetch_k: int = 8,
    ) -> List[Document]:
        """Perform a hybrid search with optional re-ranking."""
        
        # 1. Parallel Fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self.vectorstore.similarity_search, query, k=fetch_k)
            
            if self.bm25_retriever is not None:
                self.bm25_retriever.k = fetch_k
                future_bm25 = executor.submit(self.bm25_retriever.invoke, query)
            else:
                future_bm25 = None
            
            vector_docs = future_vector.result()
            bm25_docs = future_bm25.result() if future_bm25 else []

        # 2. Reciprocal Rank Fusion
        if not bm25_docs:
            fused_docs = vector_docs
        else:
            fused_docs = self._apply_rrf(vector_docs, bm25_docs, fetch_k)

        # 3. Precision Re-ranking
        if self.reranker is not None and fused_docs:
            fused_docs = self._apply_reranking(query, fused_docs)

        return fused_docs[:k]

    def _apply_rrf(
        self, 
        vector_docs: List[Document], 
        bm25_docs: List[Document], 
        fetch_k: int
    ) -> List[Document]:
        """Combine results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        mapping: Dict[str, Document] = {}

        def _compute(docs):
            for rank, doc in enumerate(docs):
                content = doc.page_content
                mapping[content] = doc
                scores[content] = scores.get(content, 0) + (1.0 / (rank + 1 + self.rrf_k))

        _compute(vector_docs)
        _compute(bm25_docs)

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [mapping[content] for content, _ in sorted_results[:fetch_k]]

    def _apply_reranking(self, query: str, docs: List[Document]) -> List[Document]:
        """Re-rank documents using the Cross-Encoder model."""
        pairs = [[query, doc.page_content] for doc in docs]
        rerank_scores = self.reranker.predict(pairs)
        
        scored_docs = sorted(zip(docs, rerank_scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]


def save_hybrid_store(
    chunks: List[Document],
    embedding_model: Embeddings,
    persist_dir: str,
    index_name: str,
) -> Tuple[PineconeVectorStore, BM25Retriever]:
    """Index document chunks into Pinecone and save a local BM25 index."""
    os.makedirs(persist_dir, exist_ok=True)

    logger.info("Uploading %d chunks to Pinecone index '%s'", len(chunks), index_name)
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_path = os.path.join(persist_dir, BM25_FILENAME)
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

    bm25_path = os.path.join(persist_dir, BM25_FILENAME)
    bm25_retriever = None
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as fh:
            bm25_retriever = pickle.load(fh)
        logger.info("Loaded BM25 index from %s", bm25_path)
    else:
        logger.warning("BM25 index not found at %s — keyword search disabled", bm25_path)

    return vectorstore, bm25_retriever


def clear_hybrid_store(vectorstore: PineconeVectorStore, persist_dir: str) -> None:
    """Clear all vectors from Pinecone and delete local BM25 directory."""
    logger.info("Clearing Knowledge Base: Vector database and BM25.")
    
    try:
        vectorstore.delete(delete_all=True)
        logger.info("Deleted all vectors from Pinecone.")
    except Exception as exc:
        logger.error("Failed to delete from Pinecone: %s", exc)

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        logger.info("Cleared BM25 index directory: %s", persist_dir)


def get_reranker(model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> CrossEncoder:
    """Load a Cross-Encoder model for precision re-ranking."""
    logger.info("Loading Re-ranker model: %s (device=%s)", model_name, device)
    return CrossEncoder(model_name, max_length=512, device=device)
