"""Ingestion pipeline to orchestrate loading, chunking, and indexing."""

import logging
import os
import time
from typing import List, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever

from src.ingestion.chunker import chunk_documents
from src.ingestion.loader import process_uploaded_pdf_path
from src.rag.vectorstore import save_hybrid_store

logger = logging.getLogger(__name__)

def run_ingestion_pipeline(
    file_paths: List[str],
    embedding_model: Embeddings,
    index_name: str,
    persist_dir: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Tuple[PineconeVectorStore, BM25Retriever, int, float]:
    """
    Process PDFs from paths and index them into the hybrid store.
    Returns: (vectorstore, bm25_retriever, total_chunks, processing_time)
    """
    start_time = time.time()
    
    all_docs = []
    for path in file_paths:
        all_docs.extend(process_uploaded_pdf_path(path))
        
    chunks = chunk_documents(all_docs, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("No text found in the provided documents.")

    vectorstore, bm25_retriever = save_hybrid_store(
        chunks=chunks,
        embedding_model=embedding_model,
        persist_dir=persist_dir,
        index_name=index_name,
    )
    
    ingest_time = time.time() - start_time
    return vectorstore, bm25_retriever, len(chunks), ingest_time
