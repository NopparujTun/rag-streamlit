"""Ingestion pipeline to orchestrate loading, chunking, and indexing."""

import logging
import os
import re
import tempfile
import time
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_processor import process_pdf
from src.rag.retrieval import save_hybrid_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def clean_pdf_text(text: str) -> str:
    """Normalise mixed Thai/English text extracted from a PDF.

    Processing steps:
    1. Fix Thai broken vowel ``ํา`` → ``ำ``.
    2. Insert space between English and Thai character boundaries.
    3. Normalise colons.
    4. Clean up spaces but PRESERVE newlines (\n).
    """
    text = text.replace("ํา", "ำ")
    text = re.sub(r"([a-zA-Z])([ก-๙])", r"\1 \2", text)
    text = re.sub(r"([ก-๙])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\s*:\s*", ":", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Split documents into overlapping chunks."""
    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "  ", " ", ""],
        length_function=len, 
    )

    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d documents → %d chunks (size=%d chars, overlap=%d chars)",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def process_uploaded_pdf(file_bytes: bytes) -> List[Document]:
    """Parse PDF bytes by writing to a temporary file, then loading."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name
        return process_uploaded_pdf_path(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_uploaded_pdf_path(file_path: str) -> List[Document]:
    """Parse a PDF from a path using the Smart Markdown Pipeline."""
    logger.info("Loading PDF via Smart Pipeline: %s", file_path)
    output_dir = os.path.join(os.path.dirname(file_path), "md_output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        md_file_path = process_pdf(pdf_path=file_path, output_dir=output_dir)
        with open(md_file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
        return [Document(
            page_content=markdown_content,
            metadata={"source": file_path}
        )]
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return []

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

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
