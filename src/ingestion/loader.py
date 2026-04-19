"""PDF document loader using the new smart routing pipeline.

Integrates the modular PDF-to-Markdown pipeline into the RAG ingestion flow.
"""

import logging
import os
import tempfile
from typing import List

from langchain_core.documents import Document

from src.ingestion.pdf_processor import process_pdf

logger = logging.getLogger(__name__)


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
    
    # Store markdown outputs in a dedicated directory next to the original file
    # or in a temporary directory if we prefer. For now, let's use a standard output dir.
    output_dir = os.path.join(os.path.dirname(file_path), "md_output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Route through the intelligent pipeline
        md_file_path = process_pdf(pdf_path=file_path, output_dir=output_dir)
        
        with open(md_file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
            
        # Return as a single document; the chunker will handle splitting it up
        return [Document(
            page_content=markdown_content,
            metadata={"source": file_path}
        )]
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        # Fallback to an empty list or raise depending on preference
        return []
