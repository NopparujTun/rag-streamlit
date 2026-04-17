"""Unit tests for the document ingestion pipeline.

Tests the chunking logic to verify that documents are correctly split
into overlapping chunks while preserving metadata.
"""

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents


class TestChunkDocuments:
    """Tests for :func:`chunk_documents`."""

    def test_splits_long_text_into_multiple_chunks(self) -> None:
        """Verify that a document longer than ``chunk_size`` is split."""
        mock_docs = [
            Document(
                page_content=(
                    "นี่คือข้อความยาวๆ ที่เราต้องการจะทดสอบระบบการหั่นคำ"
                    "ของ LangChain ว่าทำงานได้ถูกต้องหรือไม่"
                ),
                metadata={"page": 1},
            )
        ]

        chunks = chunk_documents(mock_docs, chunk_size=20, chunk_overlap=5)

        assert len(chunks) > 1, "Expected multiple chunks from a long document"

    def test_preserves_metadata(self) -> None:
        """Verify that chunk metadata is inherited from the source document."""
        mock_docs = [
            Document(
                page_content=(
                    "นี่คือข้อความยาวๆ ที่เราต้องการจะทดสอบระบบการหั่นคำ"
                    "ของ LangChain ว่าทำงานได้ถูกต้องหรือไม่"
                ),
                metadata={"page": 1},
            )
        ]

        chunks = chunk_documents(mock_docs, chunk_size=20, chunk_overlap=5)

        assert chunks[0].metadata["page"] == 1