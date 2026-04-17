"""Thai-aware document chunking pipeline.

Applies text normalisation (PDF text cleaning, boundary insertion)
and splits the result into overlapping chunks using LangChain's 
RecursiveCharacterTextSplitter while preserving document structure.
"""

import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    # 1. Fix broken Thai vowels
    text = text.replace("ํา", "ำ")

    # 2. Insert boundary between EN↔TH characters (helps readability & search)
    text = re.sub(r"([a-zA-Z])([ก-๙])", r"\1 \2", text)
    text = re.sub(r"([ก-๙])([a-zA-Z])", r"\1 \2", text)

    # 3. Normalise colons
    text = re.sub(r"\s*:\s*", ":", text)

    # 4. Clean up whitespace but PRESERVE newlines
    # Replace multiple horizontal spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into just 2 newlines (standard paragraph break)
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
    """Split documents into overlapping chunks.

    Args:
        documents: LangChain ``Document`` objects.
        chunk_size: Maximum number of CHARACTERS per chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        A flat list of chunked ``Document`` objects.
    """
    # Preprocess every page
    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)

    # ใช้ตัวนับความยาวเป็น Character ธรรมดา เพื่อให้กะขนาด Chunk ได้แม่นยำ
    # และใช้ Separator ที่เข้ากับภาษาไทย (มักใช้ Space เว้นวรรคประโยค)
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