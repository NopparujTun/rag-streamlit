"""Embedding model loader.

Provides a factory function for HuggingFace sentence-transformer models
used to embed document chunks and queries for vector search.
"""

import logging
from typing import Dict

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str, device: str) -> HuggingFaceEmbeddings:
    """Load and return a HuggingFace embedding model.

    Uses L2-normalised embeddings by default, which is required for
    cosine-similarity based vector stores like Pinecone.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"BAAI/bge-m3"``).
        device: Torch device string (``"cpu"`` or ``"cuda"``).

    Returns:
        A configured ``HuggingFaceEmbeddings`` instance.
    """
    logger.info("Loading embedding model: %s (device=%s)", model_name, device)

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Embedding model loaded successfully")
    return embeddings