"""
Local embeddings using sentence-transformers.
Lazily loads the model on first call to avoid slow import-time init.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

import config

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _model = SentenceTransformer(
            config.EMBEDDING_MODEL,
            local_files_only=True,
        )
        log.info("Embedding model loaded")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Encode a list of strings into dense vectors.

    Returns
    -------
    np.ndarray of shape (len(texts), EMBEDDING_DIMENSION), dtype float32
    """
    model = _get_model()
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(vecs, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, dim)."""
    return embed_texts([query])
