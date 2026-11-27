from __future__ import annotations

from functools import lru_cache
from typing import List
import numpy as np

from .config import settings


def _e5_format(text: str, kind: str) -> str:
    """
    E5 recommends prefixes: "query:" / "passage:"
    """
    prefix = "query:" if kind == "query" else "passage:"
    text = text.strip()
    return f"{prefix} {text}" if text else prefix


def _sanitize_text(text: str) -> str:
    """
    Defensive preprocessing:
    - handle None
    - trim
    - clip very long texts to avoid hidden truncation
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    # hard clip by characters (safe, tokenizer-agnostic)
    max_chars = getattr(settings, "MAX_EMBEDDING_CHARS", 4000)
    return text[:max_chars]


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        settings.EMBEDDING_MODEL_NAME,
        device=settings.DEVICE,
    )


def embed_texts(texts: List[str], kind: str = "passage") -> np.ndarray:
    """
    Embed list of texts using E5-style prefixes.
    Always returns np.ndarray of shape (n, d), dtype float32.
    """
    model = get_model()

    sanitized = [_sanitize_text(t) for t in texts]
    formatted = [_e5_format(t, kind) for t in sanitized]

    emb = model.encode(
        formatted,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(emb, dtype=np.float32)
