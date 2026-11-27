from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

from .embeddings import embed_texts
from .config import settings


# ---------------------------
# math utils
# ---------------------------
def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize vectors to make cosine similarity valid.
    Works for 1D (d,) and 2D (n,d) arrays.
    """
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def cosine_scores(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """
    vec: (d,)
    mat: (n, d)
    Both must be L2-normalized. Then cosine = dot product.
    """
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    mat = np.asarray(mat, dtype=np.float32)

    if mat.ndim != 2:
        raise ValueError(f"mat must be 2D (n,d), got shape={mat.shape}")
    if vec.ndim != 1:
        raise ValueError(f"vec must be 1D (d,), got shape={vec.shape}")
    if mat.shape[1] != vec.shape[0]:
        raise ValueError(f"dim mismatch: mat.shape={mat.shape}, vec.shape={vec.shape}")

    sims = mat @ vec  # (n,)
    sims = np.asarray(sims, dtype=np.float32).reshape(-1)
    sims = np.nan_to_num(sims, nan=-1e9, posinf=1e9, neginf=-1e9)
    return sims


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Stable softmax for UI-style confidence distribution (not a calibrated probability).
    Returns probs shape (k,).
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x.astype(np.float32)

    t = max(float(temperature), 1e-6)
    z = x / t
    z = z - float(np.max(z))
    e = np.exp(z).astype(np.float32)
    denom = float(np.sum(e))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(e, dtype=np.float32)
    return (e / denom).astype(np.float32)


# ---------------------------
# results contracts
# ---------------------------
@dataclass(frozen=True)
class RawClassificationResult:
    """
    ML-only output (NO threshold, NO 'Не определена' logic).
    Suitable for tests and reuse.
    """
    best_label: str
    best_similarity: float

    topN_labels: List[str]
    topN_similarity: List[float]
    topN_confidence: List[float]

    scores_by_category: Dict[str, float]


@dataclass(frozen=True)
class DecisionResult:
    """
    Business decision layer for MVP ('Не определена' handling).
    """
    predicted_label: str
    is_undefined: bool
    max_similarity: float
    threshold: float


# ---------------------------
# ML-only classify
# ---------------------------
def classify(
    email_text: str,
    categories: List[str],
    top_n: Optional[int] = None,
) -> RawClassificationResult:
    """
    Computes similarities and returns raw ranking.
    Does NOT decide 'Не определена' — that is handled by make_decision().
    """
    k = int(top_n or settings.TOP_N_DEFAULT)

    if not categories:
        return RawClassificationResult(
            best_label="",
            best_similarity=0.0,
            topN_labels=[],
            topN_similarity=[],
            topN_confidence=[],
            scores_by_category={},
        )

    # Embed email as query, categories as passages (E5-like)
    email_vec = embed_texts([email_text], kind="query")[0]  # (d,)
    cat_vecs = embed_texts(categories, kind="passage")      # (n, d)

    # Defensive normalization (safe even if embed_texts already normalizes)
    email_vec = _l2_normalize(email_vec, axis=0)
    cat_vecs = _l2_normalize(cat_vecs, axis=1)

    sims = cosine_scores(email_vec, cat_vecs)  # (n,)

    # Full map for diagnostics/debug
    scores_by_category = {cat: float(sim) for cat, sim in zip(categories, sims.tolist())}

    # Sort and take top-k
    order = np.argsort(-sims)
    topk = max(1, min(int(k), len(categories)))
    top_idx = order[:topk]

    top_labels = [categories[int(i)] for i in top_idx.tolist()]
    top_sims = [float(sims[int(i)]) for i in top_idx.tolist()]

    probs = softmax(np.array(top_sims, dtype=np.float32), temperature=settings.SOFTMAX_TEMPERATURE)
    top_conf = [float(p) for p in probs.tolist()]

    best_i = int(order[0])
    best_label = categories[best_i]
    best_sim = float(sims[best_i])

    return RawClassificationResult(
        best_label=best_label,
        best_similarity=best_sim,
        topN_labels=top_labels,
        topN_similarity=top_sims,
        topN_confidence=top_conf,
        scores_by_category=scores_by_category,
    )


# ---------------------------
# MVP business rule ('Не определена')
# ---------------------------
def make_decision(
    raw: RawClassificationResult,
    threshold: Optional[float] = None,
    undefined_label: str = "Не определена",
) -> DecisionResult:
    """
    Central MVP rule:
      if best_similarity < threshold -> 'Не определена'
    """
    thr = float(threshold if threshold is not None else settings.THRESHOLD_UNDEFINED)

    if not raw.best_label:
        return DecisionResult(
            predicted_label=undefined_label,
            is_undefined=True,
            max_similarity=float(raw.best_similarity or 0.0),
            threshold=thr,
        )

    best_sim = float(raw.best_similarity)
    is_undef = best_sim < thr
    predicted = undefined_label if is_undef else raw.best_label

    return DecisionResult(
        predicted_label=predicted,
        is_undefined=bool(is_undef),
        max_similarity=best_sim,
        threshold=thr,
    )


# ---------------------------
# serialization helpers
# ---------------------------
def raw_result_to_dict(res: RawClassificationResult) -> Dict[str, Any]:
    return asdict(res)


def decision_result_to_dict(res: DecisionResult) -> Dict[str, Any]:
    return asdict(res)
