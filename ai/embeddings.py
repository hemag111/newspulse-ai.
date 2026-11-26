"""Sentence-Transformer embeddings for English, Kannada, and Hindi text."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Iterable

from .config import ENGLISH_EMBED_MODEL, MULTILINGUAL_EMBED_MODEL

_MODEL_CACHE: dict[str, object] = {}
_USE_FALLBACK_HASH_EMBEDDING = False


def truncate_words(text: str, max_words: int = 200) -> str:
    """Keep title + first N words for fast, consistent embedding."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _detect_multilingual(texts: Iterable[str], language: str | None = None) -> bool:
    if language and language.lower() in {"kn", "hi", "kannada", "hindi", "multilingual"}:
        return True
    for txt in texts:
        for ch in txt:
            code = ord(ch)
            # Devanagari (Hindi: 0900-097F) or Kannada (0C80-0CFF)
            if 0x0900 <= code <= 0x097F or 0x0C80 <= code <= 0x0CFF:
                return True
    return False


def _load_sentence_transformer(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _MODEL_CACHE[model_name] = model
        return model
    except Exception:  # noqa: BLE001
        return None


def _hash_ngram_embed(text: str, dim: int = 384) -> list[float]:
    """Deterministic lightweight fallback embedding if sentence-transformers is absent."""
    vec = [0.0] * dim
    cleaned = text.lower().strip()
    if not cleaned:
        return vec
    stopwords = {
        "the", "and", "for", "that", "with", "from", "this", "into", "over",
        "across", "amid", "after", "near", "under", "their", "its", "are", "was",
    }
    tokens = re.findall(r"[\w\u0900-\u097f\u0c80-\u0cff]+", cleaned)
    weighted_features: list[tuple[str, float]] = []
    for tok in tokens:
        if tok in stopwords or len(tok) < 2:
            continue
        weighted_features.append((f"w:{tok}", 2.5))
        padded = f"<{tok}>"
        for i in range(max(1, len(padded) - 2)):
            weighted_features.append((f"c:{padded[i : i + 3]}", 0.35))
    for feat, weight in weighted_features:
        digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if (digest[4] & 1) == 0 else -1.0
        vec[idx] += sign * weight
    return l2_normalize(vec)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def average_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for vec in vectors:
        for i, val in enumerate(vec):
            acc[i] += val
    return l2_normalize([val / len(vectors) for val in acc])


def embed(texts: list[str], language: str = "en") -> list[list[float]]:
    """
    Embed a list of texts into L2-normalized 384-d vectors.
    Uses all-MiniLM-L6-v2 for English and paraphrase-multilingual-MiniLM-L12-v2 for Kannada/Hindi.
    """
    if not texts:
        return []
    prepared = [truncate_words(t or "", max_words=220) for t in texts]
    use_multi = _detect_multilingual(prepared, language=language)
    model_name = MULTILINGUAL_EMBED_MODEL if use_multi else ENGLISH_EMBED_MODEL

    if not _USE_FALLBACK_HASH_EMBEDDING:
        model = _load_sentence_transformer(model_name)
        if model is not None:
            vectors = model.encode(prepared, normalize_embeddings=True)
            return [[float(x) for x in row] for row in vectors]

    return [_hash_ngram_embed(t) for t in prepared]
