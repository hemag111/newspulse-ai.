"""Three-signal clickbait detection: heuristics + headline-body divergence + optional LLM."""

from __future__ import annotations

import hashlib
import re

from .cache import get_cache
from .config import CLICKBAIT_FLAG_THRESHOLD
from .embeddings import cosine_similarity, embed
from .llm import LLMUnavailableError, call_gemini_json

SENSATIONAL_PHRASES = (
    "you won't believe",
    "you will not believe",
    "what happened next",
    "shocking",
    "mind-blowing",
    "jaw-dropping",
    "doctors hate",
    "one weird trick",
    "this one trick",
    "secret they don't want",
    "broke the internet",
    "gone wrong",
    "gone viral",
    "nobody expected",
    "everything changes",
    "you need to see",
    "here's why",
    "exposed",
    "unbelievable",
    "insane",
    "bombshell",
)


def _heuristic_signal(title: str) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if not title:
        return 0.0, reasons

    score = 0.0
    words = re.findall(r"[A-Za-z]+", title)
    if words:
        all_caps = [w for w in words if len(w) >= 3 and w.isupper()]
        caps_ratio = len(all_caps) / len(words)
        if caps_ratio >= 0.35:
            score += 0.35
            reasons.append("excessive ALL-CAPS words")
        elif len(all_caps) >= 2:
            score += 0.18
            reasons.append("multiple ALL-CAPS emphasis words")

    exclam_count = title.count("!")
    question_count = title.count("?")
    if exclam_count >= 2 or "!?" in title or "?!" in title:
        score += 0.28
        reasons.append("sensational punctuation (!! / ?!)")
    elif exclam_count == 1 or question_count >= 2:
        score += 0.12
        reasons.append("emphatic punctuation")

    lowered = title.lower()
    phrase_hits = [p for p in SENSATIONAL_PHRASES if p in lowered]
    if phrase_hits:
        score += min(0.45, 0.25 + 0.10 * (len(phrase_hits) - 1))
        reasons.append(f"sensational phrase ({phrase_hits[0]})")

    # Numbered curiosity bait like "7 Reasons Why..."
    if re.match(r"^\s*\d+\s+(?:reasons|things|secrets|ways|facts)\b", lowered):
        score += 0.15
        reasons.append("listicle curiosity hook")

    return min(1.0, score), reasons


def _llm_clickbait_check(title: str, body: str, cache_key: str) -> float | None:
    prompt = (
        "Rate how clickbaity or misleading this headline is relative to its body on a scale "
        "from 0.0 (straightforward, accurate journalism) to 1.0 (pure clickbait / exaggerated).\n"
        'Return strict JSON: {"score": <float between 0.0 and 1.0>}\n\n'
        f"Headline: {title}\nBody: {body[:1800]}"
    )
    try:
        res = call_gemini_json(
            prompt,
            system_instruction="You evaluate journalistic headlines objectively.",
            cache_namespace="clickbait_llm",
            cache_key=cache_key,
        )
        val = float(res.get("score", 0.5))
        return max(0.0, min(1.0, val))
    except (LLMUnavailableError, ValueError, TypeError):
        return None


def analyze_clickbait(
    title: str,
    body: str,
    use_llm: bool = False,
    article_id: str | None = None,
) -> dict:
    """Return full 3-signal breakdown plus final clickbait score and boolean flag."""
    cache_key = article_id or hashlib.sha256(f"{title}||{body[:500]}".encode("utf-8")).hexdigest()
    cache = get_cache()
    if not use_llm:
        cached = cache.get("clickbait", cache_key)
        if isinstance(cached, dict) and "score" in cached:
            return cached

    heur_score, heur_reasons = _heuristic_signal(title)

    # Signal 2: semantic divergence between headline and body opening
    divergence = 0.0
    if title.strip() and body.strip():
        vecs = embed([title, body[:1200]])
        sim = cosine_similarity(vecs[0], vecs[1])
        # Typical news headline-to-body similarity is ~0.45-0.85; lower indicates mismatch
        divergence = max(0.0, min(1.0, (0.58 - sim) / 0.45))

    base_score = 0.62 * heur_score + 0.38 * divergence
    llm_val: float | None = None

    # Signal 3: LLM check when requested or in ambiguous range
    if use_llm or (0.38 <= base_score <= 0.70 and len(body.strip()) > 40):
        llm_val = _llm_clickbait_check(title, body, cache_key=cache_key)

    if llm_val is not None:
        final_score = 0.45 * heur_score + 0.25 * divergence + 0.30 * llm_val
    else:
        final_score = base_score

    final_score = round(max(0.0, min(1.0, final_score)), 3)
    result = {
        "score": final_score,
        "flagged": final_score > CLICKBAIT_FLAG_THRESHOLD,
        "signals": {
            "heuristic": round(heur_score, 3),
            "headline_body_divergence": round(divergence, 3),
            "llm_check": round(llm_val, 3) if llm_val is not None else None,
        },
        "reasons": heur_reasons,
    }
    cache.set("clickbait", cache_key, result)
    return result


def clickbait_score(title: str, body: str, use_llm: bool = False) -> float:
    """Return clickbait probability score in [0.0, 1.0]. Flag when > 0.6."""
    return float(analyze_clickbait(title, body, use_llm=use_llm)["score"])
