"""Grounded article and cluster summarization powered by Gemini with SQLite caching."""

from __future__ import annotations

import hashlib
import re

from .cache import get_cache
from .llm import LLMUnavailableError, call_gemini_json

SUMMARIZE_SYSTEM_INSTRUCTION = (
    "You are a factual news editor. Use ONLY facts explicitly stated in the article text. "
    "Never invent names, numbers, dates, or external background information. "
    "Return strict JSON with keys 'short' and 'long':\n"
    "- 'short': exactly 3 concise bullet points separated by newlines, each starting with '- '.\n"
    "- 'long': one cohesive factual summary of approximately 90-110 words."
)


def _extractive_fallback(text: str) -> dict[str, str]:
    """Deterministic sentence-based fallback when Gemini API key is not present."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return {"short": "- No article content provided.", "long": "No article content provided."}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", clean) if len(s.strip()) > 15]
    if not sentences:
        sentences = [clean[:180]]
    bullets = [f"- {s}" for s in sentences[:3]]
    while len(bullets) < 3 and sentences:
        bullets.append(f"- {sentences[len(bullets) % len(sentences)]}")
    long_words = " ".join(sentences[:5]).split()[:100]
    return {
        "short": "\n".join(bullets[:3]),
        "long": " ".join(long_words),
    }


def summarize(text: str, article_id: str | None = None) -> dict[str, str]:
    """
    Summarize an article into {"short": 3 bullets, "long": ~100-word summary}.
    Cached by article_id (or content hash) so the LLM is never called twice for the same article.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {"short": "- Empty article.", "long": "Empty article."}

    cache_key = article_id or hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    cache = get_cache()
    cached = cache.get("summaries", cache_key)
    if isinstance(cached, dict) and "short" in cached and "long" in cached:
        return {"short": str(cached["short"]), "long": str(cached["long"])}

    prompt = f"Summarize the following news article strictly from the provided text:\n\n{cleaned[:8000]}"
    try:
        result = call_gemini_json(
            prompt,
            system_instruction=SUMMARIZE_SYSTEM_INSTRUCTION,
            cache_namespace="summaries_raw",
            cache_key=cache_key,
        )
        short_val = result.get("short", "")
        if isinstance(short_val, list):
            short_str = "\n".join(
                item if str(item).startswith("- ") else f"- {item}" for item in short_val[:3]
            )
        else:
            short_str = str(short_val).strip()
        long_str = str(result.get("long", "")).strip()
        if not short_str or not long_str:
            raise ValueError("Incomplete summary fields from Gemini.")
        out = {"short": short_str, "long": long_str}
        cache.set("summaries", cache_key, out)
        return out
    except LLMUnavailableError:
        return _extractive_fallback(cleaned)


def merge_cluster_summaries(articles: list[dict]) -> dict[str, str]:
    """Merge member summaries of a story cluster into one combined neutral overview."""
    if not articles:
        return {"short": "", "long": ""}
    if len(articles) == 1:
        art = articles[0]
        if art.get("summary_short") and art.get("summary_long"):
            return {"short": art["summary_short"], "long": art["summary_long"]}
        return summarize(f"{art.get('title', '')}. {art.get('body', '')}", article_id=art.get("id"))

    member_ids = ",".join(sorted(str(a.get("id", idx)) for idx, a in enumerate(articles)))
    cache = get_cache()
    cached = cache.get("cluster_summaries", member_ids)
    if isinstance(cached, dict) and "short" in cached and "long" in cached:
        return {"short": str(cached["short"]), "long": str(cached["long"])}

    combined_blocks = []
    for art in articles[:8]:
        src = art.get("source", "Source")
        summary_text = art.get("summary_long") or art.get("summary_short") or art.get("body") or art.get("title", "")
        combined_blocks.append(f"[{src}] {art.get('title', '')}: {summary_text}")

    joined = "\n\n".join(combined_blocks)
    prompt = (
        "Combine these multi-source reports of the same news event into a single unified, "
        "non-redundant summary using ONLY facts stated below:\n\n" + joined[:9000]
    )
    try:
        result = call_gemini_json(
            prompt,
            system_instruction=SUMMARIZE_SYSTEM_INSTRUCTION,
            cache_namespace="cluster_summaries_raw",
            cache_key=member_ids,
        )
        out = {
            "short": str(result.get("short", "")).strip(),
            "long": str(result.get("long", "")).strip(),
        }
        if out["short"] and out["long"]:
            cache.set("cluster_summaries", member_ids, out)
            return out
    except LLMUnavailableError:
        pass

    return _extractive_fallback(joined)
