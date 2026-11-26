"""Natural-language preference parser, retrieval-grounded story Q&A, and tracked timelines."""

from __future__ import annotations

import re
from collections import defaultdict

from .config import TOPICS
from .llm import LLMUnavailableError, call_gemini, call_gemini_json

_TOPIC_SYNONYMS: dict[str, str] = {
    "space": "science",
    "isro": "science",
    "nasa": "science",
    "astronomy": "science",
    "tech": "technology",
    "ai": "technology",
    "startups": "technology",
    "startup": "technology",
    "cricket": "sports",
    "football": "sports",
    "finance": "business",
    "economy": "business",
    "markets": "business",
    "stocks": "business",
    "politics": "india",
    "geopolitics": "world",
    "movies": "entertainment",
    "cinema": "entertainment",
    "medicine": "health",
    "fitness": "health",
    "schools": "education",
    "college": "education",
    "exams": "education",
}


def _validate_prefs_dict(raw: dict) -> dict[str, list[str]]:
    valid_topics = set(TOPICS)
    cleaned: dict[str, list[str]] = {"boost": [], "reduce": [], "mute": []}
    seen: set[str] = set()
    for key in ("boost", "reduce", "mute"):
        values = raw.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            norm = str(item).strip().lower()
            norm = _TOPIC_SYNONYMS.get(norm, norm)
            if norm in valid_topics and norm not in seen:
                cleaned[key].append(norm)
                seen.add(norm)
    return cleaned


def _heuristic_parse_prefs(text: str) -> dict[str, list[str]]:
    lowered = text.lower()
    out: dict[str, list[str]] = {"boost": [], "reduce": [], "mute": []}
    clauses = re.split(r"[,.;]+|\b(?:but|however|while)\b", lowered)
    for clause in clauses:
        bucket = None
        if any(w in clause for w in ("mute", "hide", "block", "stop showing", "no more")):
            bucket = "mute"
        elif any(w in clause for w in ("less", "reduce", "fewer", "lower", "dial back")):
            bucket = "reduce"
        elif any(w in clause for w in ("more", "boost", "prefer", "focus", "increase", "like")):
            bucket = "boost"
        if bucket is None:
            continue
        for candidate in list(TOPICS) + list(_TOPIC_SYNONYMS.keys()):
            if re.search(rf"\b{re.escape(candidate)}\b", clause):
                mapped = _TOPIC_SYNONYMS.get(candidate, candidate)
                if mapped not in out[bucket]:
                    out[bucket].append(mapped)
    return _validate_prefs_dict(out)


def parse_prefs(text: str) -> dict[str, list[str]]:
    """
    Convert free-form user preferences (e.g., 'less politics, more space and startups')
    into strict JSON: {"boost": [...], "reduce": [...], "mute": [...]}.
    Falls back safely on malformed responses.
    """
    if not text or not text.strip():
        return {"boost": [], "reduce": [], "mute": []}

    prompt = (
        f"Available canonical topics: {', '.join(TOPICS)}.\n"
        "Map synonyms (e.g., space -> science, startups -> technology, markets -> business, politics -> india/world) "
        "to the canonical topics above.\n"
        'Return ONLY valid JSON: {"boost": [...], "reduce": [...], "mute": [...]}.\n'
        f'User instruction: "{text.strip()}"'
    )
    try:
        parsed = call_gemini_json(
            prompt,
            system_instruction="Convert natural-language news preferences into strict topic JSON.",
            cache_namespace="parse_prefs",
            cache_key=text.strip().lower(),
        )
        return _validate_prefs_dict(parsed)
    except (LLMUnavailableError, ValueError):
        return _heuristic_parse_prefs(text)


def ask(question: str, articles: list[dict]) -> str:
    """
    Retrieval-grounded Q&A over a cluster's articles.
    Answers strictly from the provided article texts and cites the source,
    or replies 'Not in these articles.' if the answer is absent.
    """
    if not question or not question.strip():
        return "Please enter a question about this story."
    if not articles:
        return "Not in these articles."

    context_blocks = []
    for art in articles[:8]:
        src = art.get("source", "Unknown Source")
        title = art.get("title", "")
        body = art.get("body") or art.get("summary_long") or art.get("summary_short") or ""
        context_blocks.append(f"[Source: {src}] {title}\n{body[:2500]}")

    context_str = "\n\n---\n\n".join(context_blocks)
    prompt = (
        f"Articles:\n{context_str}\n\n"
        f"Question: {question.strip()}\n\n"
        "Instructions: Answer using ONLY facts explicitly stated in the articles above. "
        "Include the supporting source name in brackets like [Source: Name]. "
        "If the articles do not contain the answer, respond strictly with: Not in these articles."
    )
    try:
        return call_gemini(
            prompt,
            system_instruction=(
                "You are a strict retrieval-grounded assistant. Never rely on outside knowledge."
            ),
        )
    except LLMUnavailableError:
        # Lightweight keyword-overlap fallback when offline
        q_words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
        best_sentence = ""
        best_source = ""
        best_hits = 0
        for art in articles:
            text = art.get("body") or art.get("summary_long") or art.get("summary_short") or ""
            for sent in re.split(r"(?<=[.!?])\s+", text):
                hits = sum(1 for w in q_words if w in sent.lower())
                if hits > best_hits:
                    best_hits = hits
                    best_sentence = sent.strip()
                    best_source = art.get("source", "Source")
        if best_hits >= 2 and best_sentence:
            return f"{best_sentence} [Source: {best_source}]"
        return "Not in these articles."


def track_timeline(articles: list[dict], last_seen: str | None = None) -> list[dict]:
    """
    Group tracked-topic articles by publication date and produce one concise line per day,
    sorted chronologically from oldest to newest.
    """
    if not articles:
        return []

    by_date: dict[str, list[dict]] = defaultdict(list)
    for art in articles:
        pub = str(art.get("published_at") or "")
        day = pub[:10] if len(pub) >= 10 else "Unknown"
        by_date[day].append(art)

    timeline: list[dict] = []
    for day in sorted(by_date.keys()):
        day_articles = by_date[day]
        headlines = [a.get("title", "") for a in day_articles if a.get("title")]
        sources = sorted({str(a.get("source", "Source")) for a in day_articles})
        is_new = bool(last_seen and day > last_seen[:10])

        if len(headlines) == 1:
            day_summary = headlines[0]
        else:
            joined = "; ".join(headlines[:4])
            try:
                day_summary = call_gemini(
                    f"Write ONE concise factual sentence summarizing these {day} headlines:\n{joined}",
                    system_instruction="Produce a single neutral 1-sentence news timeline entry.",
                    cache_namespace="timeline_day",
                    cache_key=f"{day}:{joined}",
                )
            except LLMUnavailableError:
                day_summary = f"{headlines[0]} (+{len(headlines) - 1} related report{'s' if len(headlines) > 2 else ''})"

        timeline.append(
            {
                "date": day,
                "summary": day_summary,
                "article_count": len(day_articles),
                "sources": sources,
                "article_ids": [a.get("id") for a in day_articles],
                "is_new_since_last_seen": is_new,
            }
        )

    return timeline


def summarize_what_changed(articles: list[dict], last_seen: str | None) -> str:
    """Summarize only developments published after `last_seen` for the tracker banner."""
    if not articles:
        return "No tracked updates yet."
    if not last_seen:
        return f"Tracking {len(articles)} articles across {len({a.get('source') for a in articles})} sources."

    newer = [a for a in articles if str(a.get("published_at") or "") > last_seen]
    if not newer:
        return "No new developments since your last visit."

    headlines = "\n".join(f"- [{a.get('source', 'Source')}] {a.get('title', '')}" for a in newer[:6])
    try:
        return call_gemini(
            f"Summarize what changed since the user's last visit in 2 short factual sentences:\n{headlines}",
            system_instruction="Summarize only the listed new developments concisely.",
            cache_namespace="what_changed",
            cache_key=f"{last_seen}:{headlines}",
        )
    except LLMUnavailableError:
        return f"{len(newer)} new article(s) since last visit, led by: {newer[0].get('title', '')}"
