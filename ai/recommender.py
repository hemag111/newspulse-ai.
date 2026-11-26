"""Personalized diversity-aware ranking, user vector updates, honest reasons, and bubble score."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timezone

from .config import (
    DIVERSITY_SLOT_INTERVAL,
    EVENT_STEP_WEIGHTS,
    RANK_WEIGHTS,
    TOPICS,
    TOPIC_DESCRIPTIONS,
)
from .embeddings import average_vectors, cosine_similarity, embed, l2_normalize


def build_user_vector(preferred_topics: list[str], language: str = "en") -> list[float]:
    """Cold-start user vector: normalized average of chosen topic description embeddings."""
    valid = [t.lower() for t in preferred_topics if t.lower() in TOPIC_DESCRIPTIONS]
    if not valid:
        valid = ["technology", "india", "science"]
    descriptions = [TOPIC_DESCRIPTIONS[t] for t in valid]
    vecs = embed(descriptions, language=language)
    return average_vectors(vecs)


def update_user_vector(
    user_vector: list[float],
    article_vector: list[float],
    event_type: str,
) -> list[float]:
    """
    Update user interest vector on click, like, save, or skip:
      click: normalize(0.90*u + 0.10*a)
      like:  normalize(0.75*u + 0.25*a)
      save:  normalize(0.70*u + 0.30*a)
      skip:  normalize(1.00*u - 0.12*a)
    """
    if not user_vector:
        return l2_normalize(article_vector) if article_vector else []
    if not article_vector or len(user_vector) != len(article_vector):
        return l2_normalize(user_vector)

    alpha, beta = EVENT_STEP_WEIGHTS.get(event_type.lower(), (0.90, 0.10))
    updated = [alpha * u + beta * a for u, a in zip(user_vector, article_vector)]
    return l2_normalize(updated)


def _recency_score(published_at: str | None, now: datetime | None = None) -> float:
    if not published_at:
        return 0.5
    ref = now or datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5
    age_hours = max(0.0, (ref - dt).total_seconds() / 3600.0)
    # Exponential decay with ~36h half-life
    return math.exp(-age_hours / 36.0)


def make_reason(user: dict, article: dict) -> str:
    """
    Generate an honest, template-based 'Why this?' explanation directly from the
    computed score components rather than ungrounded LLM text.
    """
    topic = str(article.get("topic", "general")).title()
    source = str(article.get("source", "this publisher"))
    comp = article.get("score_components") or {}

    if comp.get("is_diversity_slot"):
        return f"Broadening your feed with {topic} coverage outside your usual topics."

    sim = float(comp.get("similarity", 0.0))
    rec = float(comp.get("recency", 0.0))
    src_pref = float(comp.get("source", 0.0))
    boosted = [t.lower() for t in (user.get("boosted_topics") or [])]
    preferred_sources = [s.lower() for s in (user.get("preferred_sources") or [])]

    if str(article.get("topic", "")).lower() in boosted:
        return f"Boosted by your preference for {topic} from {source}."

    if src_pref >= 0.8 and source.lower() in preferred_sources and sim >= 0.45:
        return f"Matches your interest in {topic} and comes from {source}, one of your preferred sources."

    if sim >= max(rec, src_pref) * 0.85 and sim > 0.25:
        return f"Matches your reading history and interest in {topic}."

    if src_pref >= 0.75:
        return f"From {source}, a publication you frequently read."

    if rec >= 0.75:
        return f"Fresh {topic} reporting from {source} published recently."

    return f"Relevant {topic} story balanced with recency and source diversity."


def rank(
    user_vector: list[float],
    articles: list[dict],
    prefs: dict | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """
    Score and rank candidate articles using:
      score = 0.50*similarity + 0.20*recency + 0.15*source_pref + 0.15*diversity_bonus
    Drops muted topics and reserves 1 in 5 slots for topics outside the user's dominant reads.
    """
    prefs = prefs or {}
    muted = {t.lower() for t in (prefs.get("muted_topics") or [])}
    preferred_topics = [t.lower() for t in (prefs.get("topics") or prefs.get("preferred_topics") or [])]
    boosted_topics = {t.lower() for t in (prefs.get("boost") or prefs.get("boosted_topics") or [])}
    reduced_topics = {t.lower() for t in (prefs.get("reduce") or prefs.get("reduced_topics") or [])}
    preferred_sources = {s.lower() for s in (prefs.get("sources") or prefs.get("preferred_sources") or [])}
    recent_topics = [t.lower() for t in (prefs.get("recent_topics") or [])]

    # Identify top-2 dominant topics for diversity bonus calculation
    if recent_topics:
        dominant_topics = {t for t, _ in Counter(recent_topics).most_common(2)}
    else:
        dominant_topics = set(preferred_topics[:2])

    candidates: list[dict] = []
    missing_embed_indices: list[int] = []
    missing_texts: list[str] = []

    for art in articles:
        topic = str(art.get("topic", "")).lower()
        if topic in muted:
            continue
        item = dict(art)
        if not isinstance(item.get("embedding"), list) or not item["embedding"]:
            missing_embed_indices.append(len(candidates))
            missing_texts.append(f"{item.get('title', '')}. {item.get('summary_short', '')}")
        candidates.append(item)

    if missing_texts and user_vector:
        vecs = embed(missing_texts)
        for idx, vec in zip(missing_embed_indices, vecs):
            candidates[idx]["embedding"] = vec

    scored: list[dict] = []
    for item in candidates:
        topic = str(item.get("topic", "")).lower()
        source = str(item.get("source", "")).lower()

        if user_vector and item.get("embedding"):
            raw_sim = cosine_similarity(user_vector, item["embedding"])
            sim = max(0.0, min(1.0, (raw_sim + 0.15) / 0.85))
        elif topic in preferred_topics:
            sim = 0.80
        else:
            sim = 0.35

        if topic in boosted_topics:
            sim = min(1.0, sim + 0.20)
        elif topic in reduced_topics:
            sim = max(0.0, sim - 0.25)

        rec = _recency_score(item.get("published_at"), now=now)
        src_pref = 1.0 if source in preferred_sources else 0.35
        div_bonus = 1.0 if (topic and topic not in dominant_topics) else 0.15

        total_score = (
            RANK_WEIGHTS["similarity"] * sim
            + RANK_WEIGHTS["recency"] * rec
            + RANK_WEIGHTS["source"] * src_pref
            + RANK_WEIGHTS["diversity"] * div_bonus
        )

        # Mild demotion for likely clickbait so factual reporting ranks higher
        cb = float(item.get("clickbait_score") or 0.0)
        if cb > 0.6:
            total_score *= 0.85

        item["score"] = round(total_score, 4)
        item["score_components"] = {
            "similarity": round(sim, 3),
            "recency": round(rec, 3),
            "source": round(src_pref, 3),
            "diversity": round(div_bonus, 3),
            "is_diversity_slot": False,
        }
        scored.append(item)

    scored.sort(key=lambda a: a["score"], reverse=True)

    # Enforce 1-in-5 diversity slot reservation for out-of-bubble topics
    out_of_bubble = [a for a in scored if str(a.get("topic", "")).lower() not in dominant_topics]
    in_bubble = [a for a in scored if str(a.get("topic", "")).lower() in dominant_topics]

    if not out_of_bubble or not dominant_topics:
        final_list = scored
    else:
        final_list = []
        used_ids: set[str] = set()
        in_idx = 0
        out_idx = 0
        total_len = len(scored)
        for pos in range(total_len):
            pick = None
            if (pos + 1) % DIVERSITY_SLOT_INTERVAL == 0:
                while out_idx < len(out_of_bubble) and out_of_bubble[out_idx].get("id") in used_ids:
                    out_idx += 1
                if out_idx < len(out_of_bubble):
                    pick = out_of_bubble[out_idx]
                    pick["score_components"]["is_diversity_slot"] = True
                    out_idx += 1
            if pick is None:
                # Pick highest remaining overall article
                for cand in scored:
                    if cand.get("id") not in used_ids:
                        pick = cand
                        break
            if pick is not None:
                used_ids.add(str(pick.get("id", pos)))
                final_list.append(pick)
        _ = in_idx

    user_ctx = {
        "boosted_topics": list(boosted_topics),
        "preferred_sources": list(preferred_sources),
    }
    for art in final_list:
        art["why_reason"] = make_reason(user_ctx, art)

    return final_list


def bubble_score(recent_topics: list[str], total_topics: int = len(TOPICS)) -> dict:
    """
    Compute normalized Shannon entropy of topics in the user's last 30 reads.
    Returns 0.0 (narrow single-topic bubble) to 1.0 (broad across all topics).
    """
    window = [t.lower() for t in recent_topics[-30:] if t]
    if not window or total_topics <= 1:
        return {"score": 0.0, "label": "Narrow", "topic_counts": {}, "sample_size": 0}

    counts = Counter(window)
    total = len(window)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)

    max_entropy = math.log(total_topics)
    normalized = max(0.0, min(1.0, entropy / max_entropy))
    score = round(normalized, 3)

    if score < 0.40:
        label = "Narrow"
    elif score <= 0.72:
        label = "Balanced"
    else:
        label = "Broad"

    return {
        "score": score,
        "label": label,
        "topic_counts": dict(counts),
        "sample_size": total,
    }
