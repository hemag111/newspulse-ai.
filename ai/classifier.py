"""Zero-shot topic classification using embedding similarity over label descriptions."""

from __future__ import annotations

import re

from .config import TOPICS, TOPIC_DESCRIPTIONS
from .embeddings import cosine_similarity, embed

_TOPIC_VECTOR_CACHE: dict[str, dict[str, list[float]]] = {}

# Bilingual keyword anchors to strengthen zero-shot classification on short Kannada/Hindi headlines
_KEYWORD_ANCHORS: dict[str, tuple[str, ...]] = {
    "technology": ("ai", "software", "chip", "cloud", "startup", "tech", "cyber", "app", "linux", "webgpu", "quantum", "rust", "risc-v", "ತಂತ್ರಜ್ಞಾನ", "एआई", "तकनीक"),
    "sports": ("cricket", "match", "ipl", "goal", "tournament", "olympic", "wicket", "football", "badminton", "chess", "marathon", "swimmers", "javelin", "archery", "offside", "ಕ್ರೀಡೆ", "ಕ್ರಿಕೆಟ್", "क्रिकेट", "खेल"),
    "business": ("market", "rbi", "sensex", "nifty", "gdp", "revenue", "funding", "bank", "pmi", "export", "logistics", "leasing", "battery", "cell", "coffee", "ವ್ಯಾಪಾರ", "ಆರ್ಥಿಕ", "बाजार", "अर्थव्यवस्था"),
    "science": ("isro", "nasa", "space", "orbit", "quantum", "climate", "telescope", "solar", "enzyme", "ocean", "glacier", "ವಿಜ್ಞಾನ", "ಬಾಹ್ಯಾಕಾಶ", "अंतरिक्ष", "विज्ञान"),
    "india": ("bengaluru", "delhi", "mumbai", "parliament", "railway", "metro", "karnataka", "chennai", "kochi", "pune", "gujarat", "ಭಾರತ", "ಬೆಂಗಳೂರು", "भारत", "दिल्ली"),
    "world": ("un", "summit", "treaty", "europe", "diplomatic", "foreign", "global", "maritime", "nordic", "pacific", "african", "ವಿಶ್ವ", "ಅಂತಾರಾಷ್ಟ್ರೀಯ", "विश्व", "अंतरराष्ट्रीय"),
    "entertainment": ("film", "cinema", "movie", "music", "album", "actor", "festival", "theater", "jazz", "audiobook", "ಸಿನಿಮಾ", "मनोरंजन", "फिल्म"),
    "health": ("health", "hospital", "vaccine", "clinic", "medical", "nutrition", "disease", "cardiology", "sleep", "neonatal", "ಆರೋಗ್ಯ", "स्वास्थ्य", "अस्पताल"),
    "education": ("university", "school", "exam", "student", "scholarship", "nep", "curriculum", "polytechnics", "textbook", "teacher", "ಶಿಕ್ಷಣ", "ವಿಶ್ವವಿದ್ಯಾಲಯ", "शिक्षा", "परीक्षा"),
}


def _get_topic_vectors(language: str) -> dict[str, list[float]]:
    space = "multi" if language in {"kn", "hi", "multilingual"} else "en"
    if space in _TOPIC_VECTOR_CACHE:
        return _TOPIC_VECTOR_CACHE[space]
    descriptions = [TOPIC_DESCRIPTIONS[t] for t in TOPICS]
    vectors = embed(descriptions, language="kn" if space == "multi" else "en")
    mapping = {topic: vec for topic, vec in zip(TOPICS, vectors)}
    _TOPIC_VECTOR_CACHE[space] = mapping
    return mapping


def classify_with_scores(text: str, language: str = "en") -> tuple[str, dict[str, float]]:
    """Return (best_topic, per_topic_scores) for transparency and evaluation."""
    if not text or not text.strip():
        return "india", {t: 0.0 for t in TOPICS}

    topic_vecs = _get_topic_vectors(language)
    text_vec = embed([text], language=language)[0]

    lowered = text.lower()
    tokens = set(re.findall(r"[\w\u0900-\u097f\u0c80-\u0cff]+", lowered))

    scores: dict[str, float] = {}
    for topic in TOPICS:
        sim = cosine_similarity(text_vec, topic_vecs[topic])
        anchors = _KEYWORD_ANCHORS.get(topic, ())
        anchor_hits = sum(1 for kw in anchors if kw in tokens or (len(kw) > 3 and kw in lowered))
        scores[topic] = round(sim + min(0.18, anchor_hits * 0.06), 4)

    best_topic = max(TOPICS, key=lambda t: scores[t])
    return best_topic, scores


def classify(text: str, language: str = "en") -> str:
    """Zero-shot classify text into one of the 9 canonical topics."""
    best_topic, _ = classify_with_scores(text, language=language)
    return best_topic
