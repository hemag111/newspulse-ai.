# NewsPulse AI Engine (`ai/` — Member 4)

Self-contained Python AI package implementing the agreed Step 0 contract for the personalized news aggregator.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r ai/requirements.txt
cp ai/.env.example .env
# Add your Gemini API key to .env (GEMINI_API_KEY=...)
```

## Run Tests & Evaluation

```bash
python -m unittest discover -s ai/tests -v
python -m ai.evaluate
```

## Integration Guide for Member 3 (FastAPI Backend)

Import all contract functions directly from `ai`:

```python
from ai import (
    TOPICS,
    embed,
    classify,
    summarize,
    cluster,
    clickbait_score,
    build_user_vector,
    update_user_vector,
    rank,
    make_reason,
    parse_prefs,
    ask,
    track_timeline,
    summarize_what_changed,
    bubble_score,
)
```

### 1. During RSS Ingestion (`ingest.py`)

```python
topic = classify(f"{title}. {body}", language=lang)
summaries = summarize(body, article_id=article_id)   # {"short": "...", "long": "..."}
vector = embed([f"{title}. {body}"], language=lang)[0]
cb_score = clickbait_score(title, body)               # float 0..1 (flag if > 0.6)

# Periodically cluster recent articles within 48h window:
cluster_ids = cluster(recent_articles)
```

### 2. Cold-Start Onboarding & Events (`POST /users/{id}/preferences`, `POST /events`)

```python
# On onboarding topic selection:
user_vec = build_user_vector(selected_topics, language=preferred_language)

# On user interaction event (click / like / save / skip):
user_vec = update_user_vector(user_vec, article_vector, event_type)
```

### 3. Personalized Feed (`GET /feed`)

```python
prefs = {
    "topics": user.preferred_topics,
    "muted_topics": user.muted_topics,
    "sources": user.preferred_sources,
    "boost": user.boosted_topics,
    "reduce": user.reduced_topics,
    "recent_topics": last_30_read_topics,
}
ranked_articles = rank(user_vec, candidate_articles, prefs)
# Each returned article contains `score`, `why_reason`, and `score_components`
```

### 4. Natural-Language Preferences (`POST /preferences/parse`)

```python
delta = parse_prefs("less politics, more space and startups")
# -> {"boost": ["science", "technology"], "reduce": ["india"], "mute": []}
```

### 5. Story Q&A, Topic Tracker & Bubble Stats

```python
answer = ask(question, cluster_articles)
timeline = track_timeline(matching_articles, last_seen=track.last_seen)
what_changed = summarize_what_changed(matching_articles, last_seen=track.last_seen)
bubble = bubble_score(last_30_read_topics)  # {"score": 0..1, "label": "Narrow|Balanced|Broad"}
```

## Module Layout

| File | Responsibility |
|---|---|
| `ai/config.py` | Topics, label descriptions, model IDs, ranking weights, thresholds |
| `ai/cache.py` | Thread-safe SQLite cache (`ai/.cache/ai_cache.sqlite`) |
| `ai/llm.py` | `google-genai` wrapper with exponential backoff & JSON validation |
| `ai/embeddings.py` | `all-MiniLM-L6-v2` (EN) & `paraphrase-multilingual-MiniLM-L12-v2` (KN/HI) |
| `ai/classifier.py` | Zero-shot topic classification via cosine similarity |
| `ai/summarizer.py` | Grounded 3-bullet + ~100-word summarizer & cluster merger |
| `ai/clustering.py` | 48-hour sliding-window agglomerative clustering |
| `ai/clickbait.py` | 3-signal clickbait detector (heuristics + divergence + LLM) |
| `ai/recommender.py` | Ranking, 1-in-5 diversity slots, user vector updates, reasons, Shannon entropy |
| `ai/assistant.py` | `parse_prefs`, retrieval-grounded `ask`, `track_timeline` |
| `ai/data/sample.json` | 200 multi-source, multilingual (EN/KN/HI) sample news articles |
| `ai/evaluate.py` | Benchmarks for classification, clickbait, clustering, and feed shift |
| `ai/tests/test_ai.py` | Unit tests covering all Step 0 contract signatures |
