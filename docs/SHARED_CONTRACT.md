# Shared Contract — Personalized AI News Aggregator

This document locks the data shapes, REST endpoints, database tables, and Python AI package signatures shared across Member 1 (UX/Testing), Member 2 (Frontend), Member 3 (Backend), and Member 4 (AI Engine).

## 1. Article Object (JSON)

```json
{
  "id": "art_001",
  "title": "ISRO Completes Cryogenic Stage Hot Test for Next-Gen Launch Vehicle",
  "url": "https://example.com/isro-cryogenic-test",
  "source": "The Hindu",
  "published_at": "2026-11-12T08:30:00Z",
  "topic": "science",
  "language": "en",
  "summary_short": "- ISRO completed a 200-second hot test of its CE-20 cryogenic engine.\n- The test validated upgraded thrust controls at Mahendragiri.\n- The stage will power upcoming heavy-lift commercial missions.",
  "summary_long": "The Indian Space Research Organisation successfully conducted a 200-second flight-acceptance hot test...",
  "cluster_id": "clu_isro_01",
  "clickbait_score": 0.08,
  "reading_time_min": 3,
  "why_reason": "Matches your interest in Science and is from The Hindu, a source you read often."
}
```

## 2. REST API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /auth/signup`, `POST /auth/login` | Account creation and authentication |
| `GET /topics` | Canonical topic list |
| `GET/POST /users/{id}/preferences` | Topics, languages, sources, daily limit, muted topics |
| `POST /preferences/parse` | Natural-language preferences -> `{boost, reduce, mute}` |
| `GET /feed?user_id&topic&source&language&max_minutes&date&limit` | Personalized, diversity-aware ranked feed |
| `GET /story/{cluster_id}` | Merged multi-source story view |
| `POST /events` | `{user_id, article_id, type: click\|like\|skip\|save}` |
| `GET /digest?user_id&minutes=5` | Time-budget reading digest |
| `GET/POST /tracks`, `GET /tracks/{id}/timeline` | Tracked topic timelines & "what changed" |
| `POST /ask` | `{cluster_id, question}` retrieval-grounded Q&A |
| `GET /stats/{user_id}` | Reading stats and filter-bubble score |

## 3. Database Tables

`users`, `preferences`, `articles`, `clusters`, `events`, `tracks`, `saved`

## 4. AI Package Function Signatures (Member 4)

```python
summarize(text: str) -> dict                 # {"short": str, "long": str}
classify(text: str) -> str                   # topic label from TOPICS
embed(texts: list[str]) -> list[list[float]] # 384-d L2-normalized vectors
cluster(articles: list[dict]) -> list[str]   # aligned list of cluster_ids
clickbait_score(title: str, body: str) -> float  # 0.0 to 1.0 (flag > 0.6)
rank(user_vector: list[float], articles: list[dict], prefs: dict) -> list[dict]
make_reason(user: dict, article: dict) -> str
parse_prefs(text: str) -> dict               # {"boost": [], "reduce": [], "mute": []}
ask(question: str, articles: list[dict]) -> str
track_timeline(articles: list[dict]) -> list[dict]
```

Additional helper exports for Member 3:
```python
build_user_vector(preferred_topics: list[str], language: str = "en") -> list[float]
update_user_vector(user_vector: list[float], article_vector: list[float], event_type: str) -> list[float]
bubble_score(recent_topics: list[str]) -> dict
merge_cluster_summaries(articles: list[dict]) -> dict
summarize_what_changed(articles: list[dict], last_seen: str | None) -> str
```
