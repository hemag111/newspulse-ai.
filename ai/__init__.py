"""
Member 4 AI Engine Package
==========================
Exports all contract functions required by Member 3's FastAPI backend:
  summarize, classify, embed, cluster, clickbait_score,
  rank, make_reason, parse_prefs, ask, track_timeline
Plus helper utilities:
  build_user_vector, update_user_vector, bubble_score,
  merge_cluster_summaries, analyze_clickbait, summarize_what_changed
"""

from .assistant import ask, parse_prefs, summarize_what_changed, track_timeline
from .classifier import classify, classify_with_scores
from .clickbait import analyze_clickbait, clickbait_score
from .clustering import cluster
from .config import TOPICS, TOPIC_DESCRIPTIONS
from .embeddings import cosine_similarity, embed
from .recommender import bubble_score, build_user_vector, make_reason, rank, update_user_vector
from .summarizer import merge_cluster_summaries, summarize

__all__ = [
    "TOPICS",
    "TOPIC_DESCRIPTIONS",
    "ask",
    "analyze_clickbait",
    "bubble_score",
    "build_user_vector",
    "classify",
    "classify_with_scores",
    "clickbait_score",
    "cluster",
    "cosine_similarity",
    "embed",
    "make_reason",
    "merge_cluster_summaries",
    "parse_prefs",
    "rank",
    "summarize",
    "summarize_what_changed",
    "track_timeline",
    "update_user_vector",
]
