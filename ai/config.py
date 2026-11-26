"""Central configuration constants for the Member 4 AI Engine."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
CACHE_DIR = Path(os.getenv("AI_CACHE_DIR", str(PACKAGE_DIR / ".cache")))

# Canonical topic list agreed in Step 0 shared contract
TOPICS: list[str] = [
    "technology",
    "sports",
    "business",
    "science",
    "india",
    "world",
    "entertainment",
    "health",
    "education",
]

# Rich label descriptions used for zero-shot embedding classification
TOPIC_DESCRIPTIONS: dict[str, str] = {
    "technology": (
        "Technology, artificial intelligence, software engineering, startups, "
        "smartphones, cybersecurity, cloud computing, semiconductors, and apps."
    ),
    "sports": (
        "Sports, cricket, football, tennis, Olympics, tournaments, match scores, "
        "athletes, leagues, and championships."
    ),
    "business": (
        "Business, economy, stock markets, RBI, inflation, startups funding, "
        "corporate earnings, banking, and trade."
    ),
    "science": (
        "Science, space exploration, ISRO, NASA, astronomy, physics, climate "
        "research, biology, and scientific discoveries."
    ),
    "india": (
        "India national news, Bengaluru, Delhi, Mumbai, state policies, railways, "
        "public infrastructure, elections, and domestic affairs."
    ),
    "world": (
        "World news, international diplomacy, geopolitics, United Nations, "
        "global summits, foreign affairs, and international conflicts."
    ),
    "entertainment": (
        "Entertainment, cinema, Bollywood, Sandalwood, music, streaming series, "
        "film festivals, books, and cultural arts."
    ),
    "health": (
        "Health, medicine, hospitals, public health, nutrition, mental wellness, "
        "vaccines, clinical trials, and fitness."
    ),
    "education": (
        "Education, universities, schools, exams, NEP policy, scholarships, "
        "student research, online courses, and edtech."
    ),
}

# Embedding models (Member 4 spec)
ENGLISH_EMBED_MODEL = "all-MiniLM-L6-v2"
MULTILINGUAL_EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Gemini model for text tasks (free tier supported)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Clustering parameters
CLUSTER_SIMILARITY_THRESHOLD = 0.75
CLUSTER_WINDOW_HOURS = 48

# Clickbait flagging threshold
CLICKBAIT_FLAG_THRESHOLD = 0.60

# Recommendation ranking weights
RANK_WEIGHTS = {
    "similarity": 0.50,
    "recency": 0.20,
    "source": 0.15,
    "diversity": 0.15,
}
DIVERSITY_SLOT_INTERVAL = 5  # Reserve 1 in 5 slots for out-of-bubble topics

# User vector update coefficients
EVENT_STEP_WEIGHTS = {
    "click": (0.90, 0.10),
    "like": (0.75, 0.25),
    "save": (0.70, 0.30),
    "skip": (1.00, -0.12),
}
