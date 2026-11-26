"""
Member 4 Evaluation Suite
=========================
Computes quantitative evaluation metrics required for Member 1's report:
1. Zero-shot topic classification accuracy across 50 labeled articles.
2. Clickbait detection precision, recall, F1, and accuracy on 30 labeled examples.
3. Story clustering purity / pairwise F1 across 20 ground-truth multi-source clusters.
4. Personalized recommendation feed-shift simulation for a Tech-only user + bubble score.

Run:
    python3 -m ai.evaluate
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .classifier import classify
from .clickbait import clickbait_score
from .clustering import cluster
from .config import CLICKBAIT_FLAG_THRESHOLD, DATA_DIR
from .embeddings import embed
from .recommender import bubble_score, build_user_vector, rank, update_user_vector


def evaluate_classification() -> dict:
    items = json.loads((DATA_DIR / "eval_classification_50.json").read_text(encoding="utf-8"))
    correct = 0
    per_topic: dict[str, dict[str, int]] = {}
    for row in items:
        pred = classify(row["text"], language=row.get("language", "en"))
        true_lbl = row["true_topic"]
        bucket = per_topic.setdefault(true_lbl, {"total": 0, "correct": 0})
        bucket["total"] += 1
        if pred == true_lbl:
            correct += 1
            bucket["correct"] += 1
    accuracy = round(correct / max(1, len(items)), 4)
    return {
        "total": len(items),
        "correct": correct,
        "accuracy": accuracy,
        "per_topic": per_topic,
    }


def evaluate_clickbait() -> dict:
    items = json.loads((DATA_DIR / "eval_clickbait_30.json").read_text(encoding="utf-8"))
    tp = fp = tn = fn = 0
    for row in items:
        score = clickbait_score(row["title"], row["body"], use_llm=False)
        pred = score > CLICKBAIT_FLAG_THRESHOLD
        actual = bool(row["is_clickbait"])
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)
    accuracy = (tp + tn) / max(1, len(items))
    return {
        "total": len(items),
        "threshold": CLICKBAIT_FLAG_THRESHOLD,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def evaluate_clustering() -> dict:
    articles = json.loads((DATA_DIR / "sample.json").read_text(encoding="utf-8"))
    # Evaluate on the first 20 multi-source clusters
    multi = [a for a in articles if not str(a["cluster_id"]).startswith("clu_single_")]
    pred_clusters = cluster(multi, similarity_threshold=0.72, window_hours=48)

    # Pairwise precision, recall, F1 + cluster purity
    n = len(multi)
    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_true = multi[i]["cluster_id"] == multi[j]["cluster_id"]
            same_pred = pred_clusters[i] == pred_clusters[j]
            if same_true and same_pred:
                tp += 1
            elif not same_true and same_pred:
                fp += 1
            elif same_true and not same_pred:
                fn += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-9, precision + recall)

    pred_groups: dict[str, list[str]] = {}
    for art, pc in zip(multi, pred_clusters):
        pred_groups.setdefault(pc, []).append(art["cluster_id"])
    pure_count = sum(Counter(g).most_common(1)[0][1] for g in pred_groups.values())
    purity = pure_count / max(1, n)

    return {
        "evaluated_articles": n,
        "ground_truth_clusters": len({a["cluster_id"] for a in multi}),
        "predicted_clusters": len(pred_groups),
        "purity": round(purity, 4),
        "pairwise_precision": round(precision, 4),
        "pairwise_recall": round(recall, 4),
        "pairwise_f1": round(f1, 4),
    }


def evaluate_recommendation_shift() -> dict:
    articles = json.loads((DATA_DIR / "sample.json").read_text(encoding="utf-8"))
    ref_now = datetime(2026, 11, 12, 18, 0, 0, tzinfo=timezone.utc)

    # Pre-embed sample pool once for fast simulation
    texts = [f"{a['title']}. {a['summary_short']}" for a in articles]
    vecs = embed(texts)
    for a, v in zip(articles, vecs):
        a["embedding"] = v

    # Cold start: user picking sports, health, entertainment (non-tech baseline)
    user_vec = build_user_vector(["sports", "health", "entertainment"])
    prefs_before = {
        "topics": ["sports", "health", "entertainment"],
        "sources": ["BBC"],
        "recent_topics": ["sports", "health", "entertainment"],
    }
    before_feed = rank(user_vec, articles, prefs_before, now=ref_now)[:10]
    tech_before = sum(1 for a in before_feed if a["topic"] == "technology")

    # Simulate user liking 6 technology articles
    tech_articles = [a for a in articles if a["topic"] == "technology"][:6]
    recent = ["sports", "health", "entertainment"]
    for ta in tech_articles:
        user_vec = update_user_vector(user_vec, ta["embedding"], "like")
        recent.append("technology")

    prefs_after = {
        "topics": ["sports", "health", "entertainment"],
        "sources": ["BBC"],
        "recent_topics": recent,
    }
    after_feed = rank(user_vec, articles, prefs_after, now=ref_now)[:10]
    tech_after = sum(1 for a in after_feed if a["topic"] == "technology")

    bubble_narrow = bubble_score(["technology"] * 10)
    bubble_balanced = bubble_score(recent)

    return {
        "top_10_tech_before_likes": tech_before,
        "top_10_tech_after_5_tech_likes": tech_after,
        "feed_shift_delta": tech_after - tech_before,
        "bubble_score_tech_only": bubble_narrow,
        "bubble_score_after_mixed": bubble_balanced,
    }


def run_all() -> dict:
    results = {
        "classification_50": evaluate_classification(),
        "clickbait_30": evaluate_clickbait(),
        "clustering_20": evaluate_clustering(),
        "recommendation_shift": evaluate_recommendation_shift(),
    }
    out_path = DATA_DIR.parent / "evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    summary = run_all()
    print(json.dumps(summary, indent=2))
