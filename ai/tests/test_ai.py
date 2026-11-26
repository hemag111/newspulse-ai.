"""Unit tests verifying Member 4's AI Engine contract signatures and behavior."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from ai import (
    TOPICS,
    ask,
    bubble_score,
    build_user_vector,
    classify,
    clickbait_score,
    cluster,
    embed,
    make_reason,
    parse_prefs,
    rank,
    summarize,
    summarize_what_changed,
    track_timeline,
    update_user_vector,
)
from ai.config import DATA_DIR


class Member4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample_articles = json.loads((DATA_DIR / "sample.json").read_text(encoding="utf-8"))

    def test_sample_dataset_has_200_articles_and_contract_fields(self) -> None:
        self.assertEqual(len(self.sample_articles), 200)
        required = {
            "id",
            "title",
            "url",
            "source",
            "published_at",
            "topic",
            "language",
            "summary_short",
            "summary_long",
            "cluster_id",
            "clickbait_score",
            "reading_time_min",
            "why_reason",
        }
        for art in self.sample_articles[:10]:
            self.assertTrue(required.issubset(art.keys()))

    def test_embed_returns_normalized_384d_vectors(self) -> None:
        vecs = embed(["ISRO cryogenic rocket engine test", "ಬೆಂಗಳೂರು ಮೆಟ್ರೋ ಹಳದಿ ಮಾರ್ಗ"], language="kn")
        self.assertEqual(len(vecs), 2)
        self.assertEqual(len(vecs[0]), 384)
        norm_sq = sum(x * x for x in vecs[0])
        self.assertAlmostEqual(norm_sq, 1.0, places=3)

    def test_classify_returns_canonical_topic(self) -> None:
        topic = classify("ISRO launched a cryogenic space propulsion rocket into orbit")
        self.assertIn(topic, TOPICS)
        self.assertEqual(topic, "science")

    def test_summarize_contract_shape(self) -> None:
        out = summarize(
            "ISRO completed a 200-second hot test of its CE-20 cryogenic engine at Mahendragiri. "
            "Engineers verified chamber stability. The stage will support heavy-lift launches."
        )
        self.assertIn("short", out)
        self.assertIn("long", out)
        self.assertTrue(len(out["short"]) > 10)
        self.assertTrue(len(out["long"]) > 10)

    def test_clickbait_score_separates_bait_and_factual(self) -> None:
        bait = clickbait_score(
            "YOU WON'T BELIEVE What This Secret Setting Does!!",
            "Routine software settings adjust background sync every eight hours.",
        )
        factual = clickbait_score(
            "ISRO Completes 200-Second Cryogenic Engine Hot Test at Mahendragiri",
            "ISRO completed a 200-second hot test of the CE-20 cryogenic engine at Mahendragiri propulsion complex.",
        )
        self.assertGreater(bait, 0.6)
        self.assertLess(factual, 0.4)

    def test_cluster_respects_48h_window(self) -> None:
        a1 = {
            "id": "t1",
            "title": "ISRO Completes Cryogenic Engine Hot Test at Mahendragiri",
            "body": "ISRO completed a 200-second test of the CE-20 engine.",
            "published_at": "2026-11-12T10:00:00Z",
        }
        a2 = {
            "id": "t2",
            "title": "ISRO Completes Cryogenic Engine Hot Test at Mahendragiri",
            "body": "ISRO completed a 200-second test of the CE-20 engine.",
            "published_at": "2026-11-12T14:00:00Z",
        }
        a3_old = {
            "id": "t3",
            "title": "ISRO Completes Cryogenic Engine Hot Test at Mahendragiri",
            "body": "ISRO completed a 200-second test of the CE-20 engine.",
            "published_at": "2026-11-05T10:00:00Z",  # > 48h older
        }
        cids = cluster([a1, a2, a3_old], similarity_threshold=0.75, window_hours=48)
        self.assertEqual(len(cids), 3)
        self.assertEqual(cids[0], cids[1])
        self.assertNotEqual(cids[0], cids[2])

    def test_rank_mutes_topics_and_reserves_diversity_slots(self) -> None:
        subset = self.sample_articles[:25]
        uvec = build_user_vector(["technology", "science"])
        prefs = {
            "topics": ["technology", "science"],
            "muted_topics": ["entertainment"],
            "sources": ["The Hindu"],
            "recent_topics": ["technology", "technology", "science"],
        }
        ranked = rank(uvec, subset, prefs, now=datetime(2026, 11, 12, 18, 0, 0, tzinfo=timezone.utc))
        self.assertTrue(all(a["topic"] != "entertainment" for a in ranked))
        self.assertTrue(all("why_reason" in a and len(a["why_reason"]) > 5 for a in ranked))
        if len(ranked) >= 5:
            self.assertTrue(ranked[4]["score_components"]["is_diversity_slot"])

    def test_update_user_vector_moves_toward_liked_article(self) -> None:
        u = build_user_vector(["sports"])
        tech_vec = embed(["Quantum computing and RISC-V AI chip architecture"])[0]
        u_after = update_user_vector(u, tech_vec, "like")
        self.assertNotEqual(u, u_after)
        self.assertAlmostEqual(sum(x * x for x in u_after), 1.0, places=3)

    def test_bubble_score_entropy_scale(self) -> None:
        narrow = bubble_score(["technology"] * 12)
        broad = bubble_score(["technology", "science", "sports", "business", "india", "world", "health", "education", "entertainment"])
        self.assertLess(narrow["score"], 0.1)
        self.assertEqual(narrow["label"], "Narrow")
        self.assertGreater(broad["score"], 0.9)
        self.assertEqual(broad["label"], "Broad")

    def test_parse_prefs_valid_structure(self) -> None:
        parsed = parse_prefs("less politics, more space and startups, mute entertainment")
        self.assertIn("science", parsed["boost"])
        self.assertIn("technology", parsed["boost"])
        self.assertIn("india", parsed["reduce"])
        self.assertIn("entertainment", parsed["mute"])

    def test_ask_and_track_timeline(self) -> None:
        cluster_arts = [a for a in self.sample_articles if a["cluster_id"] == "clu_isro_ce20"]
        ans = ask("How many seconds was the CE-20 cryogenic test?", cluster_arts)
        self.assertIn("200", ans)
        missing = ask("What was the launch price of the lunar hotel?", cluster_arts)
        self.assertEqual(missing, "Not in these articles.")

        tl = track_timeline(cluster_arts, last_seen="2026-11-11T00:00:00Z")
        self.assertIsInstance(tl, list)
        self.assertGreaterEqual(len(tl), 1)
        changed = summarize_what_changed(cluster_arts, last_seen="2026-11-11T00:00:00Z")
        self.assertTrue(len(changed) > 5)

    def test_make_reason_honest_template(self) -> None:
        reason = make_reason(
            {"boosted_topics": ["science"], "preferred_sources": ["The Hindu"]},
            {"topic": "science", "source": "The Hindu", "score_components": {"similarity": 0.9, "recency": 0.8, "source": 1.0, "is_diversity_slot": False}},
        )
        self.assertIn("Science", reason)


if __name__ == "__main__":
    unittest.main()
