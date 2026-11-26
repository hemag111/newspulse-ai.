"""48-hour sliding-window agglomerative story clustering."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from .config import CLUSTER_SIMILARITY_THRESHOLD, CLUSTER_WINDOW_HOURS
from .embeddings import cosine_similarity, embed


def _parse_dt(raw: str | None) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    try:
        normalized = raw.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def cluster(
    articles: list[dict],
    similarity_threshold: float = CLUSTER_SIMILARITY_THRESHOLD,
    window_hours: int = CLUSTER_WINDOW_HOURS,
) -> list[str]:
    """
    Group articles covering the same story within a 48-hour window using cosine similarity.
    Returns a list of cluster_id strings aligned 1-to-1 with the input `articles` list.
    """
    n = len(articles)
    if n == 0:
        return []
    if n == 1:
        art_id = str(articles[0].get("id", "0"))
        digest = hashlib.sha1(art_id.encode("utf-8")).hexdigest()[:8]
        return [f"clu_{digest}"]

    # Reuse stored embeddings when valid, otherwise batch-embed title + body
    vectors: list[list[float]] = []
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for idx, art in enumerate(articles):
        existing = art.get("embedding")
        if isinstance(existing, list) and len(existing) > 0:
            vectors.append([float(x) for x in existing])
        else:
            vectors.append([])
            missing_indices.append(idx)
            title = art.get("title", "")
            body = art.get("body") or art.get("summary_long") or art.get("summary_short") or ""
            missing_texts.append(f"{title}. {body}".strip())

    if missing_texts:
        computed = embed(missing_texts)
        for idx, vec in zip(missing_indices, computed):
            vectors[idx] = vec

    timestamps = [_parse_dt(art.get("published_at")) for art in articles]
    max_seconds = window_hours * 3600

    # Try sklearn AgglomerativeClustering on a window-masked distance matrix
    labels: list[int] | None = None
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering

        dist_matrix = np.ones((n, n), dtype=float)
        np.fill_diagonal(dist_matrix, 0.0)
        for i in range(n):
            for j in range(i + 1, n):
                dt_diff = abs((timestamps[i] - timestamps[j]).total_seconds())
                if dt_diff <= max_seconds:
                    sim = cosine_similarity(vectors[i], vectors[j])
                    dist = max(0.0, 1.0 - sim)
                else:
                    dist = 1.0
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        model = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="average",
            distance_threshold=1.0 - similarity_threshold,
        )
        labels = [int(lbl) for lbl in model.fit_predict(dist_matrix)]
    except Exception:  # noqa: BLE001
        labels = None

    # Pure-Python average-linkage agglomerative fallback if sklearn is not installed
    if labels is None:
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            sim_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                if abs((timestamps[i] - timestamps[j]).total_seconds()) <= max_seconds:
                    s = cosine_similarity(vectors[i], vectors[j])
                else:
                    s = 0.0
                sim_matrix[i][j] = s
                sim_matrix[j][i] = s

        active_clusters: list[list[int]] = [[i] for i in range(n)]
        while len(active_clusters) > 1:
            best_sim = -1.0
            best_pair = (-1, -1)
            m = len(active_clusters)
            for a in range(m):
                ca = active_clusters[a]
                for b in range(a + 1, m):
                    cb = active_clusters[b]
                    # Respect 48h window across all cluster members
                    valid_window = all(
                        abs((timestamps[i] - timestamps[j]).total_seconds()) <= max_seconds
                        for i in ca
                        for j in cb
                    )
                    if not valid_window:
                        continue
                    avg_sim = sum(sim_matrix[i][j] for i in ca for j in cb) / (len(ca) * len(cb))
                    if avg_sim > best_sim:
                        best_sim = avg_sim
                        best_pair = (a, b)
            if best_sim < similarity_threshold or best_pair[0] == -1:
                break
            a_idx, b_idx = best_pair
            merged = active_clusters[a_idx] + active_clusters[b_idx]
            active_clusters[a_idx] = merged
            active_clusters.pop(b_idx)

        labels = [0] * n
        for cid, members in enumerate(active_clusters):
            for idx in members:
                labels[idx] = cid

    # Build deterministic human-friendly cluster IDs keyed by earliest/smallest article id in group
    groups: dict[int, list[int]] = {}
    for idx, lbl in enumerate(labels):
        groups.setdefault(lbl, []).append(idx)

    cluster_ids = [""] * n
    for member_indices in groups.values():
        member_indices_sorted = sorted(member_indices, key=lambda i: (timestamps[i], str(articles[i].get("id", i))))
        anchor = str(articles[member_indices_sorted[0]].get("id", member_indices_sorted[0]))
        digest = hashlib.sha1(anchor.encode("utf-8")).hexdigest()[:8]
        cid = f"clu_{digest}"
        for idx in member_indices:
            cluster_ids[idx] = cid

    return cluster_ids
