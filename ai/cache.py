"""Thread-safe SQLite disk cache for LLM responses, summaries, and scores."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from .config import CACHE_DIR


class AICache:
    """Persistent key-value cache backed by SQLite so no LLM call is repeated."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or (CACHE_DIR / "ai_cache.sqlite")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key_hash)
                )
                """
            )
            conn.commit()

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, namespace: str, raw_key: str) -> Any | None:
        key_hash = self._hash_key(raw_key)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM cache_entries WHERE namespace = ? AND key_hash = ?",
                (namespace, key_hash),
            ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, namespace: str, raw_key: str, value: Any) -> None:
        key_hash = self._hash_key(raw_key)
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (namespace, key_hash, value_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key_hash) DO UPDATE SET
                    value_json = excluded.value_json,
                    created_at = excluded.created_at
                """,
                (namespace, key_hash, payload, time.time()),
            )
            conn.commit()

    def clear(self, namespace: str | None = None) -> None:
        with self._lock, self._connect() as conn:
            if namespace:
                conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
            else:
                conn.execute("DELETE FROM cache_entries")
            conn.commit()


_default_cache: AICache | None = None


def get_cache() -> AICache:
    global _default_cache
    if _default_cache is None:
        _default_cache = AICache()
    return _default_cache
