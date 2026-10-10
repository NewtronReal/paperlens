"""Tiny thread-safe SQLite key/value cache for API responses."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT, t REAL)")
            self._conn.commit()

    def get(self, key: str, max_age: float | None = None) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT v, t FROM kv WHERE k = ?", (key,)).fetchone()
        if row is None:
            return None
        if max_age is not None and time.time() - row[1] > max_age:
            return None
        return json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kv (k, v, t) VALUES (?, ?, ?)",
                (key, json.dumps(value), time.time()),
            )
            self._conn.commit()


class NullCache:
    def get(self, key, max_age=None):
        return None

    def set(self, key, value):
        pass
