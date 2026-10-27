"""Replies cached on disk: a cachetools LRU holds the hot set, SQLite keeps every entry across runs."""

import hashlib
import json
import sqlite3
from pathlib import Path

import cachetools


class DiskCache(cachetools.LRUCache):
    def __init__(self, path: Path, maxsize: int = 4096):
        super().__init__(maxsize)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("create table if not exists replies (key text primary key, value text)")

    def __missing__(self, key: str):
        row = self.db.execute("select value from replies where key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(key)
        value = json.loads(row[0])
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key: str, value) -> None:
        super().__setitem__(key, value)
        self.db.execute("insert or replace into replies values (?, ?)", (key, json.dumps(value, ensure_ascii=False)))

    def __delitem__(self, key: str) -> None:
        """Eviction from the hot set only (cachetools evicts through here); the row stays on disk."""
        super().__delitem__(key)

    def discard(self, key: str) -> None:
        """Forget a reply everywhere."""
        self.db.execute("delete from replies where key = ?", (key,))
        self.pop(key, None)


def key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
