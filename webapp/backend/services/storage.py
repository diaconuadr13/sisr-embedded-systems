from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from backend.config import DB_PATH, SCHEMA_SQL_PATH, WEBAPP_DATA_DIR

_init_lock = threading.Lock()
_initialized = False


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        WEBAPP_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            conn.executescript(SCHEMA_SQL_PATH.read_text(encoding="utf-8"))
        _initialized = True


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    _ensure_initialized()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---- Notes ----

def get_note(run_id: str) -> str | None:
    with connection() as conn:
        row = conn.execute("SELECT body FROM notes WHERE run_id = ?", (run_id,)).fetchone()
    return row["body"] if row else None


def upsert_note(run_id: str, body: str) -> None:
    body = body.strip()
    with connection() as conn:
        if not body:
            conn.execute("DELETE FROM notes WHERE run_id = ?", (run_id,))
            return
        conn.execute(
            "INSERT INTO notes(run_id, body, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id) DO UPDATE SET body=excluded.body, updated_at=excluded.updated_at",
            (run_id, body, _iso_now()),
        )


# ---- Tags ----

def get_tags(run_id: str) -> list[str]:
    with connection() as conn:
        rows = conn.execute("SELECT tag FROM tags WHERE run_id = ? ORDER BY tag", (run_id,)).fetchall()
    return [r["tag"] for r in rows]


def get_tags_bulk(run_ids: list[str]) -> dict[str, list[str]]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" * len(run_ids))
    with connection() as conn:
        rows = conn.execute(
            f"SELECT run_id, tag FROM tags WHERE run_id IN ({placeholders}) ORDER BY run_id, tag",
            run_ids,
        ).fetchall()
    result: dict[str, list[str]] = {rid: [] for rid in run_ids}
    for row in rows:
        result.setdefault(row["run_id"], []).append(row["tag"])
    return result


def add_tag(run_id: str, tag: str) -> None:
    tag = tag.strip().lower()
    if not tag:
        return
    with connection() as conn:
        conn.execute("INSERT OR IGNORE INTO tags(run_id, tag) VALUES (?, ?)", (run_id, tag))


def remove_tag(run_id: str, tag: str) -> None:
    with connection() as conn:
        conn.execute("DELETE FROM tags WHERE run_id = ? AND tag = ?", (run_id, tag.lower()))


def all_tags() -> list[str]:
    with connection() as conn:
        rows = conn.execute("SELECT DISTINCT tag FROM tags ORDER BY tag").fetchall()
    return [r["tag"] for r in rows]
