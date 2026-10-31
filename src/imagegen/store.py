"""Task rows in SQLite: the single source of truth for task state, so tasks survive a restart."""

from __future__ import annotations

import os
import sqlite3
import threading

CREATED, IN_PROGRESS, COMPLETED, FAILED = "CREATED", "IN_PROGRESS", "COMPLETED", "FAILED"

# ponytail: new columns go in by hand in TaskDB.__init__; anything bigger needs a real migration (or deleting the db file)
SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL,
    model TEXT NOT NULL,
    seed INTEGER NOT NULL,
    request TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    step INTEGER,
    steps INTEGER,
    error TEXT,
    image BLOB,
    steps_run INTEGER,
    caption TEXT
);
CREATE INDEX IF NOT EXISTS tasks_status_seq ON tasks (status, seq);
CREATE INDEX IF NOT EXISTS tasks_finished_at ON tasks (finished_at);
"""

_VIEW = f"""SELECT id, status, model, seed, created_at, started_at, finished_at, step, steps, error, caption,
    CASE WHEN status = '{CREATED}' THEN
        (SELECT COUNT(*) FROM tasks q WHERE q.status = '{CREATED}' AND q.seq < tasks.seq) END AS position
FROM tasks WHERE id = ?"""


class TaskDB:
    """One connection shared by all threads behind a lock; each statement commits on its own."""

    def __init__(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        if "caption" not in [c["name"] for c in self._conn.execute("PRAGMA table_info(tasks)")]:  # a db from before captions
            self._conn.execute("ALTER TABLE tasks ADD COLUMN caption TEXT")

    def _exec(self, sql: str, args=()) -> list[sqlite3.Row]:
        with self._lock:  # rows fetched under the lock too: the cursor shares the connection
            return self._conn.execute(sql, args).fetchall()

    def insert(self, task_id: str, model: str, seed: int, request: str, now: float) -> None:
        self._exec("INSERT INTO tasks (id, status, model, seed, request, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                   (task_id, CREATED, model, seed, request, now))

    def set_started(self, task_id: str, now: float) -> str | None:
        """Flip a CREATED task to IN_PROGRESS and hand back its request JSON (dropped from the row); else None."""
        with self._lock:
            row = self._conn.execute("SELECT request FROM tasks WHERE id = ? AND status = ?", (task_id, CREATED)).fetchone()
            if row is None:
                return None
            done = self._conn.execute(
                "UPDATE tasks SET status = ?, started_at = ?, request = NULL WHERE id = ? AND status = ?",
                (IN_PROGRESS, now, task_id, CREATED)).rowcount
            return row["request"] if done else None

    # the IN_PROGRESS guards keep a render from a crashed-and-replaced store from overwriting its recovered row
    def set_step(self, task_id: str, step: int, steps: int) -> None:
        self._exec("UPDATE tasks SET step = ?, steps = ? WHERE id = ? AND status = ?", (step, steps, task_id, IN_PROGRESS))

    def set_finished(self, task_id: str, now: float, image: bytes | None, steps_run: int | None, error: str | None,
                     caption: str | None = None) -> None:
        self._exec("UPDATE tasks SET status = ?, finished_at = ?, image = ?, steps_run = ?, error = ?, caption = ?,"
                   " request = NULL WHERE id = ? AND status = ?",
                   (FAILED if error else COMPLETED, now, image, steps_run, error, caption, task_id, IN_PROGRESS))

    def get(self, task_id: str) -> dict | None:
        """The task's fields (no request or image) plus `position` among CREATED tasks."""
        rows = self._exec(_VIEW, (task_id,))
        return dict(rows[0]) if rows else None

    def image(self, task_id: str) -> tuple[bytes, int] | None:
        rows = self._exec("SELECT image, steps_run FROM tasks WHERE id = ? AND status = ?", (task_id, COMPLETED))
        return (rows[0]["image"], rows[0]["steps_run"]) if rows else None

    def counts(self) -> tuple[int, int]:
        """(CREATED tasks, all tasks)."""
        queued, total = self._exec("SELECT COUNT(*) FILTER (WHERE status = ?), COUNT(*) FROM tasks", (CREATED,))[0]
        return queued, total

    def recover(self, now: float) -> list[str]:
        """After a restart: fail the IN_PROGRESS rows; the CREATED ids, in submit order, to queue again."""
        self._exec("UPDATE tasks SET status = ?, finished_at = ?, error = ?, request = NULL WHERE status = ?",
                   (FAILED, now, "server restarted during render", IN_PROGRESS))
        return [r["id"] for r in self._exec("SELECT id FROM tasks WHERE status = ? ORDER BY seq", (CREATED,))]

    def sweep(self, cutoff: float) -> None:
        """Delete tasks finished before `cutoff`; unfinished ones never are."""
        self._exec("DELETE FROM tasks WHERE finished_at < ?", (cutoff,))

    def close(self) -> None:
        with self._lock:
            self._conn.close()
