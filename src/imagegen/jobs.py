"""Render tasks: submitted, run one at a time FIFO on a single worker thread, polled and fetched by id."""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from .store import COMPLETED, CREATED, FAILED, IN_PROGRESS, TaskDB  # noqa: F401  (statuses re-exported)


class Cancelled(Exception):
    pass


class QueueFull(Exception):
    pass


class TaskStore:
    """Tasks by id, kept in the SQLite file `db_path`. `render(data, on_step)` runs on the worker thread with the
    stored request dict and returns (png_bytes, steps, caption or None).
    """

    def __init__(self, render, db_path: str, max_queued: int = 64, ttl: float = 3600, clock=time.time) -> None:
        self._render = render
        self._db = TaskDB(db_path)
        self._max_queued = max_queued
        self._ttl = ttl
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
        self._lock = threading.Lock()
        self._busy = False
        self._closing = False

    def preload(self, fn, *args) -> Future:
        """Run fn(*args) on the worker ahead of later tasks; not a task, so never listed."""
        return self._executor.submit(self._busy_while, fn, *args)

    def recover(self) -> None:
        """After a restart: fail the task that was rendering, queue the waiting ones again in submit order."""
        with self._lock:
            for task_id in self._db.recover(self._clock()):
                self._executor.submit(self._run, task_id)
            self._sweep()

    def admit(self) -> None:
        """Raise what submit would right now: QueueFull at max_queued waiting tasks, Cancelled once shutting down."""
        with self._lock:
            self._admit()

    def submit(self, data: dict) -> dict:
        """Queue the request dict `data` (schemas.dump_request); its task view. Raises as admit() does."""
        with self._lock:
            self._admit()
            task_id = uuid.uuid4().hex
            self._db.insert(task_id, data["model"], data["seed"], json.dumps(data), self._clock())
            self._executor.submit(self._run, task_id)  # under the lock, so shutdown can't slip in between
            return self._view(self._db.get(task_id))

    def get(self, task_id: str) -> dict | None:
        self._sweep()
        row = self._db.get(task_id)
        return self._view(row) if row else None

    def result(self, task_id: str) -> tuple[dict, tuple[bytes, int] | None] | None:
        """(view, (png, steps) or None until COMPLETED); None if unknown or expired."""
        view = self.get(task_id)
        if view is None or view["status"] != COMPLETED:
            return (view, None) if view else None
        image = self._db.image(task_id)
        return (view, image) if image else None  # swept in between

    def stats(self) -> dict:
        self._sweep()
        queued, total = self._db.counts()
        return {"queued": queued, "busy": self._busy, "tasks": total}

    def shutdown(self) -> None:
        """Refuse new tasks, wait for the one rendering; queued tasks stay CREATED and resume on the next start."""
        with self._lock:
            self._closing = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._db.close()

    def _admit(self) -> None:
        if self._closing:
            raise Cancelled("server is shutting down")
        self._sweep()
        if self._db.counts()[0] >= self._max_queued:
            raise QueueFull(f"{self._max_queued} tasks already queued")

    def _busy_while(self, fn, *args):
        with self._lock:
            self._busy = True
        try:
            return fn(*args)
        finally:
            with self._lock:
                self._busy = False

    def _run(self, task_id: str) -> None:
        try:
            self._run_task(task_id)
        except Exception:  # a bare executor future: nobody would ever see this otherwise
            traceback.print_exc(file=sys.stderr)

    def _run_task(self, task_id: str) -> None:
        with self._lock:
            if self._closing:  # picked up just before shutdown cancelled the queue: leave it CREATED
                return
            request = self._db.set_started(task_id, self._clock())
        if request is None:
            return

        def on_step(i: int, n: int) -> None:
            self._db.set_step(task_id, i, n)

        try:
            (png, steps, caption), error = self._busy_while(self._render, json.loads(request), on_step), None
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            png, steps, caption, error = None, None, None, f"{type(e).__name__}: {e}"
        try:
            self._db.set_finished(task_id, self._clock(), png, steps, error, caption)
        except Exception as e:  # e.g. disk full on the PNG blob: a small FAILED write may still land
            traceback.print_exc(file=sys.stderr)
            self._db.set_finished(task_id, self._clock(), None, None, f"{type(e).__name__}: {e}")

    def _sweep(self) -> None:
        self._db.sweep(self._clock() - self._ttl)

    def _view(self, row: dict) -> dict:
        elapsed = row["started_at"] is not None and row["finished_at"] is not None
        return {
            "task_id": row["id"],
            "status": row["status"],
            "model": row["model"],
            "seed": row["seed"],
            "position": row["position"],
            "step": row["step"],
            "steps": row["steps"],
            "error": row["error"],
            "caption": row["caption"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "elapsed_ms": round((row["finished_at"] - row["started_at"]) * 1000) if elapsed else None,
        }
