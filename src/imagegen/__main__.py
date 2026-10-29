"""Run the server: uv run --env-file .env image-gen-server"""

from __future__ import annotations

import fcntl
import os
import signal
import sys
import threading

from .api import make_server, png_render
from .config import load_settings
from .jobs import TaskStore

def main() -> None:
    try:
        settings = load_settings()
    except ValueError as e:
        print(f"image-gen-server: {e}", file=sys.stderr)
        sys.exit(2)

    from .engine.device import pick_device  # pulls in torch, so only once the settings are good
    from .engine.registry import ModelRegistry

    try:
        device = pick_device(settings.device)
    except ValueError as e:
        print(f"image-gen-server: {e}", file=sys.stderr)
        sys.exit(2)
    registry = ModelRegistry(settings.model_paths, device)
    tasks = TaskStore(png_render(registry), settings.db_path, settings.max_queued, settings.task_ttl)
    lock = open(settings.db_path + ".lock", "w")  # held until exit: a second server on this db would recover() our rows
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"image-gen-server: {settings.db_path} is in use by another server", file=sys.stderr)
        os._exit(2)  # not sys.exit: the executor's exit hook would wait on the worker

    def report(future) -> None:
        if not future.cancelled() and future.exception():
            print(f"preload failed: {future.exception()!r}", file=sys.stderr, flush=True)

    for name in settings.preload:
        tasks.preload(registry.get, name).add_done_callback(report)
    tasks.recover()  # after the preloads, before serving: the last run's waiting tasks queue ahead of new ones
    server = make_server(settings.host, settings.port, tasks, registry)

    def on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_sigterm)
    print(
        f"Serving on http://{settings.host}:{server.server_address[1]} ({device}, {', '.join(registry.configured)})",
        flush=True,
    )
    # served off the main thread, so polls are still answered while shutdown waits for the running render
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        signal.pause()
    except KeyboardInterrupt:
        print("Stopping", flush=True)
    finally:
        # ponytail: a poll landing between the db closing and server.shutdown() gets a dropped connection; it retries
        try:
            tasks.shutdown()
        except KeyboardInterrupt:  # second Ctrl+C: skip joining the render; the next start fails that task
            os._exit(130)
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
