"""The HTTP API: submit a render task, poll it, download its PNG; GET /health reports models and queue."""

from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .jobs import COMPLETED, Cancelled, QueueFull, TaskStore
from .schemas import ValidationError, dump_request, load_images, parse_request

MAX_BODY = 64 * 1024 * 1024
TASKS = "/v1/tasks"


def png_render(registry):
    """The worker's render of a stored request: parsed again (a model may no longer be configured since a restart),
    inputs decoded, registry.render, PNG-encoded, all there, so request threads only copy bytes."""

    def render(data, on_step):
        req = parse_request(data, registry.configured)
        image, steps = registry.render(load_images(req), on_step)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue(), steps

    return render


def make_server(host: str, port: int, tasks: TaskStore, registry) -> ThreadingHTTPServer:
    """A threaded server in front of `tasks`; `registry` validates model names and reports status."""

    class Handler(BaseHTTPRequestHandler):
        timeout = 60  # socket reads only: a stalled upload frees its thread

        def send_json(self, status: int, body: dict, headers: dict | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
            self.close_connection = True
            self.send_json(code, {"error": message or explain or "error"})

        def refuse(self, e: Exception) -> None:
            if isinstance(e, QueueFull):
                return self.send_json(429, {"error": str(e)}, {"Retry-After": "30"})
            self.send_json(503, {"error": str(e)})

        def task_path(self, path: str) -> tuple[str, bool] | None:
            """(task_id, wants_image) for /v1/tasks/{id}[/image], else None."""
            parts = path.split("/")
            if len(parts) in (4, 5) and "/".join(parts[:3]) == TASKS and parts[3]:
                if len(parts) == 4:
                    return parts[3], False
                if parts[4] == "image":
                    return parts[3], True
            return None

        def do_GET(self) -> None:
            path = urlsplit(self.path).path  # a cache-buster query must not turn a live task into a 404
            if path == "/health":
                return self.send_json(200, {"device": registry.device, "models": registry.status(), **tasks.stats()})
            if path == TASKS:
                return self.send_json(405, {"error": "use POST"}, {"Allow": "POST"})
            route = self.task_path(path)
            if route is None:
                return self.send_json(404, {"error": "not found"})
            task_id, wants_image = route
            if not wants_image:
                view = tasks.get(task_id)
                return self.send_json(200, {"data": view}) if view else self.send_json(404, {"error": "unknown task"})
            found = tasks.result(task_id)
            if found is None:
                return self.send_json(404, {"error": "unknown task"})
            view, result = found
            if view["status"] != COMPLETED:
                return self.send_json(409, {"error": view["error"] or f"task is {view['status']}"})
            data, steps = result
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Model", view["model"])
            self.send_header("X-Seed", str(view["seed"]))
            self.send_header("X-Steps", str(steps))
            self.send_header("X-Elapsed-Ms", str(view["elapsed_ms"]))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            path = urlsplit(self.path).path
            if path != TASKS:
                if path == "/health" or self.task_path(path):
                    return self.send_json(405, {"error": "use GET"}, {"Allow": "GET"})
                return self.send_json(404, {"error": "not found"})
            length = self.headers.get("Content-Length")
            if length is None:
                return self.send_json(411, {"error": "Content-Length required"})
            try:
                size = int(length)
            except ValueError:
                size = -1
            if size < 0:
                return self.send_json(400, {"error": "invalid Content-Length"})
            if size > MAX_BODY:
                self.close_connection = True  # the unread body would be parsed as the next request
                return self.send_json(413, {"error": f"body larger than {MAX_BODY} bytes"})
            try:
                tasks.admit()  # a full queue answers before the body is read and its images decoded
            except (QueueFull, Cancelled) as e:
                self.close_connection = True
                return self.refuse(e)
            try:
                req = parse_request(json.loads(self.rfile.read(size)), registry.configured)
            except ValidationError as e:
                return self.send_json(400, {"error": str(e)})
            except (ValueError, RecursionError) as e:  # JSONDecodeError, UnicodeDecodeError, int digit limit, nesting
                return self.send_json(400, {"error": f"invalid JSON: {e}"})

            try:
                view = tasks.submit(dump_request(req))
            except (QueueFull, Cancelled) as e:
                return self.refuse(e)
            data = {k: view[k] for k in ("task_id", "status", "position")}
            self.send_json(202, {"data": data})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = False  # server_close() joins handlers, so shutdown still answers them
    return server
