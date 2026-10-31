"""API, queue, config and schema tests with a fake registry: no torch, no weights."""

import base64
import http.client
import io
import json
import os
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error
import urllib.request

from PIL import Image

from imagegen.api import make_server, model_views, png_render, to_jpeg
from imagegen.config import FamilySettings, load_settings
from imagegen.jobs import TaskStore
from imagegen.schemas import MODELS, GenerateRequest, ValidationError, dump_request, load_images, parse_request
from imagegen.store import TaskDB

KEY = "test-key-0123456789"
AUTH = {"Authorization": f"Bearer {KEY}"}


class FakeRegistry:
    device = "cpu"
    families = {"qwen": FamilySettings("int8", "bf16", 7 * 1024**2), "flux": FamilySettings("bf16", "fp8", 5 * 1024**2),
                "fibo": FamilySettings("int8", "int8", 3 * 1024**2)}
    precision = {"qwen": {"transformer": "int8", "text_encoder": "bf16"}, "flux": {"transformer": "bf16", "text_encoder": "fp8"},
                 "fibo": {"transformer": "int8", "text_encoder": "int8"}}

    def __init__(self, configured=tuple(MODELS), can_upsample=True):
        self.configured = configured
        self.can_upsample = can_upsample
        self.log = []  # (event, prompt)
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.gate = threading.Event()  # "hold" renders wait on it after step 1

    def status(self):
        return {n: {"configured": n in self.configured, "loaded": False} for n in MODELS}

    def upsampler_status(self):
        return {"configured": self.can_upsample, "loaded": False}

    def render(self, req: GenerateRequest, on_step=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.log.append(("start", req.prompt))
        try:
            assert all(isinstance(img, Image.Image) for img in req.images)  # decoded on the worker by png_render
            on_step(1, 4)
            if req.prompt == "hold":
                self.gate.wait(5)
            time.sleep(0.02)
            if req.prompt == "boom":
                raise RuntimeError("kaput")
            on_step(4, 4)
            caption = f"caption of {req.prompt}" if req.upsample else None
            return Image.new("RGB", (8, 8), "red"), 8 if req.model == "qwen-turbo" else req.steps, caption
        finally:
            with self.lock:
                self.active -= 1
                self.log.append(("end", req.prompt))


def b64png(size=(4, 4), corrupt_at=None) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size).save(buf, format="PNG")
    raw = bytearray(buf.getvalue())
    if corrupt_at is not None:
        raw[corrupt_at] = 0  # breaks the IDAT chunk header: Pillow raises SyntaxError on load()
    return base64.b64encode(raw).decode()


class ServerCase(unittest.TestCase):
    configured = tuple(MODELS)
    can_upsample = True
    max_queued = 64

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "data", "tasks.sqlite3")
        self.registry = FakeRegistry(self.configured, self.can_upsample)
        self.now = [1000.0]
        self.tasks = TaskStore(png_render(self.registry), self.db_path, self.max_queued, ttl=60, clock=lambda: self.now[0])
        self.server = make_server("127.0.0.1", 0, self.tasks, self.registry, KEY)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.registry.gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.tasks.shutdown()

    def request(self, method, path, body=None, raw=None, headers=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=headers or AUTH)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def submit(self, body):
        status, _, data = self.request("POST", "/v1/tasks", body)
        self.assertEqual(status, 202, data)
        return json.loads(data)["data"]

    def task(self, task_id):
        status, _, data = self.request("GET", f"/v1/tasks/{task_id}")
        self.assertEqual(status, 200, data)
        return json.loads(data)["data"]

    def wait(self, task_id, *statuses):
        deadline = time.monotonic() + 5
        while (t := self.task(task_id))["status"] not in statuses:
            self.assertLess(time.monotonic(), deadline, t)
            time.sleep(0.01)
        return t

    def assert400(self, body, raw=None, path="/v1/tasks"):
        status, headers, data = self.request("POST", path, body, raw)
        self.assertEqual(status, 400, data)
        self.assertIn("error", json.loads(data))
        self.assertEqual(headers["Content-Type"], "application/json")
        return json.loads(data)["error"]

    def models(self):
        status, _, data = self.request("GET", "/v1/models")
        self.assertEqual(status, 200, data)
        return {m["id"]: m for m in json.loads(data)["data"]}


class TestApi(ServerCase):
    def test_bad_requests(self):
        for body in [
            {"prompt": ""},
            {"prompt": "x", "ratio": "5:4"},
            {"prompt": "x", "ratio": ["1:1"]},
            {"prompt": "x", "ratio": {}},
            {"prompt": "x", "images": [b64png(corrupt_at=36)]},
            {"prompt": "x", "images": [b64png((4000, 4000))]},
            {"prompt": "x", "width": 1000},
            {"prompt": "x", "width": True},
            {"prompt": "x", "steps": 0},
            {"prompt": "x", "steps": 101},
            {"prompt": "x", "images": [b64png()] * 11},
            {"prompt": "x", "images": ["not base64!!"]},
            {"prompt": "x", "images": [base64.b64encode(b"not an image").decode()]},
            {"prompt": "x", "colour": "red"},
            {"prompt": "x", "transparent": 1},
            ["prompt"],
        ]:
            with self.subTest(body=body):
                self.assert400(body)
        self.assert400(None, raw=b"{not json")
        self.assert400(None, raw=b"[" * 100000)
        self.assert400(None, raw=b'{"prompt": "x", "seed": ' + b"1" * 5000 + b"}")
        self.assertEqual(self.registry.log, [])
        self.assertEqual(self.tasks.stats()["tasks"], 0)

    def test_bad_requests_per_model(self):
        for fields, error in [
            ({"model": "flux-klein", "height": 72}, "height must be an integer in 64..4096, a multiple of 16"),
            ({"model": "flux-klein", "width": 4112}, "width must be an integer in 64..4096, a multiple of 16"),
            ({"model": "qwen-turbo", "width": 4096}, "width must be an integer in 256..2752, a multiple of 32"),
            ({"model": "qwen-turbo", "width": 1040}, "width must be an integer in 256..2752, a multiple of 32"),
            ({"model": "flux-klein", "ratio": "5:4"}, "ratio must be one of"),
            ({"model": "qwen-base", "ratio": "5:4"}, "ratio must be one of"),
            ({"model": "flux-klein", "transparent": True}, "model 'flux-klein' has no transparent option"),
            ({"model": "qwen-turbo", "upsample": True}, "model 'qwen-turbo' has no upsample option"),
            ({"model": "qwen-base", "upsample": True}, "model 'qwen-base' has no upsample option"),
            ({"model": "flux-klein", "upsample": "yes"}, "upsample must be a boolean"),
            ({"model": "flux-klein", "upsample": 1}, "upsample must be a boolean"),
            ({"model": "qwen-turbo", "transparent": "yes"}, "transparent must be a boolean"),
            ({"model": "flux-klein", "transparent": None}, "transparent must be a boolean"),
            ({"model": ["flux-klein"]}, "model must be one of"),
            ({"model": {"flux-klein": 1}}, "model must be one of"),
            ({"model": None}, "model must be one of"),
            ({"model": "huge"}, "model must be one of"),
            ({"model": "flux-klein", "steps": 101}, "steps must be an integer in 1..100"),
            ({"model": "flux-klein", "images": [b64png((8192, 4097))]}, "over 33554432 pixels"),
            ({"model": "fibo-edit-turbo"}, "model 'fibo-edit-turbo' edits images: send at least one in images"),
            ({"model": "fibo-edit-turbo", "images": []}, "model 'fibo-edit-turbo' edits images"),
            ({"model": "fibo-edit-turbo", "images": [b64png()] * 5}, "at most 4 images"),
            ({"model": "fibo-edit-turbo", "images": [b64png()], "width": 2768}, "width must be an integer in 64..2752, a multiple of 16"),
            ({"model": "fibo-edit-turbo", "images": [b64png()], "height": 72}, "height must be an integer in 64..2752, a multiple of 16"),
            ({"model": "fibo-edit-turbo", "images": [b64png()], "transparent": True}, "model 'fibo-edit-turbo' has no transparent option"),
            ({"model": "fibo-edit-turbo", "images": [b64png()], "upsample": True}, "model 'fibo-edit-turbo' has no upsample option"),
        ]:
            with self.subTest(fields=str(fields)[:80]):
                self.assertIn(error, self.assert400({"prompt": "x", **fields}))
        self.assertEqual(self.tasks.stats()["tasks"], 0)

    def test_api_key_required(self):
        task_id = self.submit({"prompt": "cat"})["task_id"]
        self.wait(task_id, "COMPLETED")
        before = json.loads(self.request("GET", "/health")[2])["tasks"]
        for headers in ({"X-None": "1"}, {"Authorization": "Bearer wrong"}, {"Authorization": KEY},
                        {"Authorization": f"Bearer {KEY}x"}, {"Authorization": "Bearer "}):
            for method, path, body in (("POST", "/v1/tasks", {"prompt": "x"}), ("GET", f"/v1/tasks/{task_id}", None),
                                       ("GET", f"/v1/tasks/{task_id}/image", None),
                                       ("GET", f"/v1/tasks/{task_id}/image?format=jpeg", None), ("GET", "/v1/nope", None),
                                       ("GET", "/v1/models", None), ("GET", "/v1/models/flux-klein/tasks", None),
                                       ("POST", "/v1/models/flux-klein/tasks", {"prompt": "x"}),
                                       ("POST", "/v1/models/nope/tasks", {"prompt": "x"})):
                with self.subTest(headers=headers, path=path):
                    status, response, data = self.request(method, path, body, headers=headers)
                    self.assertEqual(status, 401, data)
                    self.assertEqual(response["WWW-Authenticate"], "Bearer")
                    self.assertEqual(json.loads(data), {"error": "missing or invalid API key"})
        self.assertEqual(json.loads(self.request("GET", "/health")[2])["tasks"], before)  # none of them queued a task
        self.assertEqual(self.request("GET", "/health", headers={"X-None": "1"})[0], 200)  # open, for uptime checks

    def test_settings_hide_api_key(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        settings = load_settings({"QWEN_IMAGE_21_PATH": tmp.name, "IMAGEGEN_API_KEY": f"  {KEY} "})
        self.assertEqual(settings.api_key, KEY)
        self.assertNotIn(KEY, repr(settings))

    def test_content_length(self):
        for headers, expected in [({}, 411), ({"Content-Length": "-5"}, 400), ({"Content-Length": "x"}, 400),
                                  ({"Content-Length": str(64 * 1024 * 1024 + 1)}, 413)]:
            with self.subTest(headers=headers):
                conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
                conn.putrequest("POST", "/v1/tasks")
                conn.putheader("Authorization", AUTH["Authorization"])
                for k, v in headers.items():
                    conn.putheader(k, v)
                conn.endheaders()
                r = conn.getresponse()
                self.assertEqual(r.status, expected)
                self.assertIn("error", json.loads(r.read()))
                conn.close()

    def test_submit_poll_download(self):
        created = self.submit({"prompt": "cat", "images": [b64png()], "seed": 7})
        self.assertEqual(set(created), {"task_id", "status", "position"})
        self.assertEqual((created["status"], created["position"]), ("CREATED", 0))
        self.assertEqual(len(created["task_id"]), 32)
        done = self.wait(created["task_id"], "COMPLETED")
        self.assertEqual((done["model"], done["seed"], done["position"], done["error"]), ("qwen-turbo", 7, None, None))
        self.assertEqual((done["step"], done["steps"]), (4, 4))
        self.assertIsInstance(done["elapsed_ms"], int)
        self.assertTrue(done["created_at"] <= done["started_at"] <= done["finished_at"])
        self.assertEqual(self.request("GET", f"/v1/tasks/{created['task_id']}?t=1")[0], 200)  # query ignored
        self.assertEqual(self.request("GET", "/health?x")[0], 200)
        for _ in range(2):  # fetching does not consume the result
            status, headers, data = self.request("GET", f"/v1/tasks/{created['task_id']}/image?t=1")
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "image/png")
            self.assertEqual(int(headers["Content-Length"]), len(data))
            self.assertEqual((headers["X-Model"], headers["X-Steps"], headers["X-Seed"]), ("qwen-turbo", "8", "7"))
            self.assertIn("X-Elapsed-Ms", headers)
            self.assertEqual(Image.open(io.BytesIO(data)).size, (8, 8))

    def test_qwen_turbo(self):
        task_id = self.submit({"prompt": "cat", "model": "qwen-turbo", "steps": 30})["task_id"]
        self.wait(task_id, "COMPLETED")
        status, headers, _ = self.request("GET", f"/v1/tasks/{task_id}/image")
        self.assertEqual(status, 200)
        self.assertEqual((headers["X-Model"], headers["X-Steps"]), ("qwen-turbo", "8"))

    def test_flux_klein_steps_cut(self):
        for fields, steps in (({"steps": 40}, "16"), ({}, "16"), ({"steps": 8}, "8")):
            with self.subTest(fields=fields):
                task_id = self.submit({"prompt": "cat", "model": "flux-klein", **fields})["task_id"]
                self.wait(task_id, "COMPLETED")
                _, headers, _ = self.request("GET", f"/v1/tasks/{task_id}/image")
                self.assertEqual((headers["X-Model"], headers["X-Steps"]), ("flux-klein", steps))

    def test_caption(self):
        task_id = self.submit({"prompt": "cat", "model": "flux-klein", "upsample": True})["task_id"]
        self.assertEqual(self.wait(task_id, "COMPLETED")["caption"], "caption of cat")
        for fields in ({"model": "flux-klein"}, {"model": "flux-klein", "upsample": False}, {"model": "qwen-turbo"}):
            with self.subTest(fields=fields):
                task_id = self.submit({"prompt": "cat", **fields})["task_id"]
                self.assertIsNone(self.wait(task_id, "COMPLETED")["caption"])
        task_id = self.submit({"prompt": "boom", "model": "flux-klein", "upsample": True})["task_id"]
        self.assertIsNone(self.wait(task_id, "FAILED")["caption"])

    def test_progress_and_image_before_done(self):
        task_id = self.submit({"prompt": "hold"})["task_id"]
        self.wait(task_id, "IN_PROGRESS")
        while (running := self.task(task_id))["step"] is None:  # status flips just before the first step
            time.sleep(0.01)
        self.assertEqual((running["step"], running["steps"], running["position"]), (1, 4, None))
        self.assertIsNotNone(running["started_at"])
        self.assertIsNone(running["elapsed_ms"])
        queued = self.submit({"prompt": "next"})["task_id"]
        for tid in (task_id, queued):
            status, _, data = self.request("GET", f"/v1/tasks/{tid}/image")
            self.assertEqual(status, 409)
            self.assertIn("error", json.loads(data))
        self.assertIsNone(self.task(queued)["step"])
        self.registry.gate.set()
        self.wait(queued, "COMPLETED")

    def test_fifo_one_at_a_time_positions(self):
        first = self.submit({"prompt": "hold"})["task_id"]
        self.wait(first, "IN_PROGRESS")
        ids = []
        for i in range(3):
            created = self.submit({"prompt": f"p{i}"})
            self.assertEqual(created["position"], i)
            ids.append(created["task_id"])
        self.assertEqual([self.task(t)["position"] for t in ids], [0, 1, 2])
        self.registry.gate.set()
        for t in ids:
            self.wait(t, "COMPLETED")
        self.assertEqual(self.registry.max_active, 1)
        expected = [(e, p) for p in ("hold", "p0", "p1", "p2") for e in ("start", "end")]
        self.assertEqual(self.registry.log, expected)

    def test_render_error_then_ok(self):
        bad = self.submit({"prompt": "boom"})["task_id"]
        good = self.submit({"prompt": "fine"})["task_id"]
        failed = self.wait(bad, "FAILED", "COMPLETED")
        self.assertEqual((failed["status"], failed["error"]), ("FAILED", "RuntimeError: kaput"))
        status, _, data = self.request("GET", f"/v1/tasks/{bad}/image")
        self.assertEqual((status, json.loads(data)["error"]), (409, "RuntimeError: kaput"))
        self.wait(good, "COMPLETED")

    def test_image_jpeg(self):
        task_id = self.submit({"prompt": "cat"})["task_id"]
        self.wait(task_id, "COMPLETED")
        png = self.request("GET", f"/v1/tasks/{task_id}/image")[2]
        status, headers, data = self.request("GET", f"/v1/tasks/{task_id}/image?format=jpeg&t=1")
        self.assertEqual((status, headers["Content-Type"]), (200, "image/jpeg"))
        self.assertEqual(int(headers["Content-Length"]), len(data))
        self.assertIn("X-Elapsed-Ms", headers)
        with Image.open(io.BytesIO(data)) as jpeg, Image.open(io.BytesIO(png)) as original:
            self.assertEqual((jpeg.format, jpeg.size), ("JPEG", original.size))
        self.assertEqual(self.request("GET", f"/v1/tasks/{task_id}/image?format=png")[2], png)
        status, _, data = self.request("GET", f"/v1/tasks/{task_id}/image?format=webp")
        self.assertEqual((status, json.loads(data)["error"]), (400, "format must be one of png, jpeg"))

    def test_jpeg_flattens_alpha_onto_white(self):
        buf = io.BytesIO()
        Image.new("RGBA", (16, 16), (255, 0, 0, 0)).save(buf, "PNG")
        with Image.open(io.BytesIO(to_jpeg(buf.getvalue()))) as jpeg:
            self.assertEqual((jpeg.mode, jpeg.size), ("RGB", (16, 16)))
            self.assertTrue(all(channel > 250 for channel in jpeg.getpixel((8, 8))))

    def test_unknown_and_expired(self):
        for path in ("/v1/tasks/nope", "/v1/tasks/nope/image"):
            self.assertEqual(self.request("GET", path)[0], 404)
        task_id = self.submit({"prompt": "cat"})["task_id"]
        self.wait(task_id, "COMPLETED")
        self.now[0] += 59
        self.assertEqual(self.request("GET", f"/v1/tasks/{task_id}/image")[0], 200)
        self.now[0] += 2
        self.assertEqual(self.request("GET", f"/v1/tasks/{task_id}")[0], 404)
        self.assertEqual(self.request("GET", f"/v1/tasks/{task_id}/image")[0], 404)
        self.assertEqual(self.tasks.stats()["tasks"], 0)

    def test_health(self):
        status, _, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(
            json.loads(data),
            {
                "device": "cpu",
                "precision": {
                    "qwen": {"transformer": "int8", "text_encoder": "bf16"},
                    "flux": {"transformer": "bf16", "text_encoder": "fp8"},
                    "fibo": {"transformer": "int8", "text_encoder": "int8"},
                },
                "models": {
                    "qwen-base": {"configured": True, "loaded": False},
                    "qwen-turbo": {"configured": True, "loaded": False},
                    "flux-klein": {"configured": True, "loaded": False},
                    "fibo-edit-turbo": {"configured": True, "loaded": False},
                },
                "upsampler": {"configured": True, "loaded": False},
                "queued": 0,
                "busy": False,
                "tasks": 0,
            },
        )
        first = self.submit({"prompt": "hold"})["task_id"]
        self.wait(first, "IN_PROGRESS")
        self.submit({"prompt": "next"})
        health = json.loads(self.request("GET", "/health")[2])
        self.assertEqual((health["queued"], health["busy"], health["tasks"]), (1, True, 2))

    def test_404_405(self):
        for method, path, expected in [("GET", "/nope", 404), ("POST", "/nope", 404), ("GET", "/generate", 404),
                                       ("POST", "/generate", 404), ("GET", "/v1/tasks/x/y", 404),
                                       ("GET", "/v1/tasks/", 404), ("GET", "/v1/tasks", 405),
                                       ("POST", "/v1/tasks/x", 405), ("POST", "/health", 405),
                                       ("POST", "/v1/models", 405), ("GET", "/v1/models/flux-klein/tasks", 405),
                                       ("GET", "/v1/models/flux-klein", 404), ("POST", "/v1/models/flux-klein", 404),
                                       ("POST", "/v1/models/flux-klein/tasks/x", 404), ("POST", "/v1/models//tasks", 404)]:
            with self.subTest(method=method, path=path):
                status, headers, data = self.request(method, path, {} if method == "POST" else None)
                self.assertEqual(status, expected)
                self.assertEqual(int(headers["Content-Length"]), len(data))
                self.assertIn("error", json.loads(data))

    def test_other_methods_json(self):
        status, headers, data = self.request("PUT", "/v1/tasks", {})
        self.assertEqual(status, 501)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertIn("error", json.loads(data))

    def test_shutdown_then_restart(self):
        running = self.submit({"prompt": "hold"})["task_id"]
        self.wait(running, "IN_PROGRESS")
        queued = [self.submit({"prompt": f"p{i}"})["task_id"] for i in range(2)]
        stopper = threading.Thread(target=self.tasks.shutdown)
        stopper.start()
        status, _, data = self.request("POST", "/v1/tasks", {"prompt": "late"})
        self.assertEqual(status, 503, data)
        self.assertEqual([self.task(t)["status"] for t in queued], ["CREATED", "CREATED"])  # polls still answered
        self.assertTrue(stopper.is_alive())  # still waiting on the running render
        self.registry.gate.set()
        stopper.join(5)
        self.assertEqual(self.registry.log, [("start", "hold"), ("end", "hold")])

        registry = FakeRegistry()
        tasks = TaskStore(png_render(registry), self.db_path, clock=lambda: self.now[0])
        self.addCleanup(tasks.shutdown)
        self.assertEqual(tasks.get(running)["status"], "COMPLETED")  # written before the db closed
        self.assertEqual([tasks.get(t)["position"] for t in queued], [0, 1])
        tasks.recover()
        for t in queued:
            wait_store(self, tasks, t, "COMPLETED")
        self.assertEqual([p for e, p in registry.log if e == "start"], ["p0", "p1"])


class TestFiboSize(ServerCase):
    def test_accepts_sizes_it_cannot_render_whole(self):
        body = {"prompt": "x", "model": "fibo-edit-turbo", "images": [b64png()], "width": 2304, "height": 1824}
        self.wait(self.submit(body)["task_id"], "COMPLETED")


class TestModels(ServerCase):
    def test_lists_models_with_limits(self):
        models = self.models()
        self.assertEqual(list(models), ["qwen-base", "qwen-turbo", "flux-klein", "fibo-edit-turbo"])
        self.assertEqual(models["flux-klein"], {
            "id": "flux-klein", "loaded": False, "max_images": 10, "max_input_pixels": 33554432, "family": "flux",
            "precision": {"transformer": "bf16", "text_encoder": "fp8"}, "max_reference_pixels": 5 * 1024**2,
            "ratios": {name: list(size) for name, size in MODELS["flux-klein"].ratios.items()},
            "min_side": 64, "max_side": 4096, "side_multiple": 16, "steps": 16, "max_steps": 16, "options": ["upsample"],
            "needs_image": False, "max_output_pixels": None,
        })
        self.assertEqual(models["flux-klein"]["ratios"]["16:9"], [1360, 768])
        turbo = models["qwen-turbo"]
        self.assertEqual((turbo["family"], turbo["min_side"], turbo["max_side"], turbo["side_multiple"]), ("qwen", 256, 2752, 32))
        self.assertEqual((turbo["steps"], turbo["max_steps"], turbo["options"]), (40, 100, ["transparent"]))
        self.assertEqual(turbo["ratios"]["16:9"], [2752, 1536])
        self.assertEqual((turbo["max_images"], turbo["needs_image"]), (10, False))
        self.assertIsNone(turbo["max_output_pixels"])
        self.assertEqual(models["fibo-edit-turbo"], {
            "id": "fibo-edit-turbo", "loaded": False, "max_images": 4, "max_input_pixels": 15147008, "family": "fibo",
            "precision": {"transformer": "int8", "text_encoder": "int8"}, "max_reference_pixels": 3 * 1024**2,
            "ratios": {name: list(size) for name, size in MODELS["flux-klein"].ratios.items()},
            "min_side": 64, "max_side": 2752, "side_multiple": 16, "steps": 4, "max_steps": 4, "options": [],
            "needs_image": True, "max_output_pixels": 1572864,
        })

    def test_fibo_edit_turbo_route(self):
        status, _, data = self.request("POST", "/v1/models/fibo-edit-turbo/tasks", {"prompt": "make it red", "images": [b64png()] * 4})
        self.assertEqual(status, 202)
        self.assertEqual(self.wait(json.loads(data)["data"]["task_id"], "COMPLETED")["model"], "fibo-edit-turbo")
        status, _, _ = self.request("POST", "/v1/models/fibo-edit-turbo/tasks", {"prompt": "make it red"})
        self.assertEqual(status, 400)

    def test_settings_per_family(self):
        models = self.models()
        for name in ("qwen-base", "qwen-turbo"):
            with self.subTest(model=name):
                self.assertEqual(models[name]["precision"], {"transformer": "int8", "text_encoder": "bf16"})
                self.assertEqual(models[name]["max_reference_pixels"], 7 * 1024**2)

    def test_model_route(self):
        status, _, data = self.request("POST", "/v1/models/flux-klein/tasks", {"prompt": "cat", "upsample": True})
        self.assertEqual(status, 202, data)
        created = json.loads(data)["data"]
        self.assertEqual(set(created), {"task_id", "status", "position"})
        done = self.wait(created["task_id"], "COMPLETED")
        self.assertEqual((done["model"], done["caption"]), ("flux-klein", "caption of cat"))

    def test_url_overrides_body_model(self):
        for model in ("qwen-turbo", "huge", ["flux-klein"], None):
            with self.subTest(model=model):
                status, _, data = self.request("POST", "/v1/models/flux-klein/tasks", {"prompt": "cat", "model": model})
                self.assertEqual(status, 202, data)
                self.assertEqual(self.wait(json.loads(data)["data"]["task_id"], "COMPLETED")["model"], "flux-klein")
        error = self.assert400({"prompt": "x", "model": "flux-klein", "upsample": True}, path="/v1/models/qwen-turbo/tasks")
        self.assertEqual(error, "model 'qwen-turbo' has no upsample option")  # the URL's model is the one validated

    def test_unknown_model_404(self):
        status, _, data = self.request("POST", "/v1/models/huge/tasks", {"prompt": "x"})
        self.assertEqual((status, json.loads(data)["error"]), (404, "model 'huge' is not configured"))

    def test_405_allow(self):
        self.assertEqual(self.request("GET", "/v1/models/flux-klein/tasks")[1]["Allow"], "POST")
        self.assertEqual(self.request("POST", "/v1/models", {})[1]["Allow"], "GET")

    def test_bad_bodies_400(self):
        path = "/v1/models/flux-klein/tasks"
        for body in (["prompt"], "cat", 7, None):
            with self.subTest(body=body):
                error = self.assert400(None, raw=json.dumps(body).encode(), path=path)
                self.assertEqual(error, "request body must be a JSON object")
        self.assertIn("invalid JSON", self.assert400(None, raw=b"{not json", path=path))
        self.assertEqual(self.tasks.stats()["tasks"], 0)


class TestNoUpsampler(ServerCase):
    can_upsample = False

    def test_upsample_refused(self):
        error = self.assert400({"prompt": "x", "model": "flux-klein", "upsample": True})
        self.assertEqual(error, "the prompt upsampler is not configured")
        self.wait(self.submit({"prompt": "x", "model": "flux-klein", "upsample": False})["task_id"], "COMPLETED")
        self.assertEqual(self.registry.log[0], ("start", "x"))

    def test_views_hide_upsample(self):
        models = self.models()
        self.assertEqual(models["flux-klein"]["options"], [])
        self.assertEqual(models["qwen-turbo"]["options"], ["transparent"])
        self.assertEqual(MODELS["flux-klein"].options, ("upsample",))  # the shared table untouched
        health = json.loads(self.request("GET", "/health")[2])
        self.assertEqual(health["upsampler"], {"configured": False, "loaded": False})


class TestOldModelNames(ServerCase):
    def test_rejected(self):
        for old in ("base", "turbo", "klein"):
            with self.subTest(old=old):
                self.assertIn("model must be one of", self.assert400({"prompt": "x", "model": old}))
                status, _, data = self.request("POST", f"/v1/models/{old}/tasks", {"prompt": "x"})
                self.assertEqual((status, json.loads(data)["error"]), (404, f"model {old!r} is not configured"))


class TestFluxKleinOnly(ServerCase):
    configured = ("flux-klein",)

    def test_default_model(self):
        done = self.wait(self.submit({"prompt": "cat"})["task_id"], "COMPLETED")
        self.assertEqual(done["model"], "flux-klein")
        _, headers, _ = self.request("GET", f"/v1/tasks/{done['task_id']}/image")
        self.assertEqual(headers["X-Steps"], "16")

    def test_only_flux_klein_served(self):
        self.assertEqual(list(self.models()), ["flux-klein"])
        for model in ("qwen-base", "qwen-turbo"):
            with self.subTest(model=model):
                status, _, data = self.request("POST", f"/v1/models/{model}/tasks", {"prompt": "x"})
                self.assertEqual((status, json.loads(data)["error"]), (404, f"model {model!r} is not configured"))
        models = json.loads(self.request("GET", "/health")[2])["models"]
        self.assertEqual({n for n, m in models.items() if m["configured"]}, {"flux-klein"})
        self.assertEqual(set(models), {"qwen-base", "qwen-turbo", "flux-klein", "fibo-edit-turbo"})


class TestQueueFull(ServerCase):
    max_queued = 2

    def test_429(self):
        first = self.submit({"prompt": "hold"})["task_id"]
        self.wait(first, "IN_PROGRESS")  # running tasks don't count against MAX_QUEUED
        self.submit({"prompt": "a"})
        self.submit({"prompt": "b"})
        status, headers, data = self.request("POST", "/v1/tasks", {"prompt": "c"})
        self.assertEqual(status, 429)
        self.assertEqual(headers["Retry-After"], "30")
        self.assertIn("error", json.loads(data))
        status, _, _ = self.request("POST", "/v1/tasks?x=1", raw=b"{not json")  # refused before the body is parsed
        self.assertEqual(status, 429)


class TestQwenBaseOnly(ServerCase):
    configured = ("qwen-base",)

    def test_qwen_turbo_not_configured(self):
        status, _, data = self.request("POST", "/v1/tasks", {"prompt": "x", "model": "qwen-turbo"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"], "model 'qwen-turbo' is not configured")

    def test_default_model(self):
        self.assertEqual(self.wait(self.submit({"prompt": "cat"})["task_id"], "COMPLETED")["model"], "qwen-base")

    def test_only_qwen_base_served(self):
        self.assertEqual(list(self.models()), ["qwen-base"])
        self.assertEqual(self.request("POST", "/v1/models/flux-klein/tasks", {"prompt": "x"})[0], 404)


class TestPreload(unittest.TestCase):
    def test_runs_first_and_is_not_a_task(self):
        order = []
        store = TaskStore(lambda data, on_step: order.append(data["prompt"]) or (b"png", 1, None), ":memory:")
        started, gate = threading.Event(), threading.Event()
        preload = store.preload(lambda: (started.set(), gate.wait(5), order.append("preload")))
        task = store.submit(dump_request(GenerateRequest(prompt="a")))
        started.wait(5)
        self.assertEqual(store.stats(), {"queued": 1, "busy": True, "tasks": 1})
        gate.set()
        preload.result(5)
        wait_store(self, store, task["task_id"], "COMPLETED")
        store.shutdown()
        self.assertEqual(order, ["preload", "a"])


def wait_store(case, store, task_id, *statuses):
    deadline = time.monotonic() + 5
    while (t := store.get(task_id))["status"] not in statuses:
        case.assertLess(time.monotonic(), deadline, t)
        time.sleep(0.01)
    return t


class TestPersistence(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "tasks.sqlite3")

    def store(self, registry, **kw):
        store = TaskStore(png_render(registry), self.path, **kw)
        self.addCleanup(store.shutdown)
        return store

    def submit(self, store, prompt, **fields):
        return store.submit(dump_request(GenerateRequest(prompt=prompt, **{"model": "qwen-base", **fields})))["task_id"]

    def test_crash_recovery(self):
        crashed = FakeRegistry()
        self.addCleanup(crashed.gate.set)  # before the old store's shutdown, which waits on it
        old = self.store(crashed)
        done = self.submit(old, "done")
        wait_store(self, old, done, "COMPLETED")
        running = self.submit(old, "hold")
        wait_store(self, old, running, "IN_PROGRESS")
        queued = [self.submit(old, f"p{i}", seed=i) for i in range(3)]

        registry = FakeRegistry()  # the process died: a fresh store on the same file, no shutdown of the old one
        new = self.store(registry)
        new.recover()
        failed = new.get(running)
        self.assertEqual((failed["status"], failed["error"]), ("FAILED", "server restarted during render"))
        for t in queued:
            wait_store(self, new, t, "COMPLETED")
        self.assertEqual([p for e, p in registry.log if e == "start"], ["p0", "p1", "p2"])
        self.assertEqual([new.get(t)["seed"] for t in queued], [0, 1, 2])
        view, (png, steps) = new.result(done)
        self.assertEqual((view["status"], steps), ("COMPLETED", 40))
        self.assertEqual(Image.open(io.BytesIO(png)).size, (8, 8))
        crashed.gate.set()  # the stale render finishing must not overwrite the recovered row
        wait_store(self, old, running, "FAILED")
        time.sleep(0.1)
        self.assertEqual(new.get(running)["error"], "server restarted during render")

    def test_unconfigured_model_fails_on_recovery(self):
        old = self.store(FakeRegistry(), max_queued=8)
        old.shutdown()  # nothing ran: shut down before the worker could pick anything
        db = TaskDB(self.path)
        db.insert("t1", "qwen-turbo", 42, json.dumps(dump_request(GenerateRequest(prompt="x", model="qwen-turbo"))), 1.0)
        db.insert("t2", "qwen-base", 42, json.dumps(dump_request(GenerateRequest(prompt="y", model="qwen-base", images=(base64.b64decode(b64png()),)))), 2.0)
        db.close()
        registry = FakeRegistry(configured=("qwen-base",))
        new = self.store(registry)
        new.recover()
        failed = wait_store(self, new, "t1", "FAILED", "COMPLETED")
        self.assertEqual((failed["status"], failed["error"]), ("FAILED", "ValidationError: model 'qwen-turbo' is not configured"))
        wait_store(self, new, "t2", "COMPLETED")  # its stored image round-tripped through base64

    def test_unconfigured_upsampler_fails_on_recovery(self):
        old = self.store(FakeRegistry(), max_queued=8)
        old.shutdown()  # nothing ran: shut down before the worker could pick anything
        db = TaskDB(self.path)
        db.insert("t1", "flux-klein", 42, json.dumps(dump_request(GenerateRequest(prompt="x", model="flux-klein", steps=16, upsample=True))), 1.0)
        db.insert("t2", "flux-klein", 42, json.dumps(dump_request(GenerateRequest(prompt="y", model="flux-klein", steps=16))), 2.0)
        db.close()
        new = self.store(FakeRegistry(can_upsample=False))
        new.recover()
        failed = wait_store(self, new, "t1", "FAILED", "COMPLETED")
        self.assertEqual((failed["status"], failed["error"]), ("FAILED", "ValidationError: the prompt upsampler is not configured"))
        wait_store(self, new, "t2", "COMPLETED")

    def test_db_from_before_captions(self):
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE tasks (seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, status TEXT NOT NULL,"
                     " model TEXT NOT NULL, seed INTEGER NOT NULL, request TEXT, created_at REAL NOT NULL, started_at REAL,"
                     " finished_at REAL, step INTEGER, steps INTEGER, error TEXT, image BLOB, steps_run INTEGER)")
        conn.execute("INSERT INTO tasks (id, status, model, seed, created_at) VALUES ('old', 'COMPLETED', 'qwen-turbo', 1, 1.0)")
        conn.commit()
        conn.close()
        for _ in range(2):  # the second open finds the column already there
            db = TaskDB(self.path)
            self.addCleanup(db.close)
            self.assertIsNone(db.get("old")["caption"])
        db.insert("new", "flux-klein", 1, "{}", 2.0)
        db.set_started("new", 3.0)
        db.set_finished("new", 4.0, b"png", 16, None, "a caption")
        self.assertEqual(db.get("new")["caption"], "a caption")

    def test_request_dropped_once_started(self):
        registry = FakeRegistry()
        self.addCleanup(registry.gate.set)
        store = self.store(registry)
        running = self.submit(store, "hold", images=(base64.b64decode(b64png()),))
        queued = self.submit(store, "next")
        wait_store(self, store, running, "IN_PROGRESS")
        db = TaskDB(self.path)
        self.addCleanup(db.close)
        request = lambda t: db._exec("SELECT request FROM tasks WHERE id = ?", (t,))[0]["request"]
        self.assertIsNone(request(running))
        self.assertEqual(json.loads(request(queued))["prompt"], "next")
        registry.gate.set()
        wait_store(self, store, queued, "COMPLETED")
        self.assertIsNone(request(queued))

    def test_failed_result_write_still_finishes(self):
        store = self.store(FakeRegistry())
        write = store._db.set_finished

        def full_disk(task_id, now, image, steps_run, error, caption=None):
            if image is not None:
                raise sqlite3.OperationalError("database or disk is full")
            write(task_id, now, image, steps_run, error, caption)

        store._db.set_finished = full_disk
        t = wait_store(self, store, self.submit(store, "x"), "FAILED", "COMPLETED")
        self.assertEqual((t["status"], t["error"]), ("FAILED", "OperationalError: database or disk is full"))

    def test_sweep_finished_only(self):
        db = TaskDB(self.path)
        self.addCleanup(db.close)
        for i, task_id in enumerate(["old", "new", "running", "queued"]):
            db.insert(task_id, "qwen-base", 42, "{}", float(i))
        for task_id, finished in (("old", 10.0), ("new", 30.0)):
            db.set_started(task_id, 5.0)
            db.set_finished(task_id, finished, b"png", 1, None)
        db.set_started("running", 5.0)
        db.sweep(20.0)
        self.assertIsNone(db.get("old"))
        self.assertEqual([db.get(t)["status"] for t in ("new", "running", "queued")], ["COMPLETED", "IN_PROGRESS", "CREATED"])
        db.sweep(1e9)
        self.assertEqual(db.counts(), (1, 2))


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = os.path.join(self.tmp.name, "base")
        os.mkdir(self.base)

    def test_defaults(self):
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY})
        self.assertEqual((s.device, s.host, s.port, s.preload), ("auto", "127.0.0.1", 8740, ()))
        self.assertEqual(s.model_paths, {"qwen-base": self.base})
        self.assertEqual((s.max_queued, s.task_ttl), (64, 3600))
        self.assertEqual(s.db_path, os.path.abspath("data/imagegen.sqlite3"))
        self.assertEqual(load_settings({"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "DB_PATH": "~/x.db"}).db_path,
                         os.path.expanduser("~/x.db"))
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "MAX_QUEUED": "8", "TASK_TTL": "60"})
        self.assertEqual((s.max_queued, s.task_ttl), (8, 60))
        self.assertEqual(s.gpu_memory_fraction, 0)
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "GPU_MEMORY_FRACTION": "0.5"})
        self.assertEqual(s.gpu_memory_fraction, 0.5)

    def families(self, **env):
        return load_settings({"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, **env}).families

    def test_family_defaults(self):
        default = FamilySettings("bf16", "bf16", 10 * 1024**2)
        self.assertEqual(self.families(), {"qwen": default, "flux": default, "fibo": default})  # a family has settings before it has a path

    def test_shared_keys_cover_every_family(self):
        shared = FamilySettings("int8", "int8", 7340032)
        families = self.families(TRANSFORMER_PRECISION="int8", TEXT_ENCODER_PRECISION=" int8 ", MAX_REFERENCE_PIXELS="7340032")
        self.assertEqual(families, {"qwen": shared, "flux": shared, "fibo": shared})
        families = self.families(TRANSFORMER_PRECISION="fp8_dynamic", TEXT_ENCODER_PRECISION="fp8")
        self.assertEqual(families["flux"], FamilySettings("fp8_dynamic", "fp8", 10 * 1024**2))

    def test_scoped_key_overrides_one_family(self):
        for key, field, value, parsed in (("TRANSFORMER_PRECISION", "transformer_precision", "fp8", "fp8"),
                                          ("TEXT_ENCODER_PRECISION", "text_encoder_precision", "fp8", "fp8"),
                                          ("MAX_REFERENCE_PIXELS", "max_reference_pixels", "7340032", 7340032)):
            for family, other in (("qwen", "flux"), ("flux", "qwen")):
                with self.subTest(key=key, family=family):
                    shared = "int8" if "PRECISION" in key else "1048576"
                    families = self.families(**{key: shared, f"{family.upper()}_{key}": f" {value} "})
                    self.assertEqual(getattr(families[family], field), parsed)
                    self.assertEqual(str(getattr(families[other], field)), shared)

    def test_blank_scoped_key_falls_back(self):
        for blank in ("", "   "):
            with self.subTest(blank=blank):
                families = self.families(TRANSFORMER_PRECISION="int8", QWEN_TRANSFORMER_PRECISION=blank,
                                         TEXT_ENCODER_PRECISION="fp8", QWEN_TEXT_ENCODER_PRECISION=blank,
                                         MAX_REFERENCE_PIXELS="7340032", QWEN_MAX_REFERENCE_PIXELS=blank,
                                         FLUX_TRANSFORMER_PRECISION=blank)
                self.assertEqual(families["qwen"], FamilySettings("int8", "fp8", 7340032))
                self.assertEqual(families["flux"], families["qwen"])
                self.assertEqual(self.families(QWEN_MAX_REFERENCE_PIXELS=blank)["qwen"].max_reference_pixels, 10 * 1024**2)

    def test_invalid_scoped_key_is_named(self):
        for env in ({"QWEN_TRANSFORMER_PRECISION": "int4"}, {"FLUX_TRANSFORMER_PRECISION": "int4"},
                    {"QWEN_TEXT_ENCODER_PRECISION": "fp8_dynamic"}, {"FLUX_TEXT_ENCODER_PRECISION": "fp8_dynamic"},
                    {"QWEN_MAX_REFERENCE_PIXELS": "0"}, {"FLUX_MAX_REFERENCE_PIXELS": "lots"}):
            with self.subTest(env=env), self.assertRaisesRegex(ValueError, f"^{next(iter(env))} must"):
                self.families(**env)

    def test_new_family_gets_settings(self):
        with mock.patch.dict(MODELS, {"new": SimpleNamespace(family="nova")}):
            families = self.families(NOVA_TRANSFORMER_PRECISION="int8", TEXT_ENCODER_PRECISION="fp8")
        self.assertEqual(families["nova"], FamilySettings("int8", "fp8", 10 * 1024**2))
        self.assertEqual(families["qwen"], FamilySettings("bf16", "fp8", 10 * 1024**2))

    def test_preload_and_qwen_turbo(self):
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "QWEN_IMAGE_21_TURBO_PATH": self.tmp.name, "PRELOAD": "qwen-turbo, qwen-base"})
        self.assertEqual(s.preload, ("qwen-turbo", "qwen-base"))
        self.assertEqual(set(s.model_paths), {"qwen-base", "qwen-turbo"})

    def test_flux_klein_only(self):
        s = load_settings({"FLUX_KLEIN_PATH": self.base, "IMAGEGEN_API_KEY": KEY})
        self.assertEqual((s.model_paths, s.upsampler_path), ({"flux-klein": self.base}, None))
        s = load_settings({"FLUX_KLEIN_PATH": self.base, "FLUX_KLEIN_UPSAMPLER_PATH": self.tmp.name, "IMAGEGEN_API_KEY": KEY, "PRELOAD": "flux-klein"})
        self.assertEqual((s.upsampler_path, s.preload), (self.tmp.name, ("flux-klein",)))

    def test_fibo_edit_turbo(self):
        s = load_settings({"FIBO_EDIT_TURBO_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "PRELOAD": "fibo-edit-turbo"})
        self.assertEqual((s.model_paths, s.preload), ({"fibo-edit-turbo": self.base}, ("fibo-edit-turbo",)))
        families = self.families(TRANSFORMER_PRECISION="fp8", FIBO_TRANSFORMER_PRECISION="int8", FIBO_MAX_REFERENCE_PIXELS="1048576")
        self.assertEqual(families["fibo"], FamilySettings("int8", "bf16", 1048576))
        self.assertEqual(families["qwen"].transformer_precision, "fp8")
        with self.assertRaisesRegex(ValueError, "FIBO_EDIT_TURBO_PATH"):
            load_settings({"FIBO_EDIT_TURBO_PATH": os.path.join(self.tmp.name, "missing"), "IMAGEGEN_API_KEY": KEY})

    def test_no_model(self):
        with self.assertRaisesRegex(ValueError, "no model configured"):
            load_settings({"IMAGEGEN_API_KEY": KEY})
        with self.assertRaisesRegex(ValueError, "no model configured"):  # the upsampler is not a model
            load_settings({"FLUX_KLEIN_UPSAMPLER_PATH": self.base, "IMAGEGEN_API_KEY": KEY})

    def test_old_klein_keys_ignored(self):
        with self.assertRaisesRegex(ValueError, "no model configured"):
            load_settings({"KLEIN_PATH": self.base, "IMAGEGEN_API_KEY": KEY})
        s = load_settings({"FLUX_KLEIN_PATH": self.base, "KLEIN_UPSAMPLER_PATH": self.base, "IMAGEGEN_API_KEY": KEY})
        self.assertIsNone(s.upsampler_path)
        families = self.families(KLEIN_TRANSFORMER_PRECISION="int4", KLEIN_TEXT_ENCODER_PRECISION="int4", KLEIN_MAX_REFERENCE_PIXELS="0")
        self.assertEqual(families["flux"], FamilySettings())

    def test_old_preload_names_refused(self):
        env = {"QWEN_IMAGE_21_PATH": self.base, "QWEN_IMAGE_21_TURBO_PATH": self.base, "FLUX_KLEIN_PATH": self.base, "IMAGEGEN_API_KEY": KEY}
        for old in ("base", "turbo", "klein"):
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, "PRELOAD"):
                load_settings({**env, "PRELOAD": old})

    def test_missing_upsampler_dir(self):
        env = {"FLUX_KLEIN_PATH": self.base, "FLUX_KLEIN_UPSAMPLER_PATH": os.path.join(self.tmp.name, "missing"), "IMAGEGEN_API_KEY": KEY}
        with self.assertRaisesRegex(ValueError, "FLUX_KLEIN_UPSAMPLER_PATH"):
            load_settings(env)

    def test_device(self):
        env = {"FLUX_KLEIN_PATH": self.base, "IMAGEGEN_API_KEY": KEY}
        self.assertEqual(load_settings({**env, "IMAGEGEN_DEVICE": " cuda "}).device, "cuda")
        with self.assertRaisesRegex(ValueError, "IMAGEGEN_DEVICE"):
            load_settings({**env, "IMAGEGEN_DEVICE": "tpu"})

    def test_preload_one_family(self):
        env = {"QWEN_IMAGE_21_PATH": self.base, "QWEN_IMAGE_21_TURBO_PATH": self.base, "FLUX_KLEIN_PATH": self.base, "IMAGEGEN_API_KEY": KEY}
        self.assertEqual(load_settings({**env, "PRELOAD": "qwen-base,qwen-turbo"}).preload, ("qwen-base", "qwen-turbo"))
        with self.assertRaisesRegex(ValueError, "one family"):
            load_settings({**env, "PRELOAD": "qwen-base,flux-klein"})

    def test_errors(self):
        for env in [
            {},
            {"QWEN_IMAGE_21_PATH": os.path.join(self.tmp.name, "missing"), "IMAGEGEN_API_KEY": KEY},
            {"QWEN_IMAGE_21_PATH": self.base},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": "   "},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": "short"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "PRELOAD": "qwen-turbo"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "PRELOAD": "huge"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "IMAGEGEN_DEVICE": "tpu"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "PORT": "http"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "MAX_QUEUED": "0"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "TASK_TTL": "1h"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "TRANSFORMER_PRECISION": "int4"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "TEXT_ENCODER_PRECISION": "fp16"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "TEXT_ENCODER_PRECISION": "fp8_dynamic"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "GPU_MEMORY_FRACTION": "half"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "GPU_MEMORY_FRACTION": "1.5"},
            {"QWEN_IMAGE_21_PATH": self.base, "IMAGEGEN_API_KEY": KEY, "MAX_REFERENCE_PIXELS": "0"},
        ]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                load_settings(env)


class TestLoadPipeline(unittest.TestCase):
    """load_pipeline's from_pretrained kwargs per precision, with from_pretrained stubbed: no weights."""

    def load(self, shared=None, **precision):
        import diffusers
        from imagegen.engine import qwen

        with mock.patch.object(diffusers.QwenImage21Pipeline, "from_pretrained") as fp:
            qwen.load_pipeline("/m", "mps", shared, **precision)
        fp.assert_called_once()
        return fp.call_args.kwargs

    def test_quant_mapping(self):
        import diffusers
        import torch
        import transformers
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int8WeightOnlyConfig,
        )

        kinds = {"transformer": diffusers.TorchAoConfig, "text_encoder": transformers.TorchAoConfig}
        quant_types = {
            "int8": Int8WeightOnlyConfig,
            "fp8": Float8WeightOnlyConfig,
            "fp8_dynamic": Float8DynamicActivationFloat8WeightConfig,
        }
        for t in ("bf16", *quant_types):
            for te in ("bf16", "int8", "fp8"):
                with self.subTest(transformer=t, text_encoder=te):
                    kw = self.load(transformer=t, text_encoder=te)
                    self.assertEqual((kw["dtype"], kw["device_map"]), (torch.bfloat16, "mps"))
                    want = {n: p for n, p in (("transformer", t), ("text_encoder", te)) if p != "bf16"}
                    if not want:
                        self.assertNotIn("quantization_config", kw)
                        continue
                    mapping = kw["quantization_config"].quant_mapping
                    self.assertEqual(set(mapping), set(want))
                    for name, cfg in mapping.items():
                        self.assertIs(type(cfg), kinds[name])
                        self.assertIsInstance(cfg.quant_type, quant_types[want[name]])

    def test_reference_side(self):
        from imagegen.engine.common import reference_side

        self.assertEqual(reference_side(0, 7 * 1024**2), 1024)
        self.assertEqual(reference_side(7, 7 * 1024**2), 1024)
        self.assertEqual(reference_side(10, 7 * 1024**2), 832)
        self.assertEqual(reference_side(10, 10 * 1024**2), 1024)

    def test_quant_config(self):
        import diffusers
        import transformers
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int8WeightOnlyConfig,
        )

        from imagegen.engine.common import quant_config

        for library in (diffusers, transformers):
            self.assertIsNone(quant_config(library, "bf16"))
            for name, quant_type in (("int8", Int8WeightOnlyConfig), ("fp8", Float8WeightOnlyConfig),
                                     ("fp8_dynamic", Float8DynamicActivationFloat8WeightConfig)):
                with self.subTest(library=library.__name__, precision=name):
                    config = quant_config(library, name)
                    self.assertIs(type(config), library.TorchAoConfig)
                    self.assertIsInstance(config.quant_type, quant_type)

    def test_shared_text_encoder_not_quantized(self):
        shared = {"text_encoder": object(), "vae": object(), "processor": object()}
        kw = self.load(shared, transformer="int8", text_encoder="int8")
        self.assertIs(kw["text_encoder"], shared["text_encoder"])
        self.assertEqual(set(kw["quantization_config"].quant_mapping), {"transformer"})
        kw = self.load(shared, transformer="bf16", text_encoder="int8")
        self.assertNotIn("quantization_config", kw)
        kw = self.load(shared, transformer="fp8", text_encoder="fp8")
        self.assertEqual(set(kw["quantization_config"].quant_mapping), {"transformer"})


class TestFluxEngine(unittest.TestCase):
    def test_canvas_line(self):
        from imagegen.engine import flux

        self.assertEqual(flux.canvas_line(1360, 768), "TARGET CANVAS: 85:48\n")
        self.assertEqual(flux.canvas_line(1024, 1024), "TARGET CANVAS: 1:1\n")

    def test_one_reference_untouched(self):
        from imagegen.engine.common import fit_references

        img = Image.new("RGB", (3000, 2000))
        self.assertIs(fit_references([img], 1024**2)[0], img)  # the pipeline fits a single one itself

    def test_references_under_budget_unchanged(self):
        from imagegen.engine.common import fit_references

        images = [Image.new("RGB", (2000, 2000)) for _ in range(3)]
        self.assertIs(fit_references(images, 10 * 1024**2), images)
        images = images[:2] * 5  # 10 at 1024x1024 fit in 10 MP exactly
        self.assertIs(fit_references(images, 10 * 1024**2), images)

    def test_references_shrink_to_budget(self):
        from imagegen.engine.common import fit_references

        out = fit_references([Image.new("RGB", (2000, 1000))] * 10, 7 * 1024**2)
        self.assertEqual(len(out), 10)
        for img in out:
            self.assertAlmostEqual(img.width * img.height, 832**2, delta=832**2 * 0.01)
            self.assertAlmostEqual(img.width / img.height, 2, delta=0.01)
        self.assertLessEqual(sum(img.width * img.height for img in out), 7 * 1024**2)

    def test_small_references_never_upscale(self):
        from imagegen.engine.common import fit_references

        small = Image.new("RGB", (100, 100))
        large = Image.new("RGB", (2000, 2000))
        out = fit_references([small, large] * 5, 7 * 1024**2)
        self.assertIs(out[0], small)
        self.assertEqual(out[1].size, (832, 832))

    def test_render_uses_the_caption(self):
        from imagegen.engine import flux

        image = Image.new("RGB", (8, 8))
        pipe = mock.MagicMock(return_value=[image])
        req = GenerateRequest(prompt="cat", model="flux-klein", ratio="16:9", steps=16, upsample=True)
        with mock.patch.object(flux, "upsample", return_value="a long caption") as upsample:
            result = flux.render(pipe, req, upsampler=("model", "processor"))
        upsample.assert_called_once_with(("model", "processor"), "cat", None, 1360, 768)
        self.assertEqual(result, (image, 16, "a long caption"))
        kw = pipe.call_args.kwargs
        self.assertEqual((kw["prompt"], kw["width"], kw["height"], kw["num_inference_steps"]), ("a long caption", 1360, 768, 16))
        pipe.transformer.register_forward_hook.return_value.remove.assert_called_once()


class TestFiboEngine(unittest.TestCase):
    def render(self, on_step=None, budget=10 * 1024**2, **fields):
        from imagegen.engine import fibo

        result = Image.new("RGB", (8, 8))
        pipe = mock.MagicMock(num_timesteps=4)
        pipe.return_value.images = [result]
        req = GenerateRequest(**{"prompt": "make it red", "model": "fibo-edit-turbo", "steps": 4, "seed": 7, **fields})
        with mock.patch("builtins.print"):
            def run(**kw):
                for i in range(4):
                    kw["callback_on_step_end"](pipe, i, None, {})
                return pipe.return_value

            pipe.side_effect = run
            out = fibo.render(pipe, req, on_step, budget)
        self.assertEqual(out, (result, 4, None))
        return pipe.call_args.kwargs

    def test_render(self):
        import inspect

        import diffusers

        images = (Image.new("RGB", (1024, 1024)), Image.new("RGB", (512, 512)))
        steps = []
        kw = self.render(lambda done, total: steps.append((done, total)), images=images)
        self.assertEqual(kw["prompt"], "make it red")
        self.assertEqual(kw["image"], list(images))
        self.assertEqual((kw["guidance_scale"], kw["num_inference_steps"]), (1.0, 4))
        self.assertEqual((kw["width"], kw["height"]), (None, None))  # the pipeline sizes from the first image
        self.assertEqual(kw["generator"].initial_seed(), 7)
        self.assertEqual(steps, [(1, 4), (2, 4), (3, 4), (4, 4)])
        self.assertLessEqual(set(kw), set(inspect.signature(diffusers.BriaFibo2Pipeline.__call__).parameters))

    def test_size(self):
        images = (Image.new("RGB", (16, 16)),)
        kw = self.render(images=images, ratio="16:9")
        self.assertEqual((kw["width"], kw["height"]), (1360, 768))
        kw = self.render(images=images, ratio="16:9", width=512)
        self.assertEqual((kw["width"], kw["height"]), (512, 768))
        kw = self.render(images=images, width=512)
        self.assertEqual((kw["width"], kw["height"]), (512, 1024))  # as the pipeline would for a missing side

    def test_size_it_cannot_render_whole_shrinks_to_1_mp(self):
        images = (Image.new("RGB", (16, 16)),)
        for size, expected in (
            ({"width": 1536, "height": 1024}, (1536, 1024)),  # the limit itself renders as asked
            ({"width": 2304, "height": 1824}, (1152, 912)),
            ({"width": 2752, "height": 2752}, (1024, 1024)),
            ({"width": 2752, "height": 576}, (2240, 464)),  # the narrowest size past the limit
            ({"width": 2304}, (1536, 688)),  # the pipeline would make the height 1024
            ({"height": 2304}, (688, 1536)),
            ({"width": 2304, "ratio": "4:3"}, (1680, 624)),  # 2304 x 864
        ):
            kw = self.render(images=images, **size)
            self.assertEqual((kw["width"], kw["height"]), expected)
            self.assertEqual((kw["width"] % 16, kw["height"] % 16), (0, 0))

    def test_reference_budget(self):
        images = (Image.new("RGB", (2000, 2000)),) * 4
        self.assertEqual({img.size for img in self.render(images=images, budget=1024**2)["image"]}, {(512, 512)})

    def test_load_quantizes_transformer_and_text_encoder(self):
        import diffusers
        import torch
        import transformers

        from imagegen.engine import fibo

        with mock.patch.object(diffusers.BriaFibo2Pipeline, "from_pretrained") as fp:
            fibo.load("/m", "cuda", [], "int8", "fp8")
            fibo.load("/m", "cuda", [], "bf16", "bf16")
        quantized, plain = fp.call_args_list
        mapping = quantized.kwargs["quantization_config"].quant_mapping
        self.assertIs(type(mapping["transformer"]), diffusers.TorchAoConfig)
        self.assertIs(type(mapping["text_encoder"]), transformers.TorchAoConfig)
        self.assertEqual(plain.args, ("/m",))
        self.assertEqual(plain.kwargs, {"dtype": torch.bfloat16, "device_map": "cuda"})


class TestModelRegistry(unittest.TestCase):
    """The real registry with stub engines patched in: no weights."""

    PATHS = {"qwen-base": "/base", "qwen-turbo": "/turbo", "flux-klein": "/klein", "fibo-edit-turbo": "/fibo"}
    FAMILIES = {"qwen": FamilySettings("int8", "bf16", 123), "flux": FamilySettings("fp8", "fp8", 456),
                "fibo": FamilySettings("bf16", "int8", 789)}

    def setUp(self):
        from imagegen.engine import registry

        self.loads = []  # (path, loaded)
        self.precisions = []  # (path, transformer, text_encoder)
        self.broken = set()  # paths whose load raises
        self.renders = []  # (pipe, model, max_reference_pixels, extra kwargs)

        def load(path, device, loaded, transformer, text_encoder):
            self.loads.append((path, loaded))
            self.precisions.append((path, transformer, text_encoder))
            if path in self.broken:
                raise RuntimeError("no weights")
            return f"pipe {path}"

        def render(pipe, req, on_step, max_reference_pixels, **extra):
            self.renders.append((pipe, req.model, max_reference_pixels, extra))
            return "image", req.steps, "caption" if req.upsample else None

        stub = SimpleNamespace(load=load, render=render)
        for patcher in (mock.patch.dict(registry.ENGINES, {"qwen": stub, "flux": stub, "fibo": stub}),
                        mock.patch.object(registry, "print", create=True)):
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = mock.patch.object(registry.flux, "load_upsampler", return_value=("model", "processor"))
        self.load_upsampler = patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = registry.ModelRegistry(self.PATHS, "cpu", self.FAMILIES, upsampler_path="/upsampler")

    def loaded(self):
        return {name for name, s in self.registry.status().items() if s["loaded"]}

    def upsample_flux_klein(self):
        return self.registry.render(GenerateRequest(prompt="x", model="flux-klein", steps=16, upsample=True))

    def test_configured_and_status(self):
        from imagegen.engine.registry import ModelRegistry

        registry = ModelRegistry({"flux-klein": "/klein", "qwen-base": "/base"}, "cpu", self.FAMILIES)
        self.assertEqual(registry.configured, ("qwen-base", "flux-klein"))  # MODELS order, whatever the dict's
        self.assertFalse(registry.can_upsample)
        self.assertEqual(registry.upsampler_status(), {"configured": False, "loaded": False})
        no_flux_klein = ModelRegistry({"qwen-turbo": "/turbo"}, "cpu", self.FAMILIES, upsampler_path="/upsampler")
        self.assertFalse(no_flux_klein.can_upsample)  # nothing could use it
        self.assertEqual(registry.status(), {
            "qwen-base": {"configured": True, "loaded": False},
            "qwen-turbo": {"configured": False, "loaded": False},
            "flux-klein": {"configured": True, "loaded": False},
            "fibo-edit-turbo": {"configured": False, "loaded": False},
        })
        self.assertEqual(self.registry.upsampler_status(), {"configured": True, "loaded": False})

    def test_qwen_family_stays_loaded_together(self):
        self.registry.get("qwen-base")
        self.registry.get("qwen-turbo")
        self.assertEqual(self.loads, [("/base", []), ("/turbo", ["pipe /base"])])
        self.assertEqual(self.loaded(), {"qwen-base", "qwen-turbo"})
        self.assertEqual(self.registry.get("qwen-base"), "pipe /base")
        self.assertEqual(len(self.loads), 2)  # already resident

    def test_flux_klein_evicts_qwen(self):
        self.registry.get("qwen-base")
        self.registry.get("qwen-turbo")
        self.registry.get("flux-klein")
        self.assertEqual(self.loads[-1], ("/klein", []))
        self.assertEqual(self.loaded(), {"flux-klein"})
        self.assertEqual({m["id"]: m["loaded"] for m in model_views(self.registry)}, {"qwen-base": False, "qwen-turbo": False, "flux-klein": True, "fibo-edit-turbo": False})

    def test_three_families_evict_each_other(self):
        self.upsample_flux_klein()
        self.registry.get("fibo-edit-turbo")
        self.assertEqual(self.loads[-1], ("/fibo", []))
        self.assertEqual(self.loaded(), {"fibo-edit-turbo"})
        self.assertEqual(self.registry.upsampler_status(), {"configured": True, "loaded": False})
        self.registry.get("qwen-base")
        self.registry.get("qwen-turbo")
        self.assertEqual(self.loaded(), {"qwen-base", "qwen-turbo"})
        self.registry.get("fibo-edit-turbo")
        self.assertEqual(self.loads[-1], ("/fibo", []))
        self.assertEqual(self.loaded(), {"fibo-edit-turbo"})
        self.registry.get("flux-klein")
        self.assertEqual(self.loaded(), {"flux-klein"})
        self.assertEqual(self.precisions[1], ("/fibo", "bf16", "int8"))

    def test_engines_cover_every_family(self):
        from imagegen.engine import fibo, registry

        self.assertIs(registry.fibo, fibo)
        self.assertEqual({m.family for m in MODELS.values()}, set(registry.ENGINES))

    def test_qwen_evicts_flux_klein_and_upsampler(self):
        self.upsample_flux_klein()
        self.upsample_flux_klein()
        self.assertEqual(self.registry.upsampler_status(), {"configured": True, "loaded": True})
        self.assertEqual(self.load_upsampler.call_count, 1)
        self.load_upsampler.assert_called_once_with("/upsampler", "cpu")
        self.registry.get("qwen-turbo")
        self.assertEqual(self.loads[-1], ("/turbo", []))
        self.assertEqual(self.loaded(), {"qwen-turbo"})
        self.assertEqual(self.registry.upsampler_status(), {"configured": True, "loaded": False})
        self.upsample_flux_klein()  # and they come back on the next ask
        self.assertEqual(self.load_upsampler.call_count, 2)
        self.assertEqual(self.loaded(), {"flux-klein"})

    def test_failed_load_leaves_registry_usable(self):
        self.registry.get("qwen-base")
        self.broken = {"/turbo"}
        with self.assertRaises(RuntimeError):
            self.registry.get("qwen-turbo")
        self.assertEqual(self.loaded(), {"qwen-base"})  # same family: nothing was evicted
        self.broken = {"/klein"}
        with self.assertRaises(RuntimeError):
            self.registry.get("flux-klein")
        self.assertEqual(self.loaded(), set())  # freed before the load, so it had the memory
        self.broken = set()
        self.assertEqual(self.registry.get("flux-klein"), "pipe /klein")
        self.assertEqual(self.loaded(), {"flux-klein"})

    def test_render_passes_upsampler_only_when_asked(self):
        self.assertEqual(self.registry.render(GenerateRequest(prompt="x", model="qwen-base", steps=7)), ("image", 7, None))
        self.registry.render(GenerateRequest(prompt="x", model="flux-klein", steps=16))
        self.assertEqual(self.upsample_flux_klein(), ("image", 16, "caption"))
        self.assertEqual(self.renders, [
            ("pipe /base", "qwen-base", 123, {}),
            ("pipe /klein", "flux-klein", 456, {}),
            ("pipe /klein", "flux-klein", 456, {"upsampler": ("model", "processor")}),
        ])

    def test_each_load_gets_its_familys_precision(self):
        for name in ("qwen-base", "qwen-turbo", "flux-klein", "qwen-base", "flux-klein"):  # the last two load again after a swap
            self.registry.get(name)
        self.assertEqual(self.precisions, [
            ("/base", "int8", "bf16"),
            ("/turbo", "int8", "bf16"),
            ("/klein", "fp8", "fp8"),
            ("/base", "int8", "bf16"),
            ("/klein", "fp8", "fp8"),
        ])

    def test_render_after_swap_keeps_its_familys_reference_budget(self):
        for model, steps in (("flux-klein", 16), ("qwen-turbo", 7), ("flux-klein", 16)):
            self.registry.render(GenerateRequest(prompt="x", model=model, steps=steps))
        self.assertEqual([r[2] for r in self.renders], [456, 123, 456])

    def test_reports_per_family(self):
        qwen = {"transformer": "int8", "text_encoder": "bf16"}
        flux = {"transformer": "fp8", "text_encoder": "fp8"}
        fibo = {"transformer": "bf16", "text_encoder": "int8"}
        self.assertEqual(self.registry.precision, {"qwen": qwen, "flux": flux, "fibo": fibo})
        views = {m["id"]: (m["precision"], m["max_reference_pixels"]) for m in model_views(self.registry)}
        self.assertEqual(views, {"qwen-base": (qwen, 123), "qwen-turbo": (qwen, 123), "flux-klein": (flux, 456), "fibo-edit-turbo": (fibo, 789)})


class TestSchemas(unittest.TestCase):
    def test_defaults(self):
        req = parse_request({"prompt": "hi"}, MODELS)
        self.assertEqual(req, GenerateRequest(prompt="hi"))
        self.assertEqual((req.model, req.steps, req.seed, req.transparent, req.images), ("qwen-turbo", 40, 42, False, ()))
        self.assertEqual(parse_request({"prompt": "hi"}, ("qwen-base",)).model, "qwen-base")
        self.assertEqual(parse_request({"prompt": "hi"}, ("flux-klein",)).model, "flux-klein")
        self.assertEqual(parse_request({"prompt": "hi"}, ("qwen-base", "flux-klein")).model, "qwen-base")

    def test_sides_per_model(self):
        for side in (64, 80, 1040, 4096):
            with self.subTest(side=side):
                req = parse_request({"prompt": "hi", "model": "flux-klein", "width": side, "height": side}, MODELS)
                self.assertEqual((req.width, req.height), (side, side))
        for side in (1040, 4096):
            with self.subTest(side=side), self.assertRaises(ValidationError):
                parse_request({"prompt": "hi", "model": "qwen-turbo", "width": side}, MODELS)

    def test_steps_per_model(self):
        klein = {"prompt": "hi", "model": "flux-klein"}
        self.assertEqual(parse_request(klein, MODELS).steps, 16)
        self.assertEqual(parse_request({**klein, "steps": 40}, MODELS).steps, 16)
        self.assertEqual(parse_request({**klein, "steps": 8}, MODELS).steps, 8)
        self.assertEqual(parse_request({"prompt": "hi", "model": "qwen-turbo"}, MODELS).steps, 40)
        self.assertEqual(parse_request({"prompt": "hi", "model": "qwen-base", "steps": 100}, MODELS).steps, 100)

    def test_input_pixels_per_model(self):
        over_qwen = b64png((4000, 4000))  # 16 MP
        self.assertEqual(len(parse_request({"prompt": "hi", "model": "flux-klein", "images": [over_qwen]}, MODELS).images), 1)
        with self.assertRaises(ValidationError):
            parse_request({"prompt": "hi", "model": "qwen-turbo", "images": [over_qwen]}, MODELS)

    def test_fibo_edit_turbo(self):
        edit = {"prompt": "hi", "model": "fibo-edit-turbo", "images": [b64png()]}
        req = parse_request(edit, MODELS)
        self.assertEqual((req.steps, req.seed, len(req.images)), (4, 42, 1))
        self.assertEqual(parse_request({**edit, "steps": 40}, MODELS).steps, 4)
        self.assertEqual(len(parse_request({**edit, "images": [b64png()] * 4}, MODELS).images), 4)
        self.assertEqual(parse_request({**edit, "width": 2752, "height": 64}, MODELS).width, 2752)
        self.assertEqual(MODELS["fibo-edit-turbo"].ratios, MODELS["flux-klein"].ratios)
        self.assertEqual(parse_request(json.loads(json.dumps(dump_request(req))), MODELS), req)
        with self.assertRaisesRegex(ValidationError, "edits images"):  # the default model when it is the only one
            parse_request({"prompt": "hi"}, ("fibo-edit-turbo",))
        self.assertEqual(len(parse_request({"prompt": "hi", "images": [b64png()] * 10}, MODELS).images), 10)  # others keep 10

    def test_options(self):
        klein = parse_request({"prompt": "hi", "model": "flux-klein", "upsample": True}, MODELS)
        self.assertEqual((klein.upsample, klein.transparent), (True, False))
        self.assertEqual(parse_request({"prompt": "hi", "transparent": True}, MODELS).transparent, True)
        self.assertEqual(parse_request({"prompt": "hi", "upsample": False, "transparent": False}, MODELS).upsample, False)
        with self.assertRaises(ValidationError):
            parse_request({"prompt": "hi", "model": "flux-klein", "upsample": True}, MODELS, can_upsample=False)

    def test_full(self):
        req = parse_request(
            {"prompt": "hi", "model": "qwen-turbo", "ratio": "16:9", "width": 512, "height": 2752, "steps": 1,
             "seed": 2**63 - 1, "transparent": True, "images": [b64png()]},
            MODELS,
        )
        self.assertEqual((req.width, req.height, req.seed, len(req.images)), (512, 2752, 2**63 - 1, 1))
        self.assertIsInstance(req.images[0], bytes)
        self.assertEqual(load_images(req).images[0].size, (4, 4))

    def test_dump_round_trips(self):
        req = parse_request({"prompt": "hi", "model": "qwen-turbo", "ratio": "4:3", "images": [b64png()]}, MODELS)
        self.assertEqual(parse_request(json.loads(json.dumps(dump_request(req))), MODELS), req)
        req = parse_request({"prompt": "hi", "model": "flux-klein", "width": 4096, "upsample": True}, MODELS)
        self.assertEqual(parse_request(json.loads(json.dumps(dump_request(req))), MODELS), req)

    def test_seed_range(self):
        for seed in (-1, 2**63, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValidationError):
                parse_request({"prompt": "hi", "seed": seed}, MODELS)


if __name__ == "__main__":
    unittest.main()
