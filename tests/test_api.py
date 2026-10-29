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
import unittest
from unittest import mock
import urllib.error
import urllib.request

from PIL import Image

from imagegen.api import make_server, png_render
from imagegen.config import load_settings
from imagegen.jobs import TaskStore
from imagegen.schemas import MODELS, GenerateRequest, ValidationError, dump_request, load_images, parse_request
from imagegen.store import TaskDB


class FakeRegistry:
    device = "cpu"
    precision = {"transformer": "int8", "text_encoder": "bf16"}

    def __init__(self, configured=("base", "turbo")):
        self.configured = configured
        self.log = []  # (event, prompt)
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        self.gate = threading.Event()  # "hold" renders wait on it after step 1

    def status(self):
        return {n: {"configured": n in self.configured, "loaded": False} for n in MODELS}

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
            return Image.new("RGB", (8, 8), "red"), 8 if req.model == "turbo" else req.steps
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
    configured = ("base", "turbo")
    max_queued = 64

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db_path = os.path.join(tmp.name, "data", "tasks.sqlite3")
        self.registry = FakeRegistry(self.configured)
        self.now = [1000.0]
        self.tasks = TaskStore(png_render(self.registry), self.db_path, self.max_queued, ttl=60, clock=lambda: self.now[0])
        self.server = make_server("127.0.0.1", 0, self.tasks, self.registry)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.05,), daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.registry.gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.tasks.shutdown()

    def request(self, method, path, body=None, raw=None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.base + path, data=data, method=method)
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


class TestApi(ServerCase):
    def assert400(self, body, raw=None):
        status, headers, data = self.request("POST", "/v1/tasks", body, raw)
        self.assertEqual(status, 400, data)
        self.assertIn("error", json.loads(data))
        self.assertEqual(headers["Content-Type"], "application/json")
        return json.loads(data)["error"]

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

    def test_content_length(self):
        for headers, expected in [({}, 411), ({"Content-Length": "-5"}, 400), ({"Content-Length": "x"}, 400),
                                  ({"Content-Length": str(64 * 1024 * 1024 + 1)}, 413)]:
            with self.subTest(headers=headers):
                conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
                conn.putrequest("POST", "/v1/tasks")
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
        self.assertEqual((done["model"], done["seed"], done["position"], done["error"]), ("turbo", 7, None, None))
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
            self.assertEqual((headers["X-Model"], headers["X-Steps"], headers["X-Seed"]), ("turbo", "8", "7"))
            self.assertIn("X-Elapsed-Ms", headers)
            self.assertEqual(Image.open(io.BytesIO(data)).size, (8, 8))

    def test_turbo(self):
        task_id = self.submit({"prompt": "cat", "model": "turbo", "steps": 30})["task_id"]
        self.wait(task_id, "COMPLETED")
        status, headers, _ = self.request("GET", f"/v1/tasks/{task_id}/image")
        self.assertEqual(status, 200)
        self.assertEqual((headers["X-Model"], headers["X-Steps"]), ("turbo", "8"))

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
                "precision": {"transformer": "int8", "text_encoder": "bf16"},
                "models": {"base": {"configured": True, "loaded": False}, "turbo": {"configured": True, "loaded": False}},
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
                                       ("POST", "/v1/tasks/x", 405), ("POST", "/health", 405)]:
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


class TestBaseOnly(ServerCase):
    configured = ("base",)

    def test_turbo_not_configured(self):
        status, _, data = self.request("POST", "/v1/tasks", {"prompt": "x", "model": "turbo"})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"], "model 'turbo' is not configured")


class TestPreload(unittest.TestCase):
    def test_runs_first_and_is_not_a_task(self):
        order = []
        store = TaskStore(lambda data, on_step: order.append(data["prompt"]) or (b"png", 1), ":memory:")
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
        return store.submit(dump_request(GenerateRequest(prompt=prompt, **{"model": "base", **fields})))["task_id"]

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
        db.insert("t1", "turbo", 42, json.dumps(dump_request(GenerateRequest(prompt="x", model="turbo"))), 1.0)
        db.insert("t2", "base", 42, json.dumps(dump_request(GenerateRequest(prompt="y", model="base", images=(base64.b64decode(b64png()),)))), 2.0)
        db.close()
        registry = FakeRegistry(configured=("base",))
        new = self.store(registry)
        new.recover()
        failed = wait_store(self, new, "t1", "FAILED", "COMPLETED")
        self.assertEqual((failed["status"], failed["error"]), ("FAILED", "ValidationError: model 'turbo' is not configured"))
        wait_store(self, new, "t2", "COMPLETED")  # its stored image round-tripped through base64

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

        def full_disk(task_id, now, image, steps_run, error):
            if image is not None:
                raise sqlite3.OperationalError("database or disk is full")
            write(task_id, now, image, steps_run, error)

        store._db.set_finished = full_disk
        t = wait_store(self, store, self.submit(store, "x"), "FAILED", "COMPLETED")
        self.assertEqual((t["status"], t["error"]), ("FAILED", "OperationalError: database or disk is full"))

    def test_sweep_finished_only(self):
        db = TaskDB(self.path)
        self.addCleanup(db.close)
        for i, task_id in enumerate(["old", "new", "running", "queued"]):
            db.insert(task_id, "base", 42, "{}", float(i))
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
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base})
        self.assertEqual((s.device, s.host, s.port, s.preload), ("auto", "127.0.0.1", 8740, ()))
        self.assertEqual(s.model_paths, {"base": self.base})
        self.assertEqual((s.max_queued, s.task_ttl), (64, 3600))
        self.assertEqual(s.db_path, os.path.abspath("data/imagegen.sqlite3"))
        self.assertEqual(load_settings({"QWEN_IMAGE_21_PATH": self.base, "DB_PATH": "~/x.db"}).db_path,
                         os.path.expanduser("~/x.db"))
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "MAX_QUEUED": "8", "TASK_TTL": "60"})
        self.assertEqual((s.max_queued, s.task_ttl), (8, 60))
        self.assertEqual((s.transformer_precision, s.text_encoder_precision), ("bf16", "bf16"))
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "TRANSFORMER_PRECISION": "int8", "TEXT_ENCODER_PRECISION": " int8 "})
        self.assertEqual((s.transformer_precision, s.text_encoder_precision), ("int8", "int8"))

    def test_preload_and_turbo(self):
        s = load_settings({"QWEN_IMAGE_21_PATH": self.base, "QWEN_IMAGE_21_TURBO_PATH": self.tmp.name, "PRELOAD": "turbo, base"})
        self.assertEqual(s.preload, ("turbo", "base"))
        self.assertEqual(set(s.model_paths), {"base", "turbo"})

    def test_errors(self):
        for env in [
            {},
            {"QWEN_IMAGE_21_PATH": os.path.join(self.tmp.name, "missing")},
            {"QWEN_IMAGE_21_PATH": self.base, "PRELOAD": "turbo"},
            {"QWEN_IMAGE_21_PATH": self.base, "PRELOAD": "huge"},
            {"QWEN_IMAGE_21_PATH": self.base, "QWEN_IMAGE_21_DEVICE": "tpu"},
            {"QWEN_IMAGE_21_PATH": self.base, "PORT": "http"},
            {"QWEN_IMAGE_21_PATH": self.base, "MAX_QUEUED": "0"},
            {"QWEN_IMAGE_21_PATH": self.base, "TASK_TTL": "1h"},
            {"QWEN_IMAGE_21_PATH": self.base, "TRANSFORMER_PRECISION": "int4"},
            {"QWEN_IMAGE_21_PATH": self.base, "TEXT_ENCODER_PRECISION": "fp16"},
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
        from torchao.quantization import Int8WeightOnlyConfig

        kinds = {"transformer": diffusers.TorchAoConfig, "text_encoder": transformers.TorchAoConfig}
        for t in ("bf16", "int8"):
            for te in ("bf16", "int8"):
                with self.subTest(transformer=t, text_encoder=te):
                    kw = self.load(transformer=t, text_encoder=te)
                    self.assertEqual((kw["dtype"], kw["device_map"]), (torch.bfloat16, "mps"))
                    want = {n for n, p in (("transformer", t), ("text_encoder", te)) if p == "int8"}
                    if not want:
                        self.assertNotIn("quantization_config", kw)
                        continue
                    mapping = kw["quantization_config"].quant_mapping
                    self.assertEqual(set(mapping), want)
                    for name, cfg in mapping.items():
                        self.assertIs(type(cfg), kinds[name])
                        self.assertIsInstance(cfg.quant_type, Int8WeightOnlyConfig)

    def test_shared_text_encoder_not_quantized(self):
        shared = {"text_encoder": object(), "vae": object(), "processor": object()}
        kw = self.load(shared, transformer="int8", text_encoder="int8")
        self.assertIs(kw["text_encoder"], shared["text_encoder"])
        self.assertEqual(set(kw["quantization_config"].quant_mapping), {"transformer"})
        kw = self.load(shared, transformer="bf16", text_encoder="int8")
        self.assertNotIn("quantization_config", kw)


class TestSchemas(unittest.TestCase):
    def test_defaults(self):
        req = parse_request({"prompt": "hi"}, MODELS)
        self.assertEqual(req, GenerateRequest(prompt="hi"))
        self.assertEqual((req.model, req.steps, req.seed, req.transparent, req.images), ("turbo", 40, 42, False, ()))
        self.assertEqual(parse_request({"prompt": "hi"}, ("base",)).model, "base")

    def test_full(self):
        req = parse_request(
            {"prompt": "hi", "model": "turbo", "ratio": "16:9", "width": 512, "height": 2752, "steps": 1,
             "seed": 2**63 - 1, "transparent": True, "images": [b64png()]},
            MODELS,
        )
        self.assertEqual((req.width, req.height, req.seed, len(req.images)), (512, 2752, 2**63 - 1, 1))
        self.assertIsInstance(req.images[0], bytes)
        self.assertEqual(load_images(req).images[0].size, (4, 4))

    def test_dump_round_trips(self):
        req = parse_request({"prompt": "hi", "model": "turbo", "ratio": "4:3", "images": [b64png()]}, MODELS)
        self.assertEqual(parse_request(json.loads(json.dumps(dump_request(req))), MODELS), req)

    def test_seed_range(self):
        for seed in (-1, 2**63, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValidationError):
                parse_request({"prompt": "hi", "seed": seed}, MODELS)


if __name__ == "__main__":
    unittest.main()
