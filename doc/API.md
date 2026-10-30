# imagegen API

How to integrate a client with the imagegen server: submit a render task, poll it, and download the PNG.

The server renders Qwen-Image-2.1 images on one GPU. It works on **one task at a time**, first in, first out. Each
model loads on the first request that needs it and stays in memory until the server stops. Tasks are stored in
SQLite, so waiting tasks and finished images survive a restart.

- Base URL: `http://127.0.0.1:8740` by default (`HOST` and `PORT` change it).
- There is no authentication, so keep the server on a private network.
- Every JSON response has a `Content-Type: application/json` header. Errors always look like `{"error": "<message>"}`.

## Flow

```
POST /v1/tasks              -> 202 {"data": {"task_id", "status": "CREATED", "position"}}
GET  /v1/tasks/{task_id}    -> 200 {"data": {..., "status": "CREATED" | "IN_PROGRESS" | "COMPLETED" | "FAILED"}}
                               poll until COMPLETED or FAILED
GET  /v1/tasks/{task_id}/image -> 200 image/png
```

Status moves `CREATED → IN_PROGRESS → COMPLETED` or `→ FAILED`. Nothing else is possible.

## Endpoints

### POST /v1/tasks

This submits a render. The body is a JSON object, at most 64 MB, and must be sent with a `Content-Length`. The server
validates the body before queueing it, so a `202` means the task was accepted.

| Field | Type | Default | Rules |
| --- | --- | --- | --- |
| `prompt` | string | required | Must not be empty. Long prompts (several thousand characters) are fine. |
| `model` | string | `"turbo"` | `"turbo"` or `"base"`. It must be configured on the server (see `/health`). When the server has no Turbo model, the default is `"base"`. |
| `ratio` | string | see [Size](#size) | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16` |
| `width` | integer | see [Size](#size) | 256–2752, a multiple of 32 |
| `height` | integer | see [Size](#size) | 256–2752, a multiple of 32 |
| `steps` | integer | `40` | 1–100. Turbo ignores it and always runs 8 steps. |
| `seed` | integer | `42` | 0 to 2^63−1. The same request with the same seed gives the same image. |
| `transparent` | boolean | `false` | Adds instructions to the prompt asking for a transparent background (RGBA alpha channel). |
| `images` | array of strings | `[]` | Up to 10 base64-encoded images (PNG, JPEG, WebP…). Any number of them turns the task into an edit or reference-guided render. Each must be at most 15,147,008 pixels. |

Unknown fields are rejected. A boolean is not accepted where an integer is expected.

The response is `202`:

```json
{"data": {"task_id": "47c558c69037443bb66d388f7d655b73", "status": "CREATED", "position": 0}}
```

`position` is the number of tasks waiting ahead of this one. `0` means it runs next, or starts right away if nothing is
rendering.

| Status | When | Client action |
| --- | --- | --- |
| 400 | Invalid JSON, a malformed `Content-Length`, or a field that breaks a rule. `error` says which. | Don't retry; fix the request. |
| 411 | `Content-Length` header missing. | Send it (HTTP clients do this by default). |
| 413 | Body over 64 MB. | Send fewer or smaller reference images. |
| 429 | The queue is full (`MAX_QUEUED` tasks waiting, default 64). Comes with a `Retry-After: 30` header. | Wait `Retry-After` seconds, then resubmit. |
| 503 | The server is shutting down. | Retry after a short wait. |

### GET /v1/tasks/{task_id}

This polls a task. A `200` response looks like this:

```json
{
  "data": {
    "task_id": "47c558c69037443bb66d388f7d655b73",
    "status": "IN_PROGRESS",
    "model": "turbo",
    "seed": 3,
    "position": null,
    "step": 5,
    "steps": 8,
    "error": null,
    "created_at": 1793306072.41,
    "started_at": 1793306072.43,
    "finished_at": null,
    "elapsed_ms": null
  }
}
```

| Field | Meaning |
| --- | --- |
| `position` | The number of tasks ahead of this one while `CREATED`, otherwise `null`. |
| `step`, `steps` | Denoising progress while `IN_PROGRESS`. Both are `null` until the first step finishes, which includes the time spent loading the model. |
| `error` | Why the task failed, set only when `FAILED`. Example: `"RuntimeError: MPS backend out of memory"`. |
| `created_at`, `started_at`, `finished_at` | Unix timestamps in seconds, or `null`. |
| `elapsed_ms` | Time from start to finish, once finished. It doesn't include time spent waiting in the queue, but it does include the model load when this task was the model's first use. |

A `404` (`{"error": "unknown task"}`) means the id never existed or has expired. Finished tasks are deleted
`TASK_TTL` seconds after they finish (default 3600).

### GET /v1/tasks/{task_id}/image

This downloads the result of a `COMPLETED` task. The `200` body is the PNG itself (`Content-Type: image/png`), with
these headers:

| Header | Value |
| --- | --- |
| `X-Model` | `base` or `turbo` |
| `X-Seed` | the seed used |
| `X-Steps` | the number of denoising steps actually run (always 8 for Turbo) |
| `X-Elapsed-Ms` | the render time (same as `elapsed_ms`) |

Downloading doesn't delete anything, so a retry is safe until the task expires. Other responses:

| Status | When |
| --- | --- |
| 409 | The task is not `COMPLETED` yet. `error` says `task is CREATED` or `task is IN_PROGRESS`. If the task failed, `error` holds the failure message. |
| 404 | Unknown or expired task. |

Output images are PNG, often RGBA even when `transparent` is false. Convert them if your storage needs RGB or JPEG.

### GET /health

This shows which models are configured and loaded, and how busy the queue is.

```json
{
  "device": "mps",
  "precision": {"transformer": "bf16", "text_encoder": "bf16"},
  "models": {"base": {"configured": true, "loaded": true}, "turbo": {"configured": true, "loaded": false}},
  "queued": 1,
  "busy": true,
  "tasks": 7
}
```

`queued` is the number of tasks waiting, `busy` is whether the worker is rendering or loading a `PRELOAD` model, and `tasks` is the number of tasks
stored, including finished ones. Use `models.<name>.configured` to decide which models to offer. `precision` is the
weight precision the server loads the transformer and text encoder in, `bf16`, `int8`, `fp8` or `fp8_dynamic`
(transformer only), the same for every model.

## Size

The output size comes from these fields, in this order:

1. `ratio` picks one of these presets, all about 4 MP:

   | ratio | width × height |
   | --- | --- |
   | 1:1 | 2048 × 2048 |
   | 4:3 | 2400 × 1792 |
   | 3:4 | 1792 × 2400 |
   | 3:2 | 2528 × 1696 |
   | 2:3 | 1696 × 2528 |
   | 16:9 | 2752 × 1536 |
   | 9:16 | 1536 × 2752 |

2. Without `ratio`, a request with no `images` uses `1:1` (2048 × 2048). A request with `images` keeps the aspect
   of the **last** input image at about **1 MP** (for example 1376 × 768 for a 16:9 input), not the ~4 MP presets.
3. `width` and `height` each replace only their own side. The other side still comes from step 1 or 2, so sending
   one alone usually changes the aspect. Send **both** to get an exact size.

For any other aspect ratio or size, compute it yourself and send both `width` and `height`. Round each to a
multiple of 32, and keep each between 256 and 2752. For example, 4:5 at about 1 MP is `896 × 1120`, and 21:9 at
about 2 MP is `2176 × 928`. The largest output is 2752 on a side, so there's no 4K output. Upscale on the client if
you need more.

## Timing

These were measured on an Apple M-series Mac with 64 GB of memory:

| Render | Time |
| --- | --- |
| First request for a model | Adds about 10–15 s to load it (set `PRELOAD=base,turbo` to load at startup instead) |
| Turbo, 1024 × 1024 | ~25 s |
| Turbo, 1344 × 768 | ~28 s |
| Turbo, 2752 × 1536 | ~2 min |
| Base, 768 × 768, 12 steps | ~18 s |
| Base, ~2K, 40 steps | several minutes |

These times are for `bf16`. With `int8` (see `precision` in `/health`), the transformer takes about half the memory
and the text encoder about 57% (13 GB less with both, measured with Turbo), renders are about 15% slower, and images
change slightly for the same seed (fine texture only). `fp8` and `fp8_dynamic` should take the same memory as `int8`,
but no full render has been measured with them yet.

A task's total wait is its own render plus every render queued ahead of it. With many tasks in flight that can be
tens of minutes, so set client deadlines with that in mind (see below).

## Restarts

- Tasks still `CREATED` when the server stops stay queued, and they run in their original order after it starts again.
- A task that was `IN_PROGRESS` when the server stopped becomes `FAILED` with `"server restarted during render"`.
  Resubmit it if you still need it.
- Finished tasks stay downloadable until `TASK_TTL` runs out.
- While the server is down, requests get connection errors. Polling should treat those as temporary and retry.

## Integrating a client

1. **Submit once.** If a `POST /v1/tasks` dies mid-flight (a read timeout after the body was sent), the task may have
   been created anyway. Resending it can queue a duplicate. Retry a POST only when the connection was never made.
2. **Poll with backoff.** Start at 1 s and grow to about 10 s. A poll costs the server almost nothing.
3. **Use two deadlines.**
   - A queue deadline while `CREATED`. `position` tells you how far back the task is.
   - A render deadline that starts at `started_at` (`IN_PROGRESS`). About 10 minutes covers base at full size.

   Don't use one deadline counted from submission: queue time can be arbitrarily long.
4. **Retry polls and downloads on temporary failures:** connection errors, `503`, and `5xx`. Both calls are
   read-only.
5. **On `FAILED`**, raise with `error` as the message.
6. **Download right away** after `COMPLETED`, and store the bytes on your side. The server only keeps results for
   `TASK_TTL` seconds.
7. **Reference images:** send at most 10, as base64, ordered by importance. If you send more, the server returns
   400; it doesn't drop the extras itself.

### Python example (async, httpx)

```python
import asyncio
import base64
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8740"
MAX_REFERENCES = 10
QUEUE_TIMEOUT = 3600.0   # s waiting as CREATED
RENDER_TIMEOUT = 900.0   # s from started_at to finished
TRANSIENT = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


class ImageGenError(RuntimeError):
    pass


def size_for(aspect_ratio: str, megapixels: float) -> tuple[int, int]:
    """Width and height for "W:H" at about `megapixels`, each a multiple of 32 within 256..2752."""
    w, h = (int(x) for x in aspect_ratio.split(":"))
    scale = (megapixels * 1_000_000 / (w * h)) ** 0.5
    fit = lambda v: max(256, min(2752, round(v * scale / 32) * 32))
    return fit(w), fit(h)


def _error(resp: httpx.Response) -> str:
    try:
        return resp.json().get("error") or resp.text
    except ValueError:
        return resp.text


async def _get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """GET with retries on connection errors and 5xx; both endpoints it is used on are read-only."""
    for attempt in range(6):
        try:
            resp = await client.get(url)
            if resp.status_code < 500:
                return resp
        except TRANSIENT:
            pass
        await asyncio.sleep(min(2.0**attempt, 30.0))
    raise ImageGenError(f"GET {url} kept failing")


async def generate(
    prompt: str,
    references: list[Path] = (),
    aspect_ratio: str = "1:1",
    megapixels: float = 1.0,
    *,
    model: str = "turbo",
    seed: int = 42,
) -> bytes:
    """Render one image and return its PNG bytes."""
    width, height = size_for(aspect_ratio, megapixels)
    body = {
        "prompt": prompt,
        "model": model,
        "width": width,
        "height": height,
        "seed": seed,
        "images": [base64.b64encode(p.read_bytes()).decode() for p in references[:MAX_REFERENCES]],
    }
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        while True:
            resp = await client.post("/v1/tasks", json=body)
            if resp.status_code != 429:
                break
            await asyncio.sleep(float(resp.headers.get("Retry-After", 30)))
        if resp.status_code != 202:
            raise ImageGenError(f"submit -> {resp.status_code}: {_error(resp)}")
        task_id = resp.json()["data"]["task_id"]

        submitted, delay = time.monotonic(), 1.0
        while True:
            resp = await _get(client, f"/v1/tasks/{task_id}")
            if resp.status_code != 200:
                raise ImageGenError(f"task {task_id} -> {resp.status_code}: {_error(resp)}")
            task = resp.json()["data"]
            if task["status"] == "COMPLETED":
                break
            if task["status"] == "FAILED":
                raise ImageGenError(f"task {task_id} failed: {task['error']}")
            if task["status"] == "CREATED" and time.monotonic() - submitted > QUEUE_TIMEOUT:
                raise ImageGenError(f"task {task_id} still queued at position {task['position']}")
            if task["status"] == "IN_PROGRESS" and time.time() - task["started_at"] > RENDER_TIMEOUT:
                raise ImageGenError(f"task {task_id} rendering for over {RENDER_TIMEOUT:.0f} s")
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

        resp = await _get(client, f"/v1/tasks/{task_id}/image")
        if resp.status_code != 200:
            raise ImageGenError(f"image {task_id} -> {resp.status_code}: {_error(resp)}")
        return resp.content
```

### curl

```bash
TASK=$(curl -s localhost:8740/v1/tasks \
  -d '{"prompt":"a ceramic mug on a wooden table, product photo","model":"turbo","width":1024,"height":1024}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["task_id"])')

until curl -s localhost:8740/v1/tasks/$TASK | grep -q '"status": "\(COMPLETED\|FAILED\)"'; do sleep 3; done

curl -s localhost:8740/v1/tasks/$TASK/image -o mug.png
```

When a request has reference images, write the body to a file rather than putting the base64 on the command line,
because shell argument limits are about 1 MB:

```bash
python3 -c 'import base64,json,sys; print(json.dumps({"prompt": sys.argv[1], "model": "turbo",
  "images": [base64.b64encode(open(p, "rb").read()).decode() for p in sys.argv[2:]]}))' \
  "the same mug in a forest" photo.png > req.json
curl -s localhost:8740/v1/tasks --data-binary @req.json
```
