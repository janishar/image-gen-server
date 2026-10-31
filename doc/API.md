# imagegen API

How to integrate a client with the imagegen server: submit a render task, poll it, and download the PNG.

The server renders Qwen-Image-2.1 (`qwen-base`, `qwen-turbo`) and FLUX 3 Image Klein (`flux-klein`) images, and edits
images with Bria Fibo 2.0 Edit Turbo (`fibo-edit-turbo`), on one GPU.
It works on **one task at a time**, first in, first out. Each model loads on the first request that needs it and stays
in memory until a model of another family is asked for (see [Memory](#memory)). Tasks are stored in SQLite, so waiting
tasks and finished images survive a restart.

- Base URL: `http://127.0.0.1:8740` by default (`HOST` and `PORT` change it).
- Every request needs `Authorization: Bearer <IMAGEGEN_API_KEY>`, except `GET /health`. A missing or wrong key gets a
  `401` with `{"error": "missing or invalid API key"}`. The server won't start without a key of 16+ characters.
- Every JSON response has a `Content-Type: application/json` header. Errors always look like `{"error": "<message>"}`.

## Flow

```
POST /v1/tasks              -> 202 {"data": {"task_id", "status": "CREATED", "position"}}
     (or POST /v1/models/{model}/tasks)
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
| `prompt` | string | required | Must not be empty. Long prompts (several thousand characters) are fine, except on Fibo, where text and images share 4,096 tokens. On Fibo, with several images, it is a string holding a JSON object; see [Fibo](#fibo). |
| `model` | string | `"qwen-turbo"` | `"qwen-base"`, `"qwen-turbo"`, `"flux-klein"` or `"fibo-edit-turbo"`. It must be configured on the server (see [`GET /v1/models`](#get-v1models)). When the server has no `"qwen-turbo"`, the default is the first configured model. |
| `ratio` | string | see [Size](#size) | `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16` |
| `width` | integer | see [Size](#size) | Qwen: 256–2752, a multiple of 32. Klein: 64–4096, a multiple of 16. Fibo: 64–2752, a multiple of 16. |
| `height` | integer | see [Size](#size) | Same as `width`. |
| `steps` | integer | `40`, Klein `16`, Fibo `4` | 1–100. Turbo ignores it and always runs 8 steps. Klein runs at most 16, the number the checkpoint is distilled for, so a longer request is cut to 16. Fibo likewise runs at most 4. |
| `seed` | integer | `42` | 0 to 2^63−1. The same request with the same seed gives the same image. |
| `transparent` | boolean | `false` | Qwen only. Adds instructions to the prompt asking for a transparent background (RGBA alpha channel). |
| `upsample` | boolean | `false` | Klein only. `true` expands `prompt` into a full scene caption with the prompt upsampler, using the references when editing. Otherwise `prompt` goes to the model as sent, so it should already be a full caption. `true` is a 400 when the server has no upsampler (see `/health`). |
| `images` | array of strings | `[]` | Up to 10 base64-encoded images (PNG, JPEG, WebP…), or up to 4 on Fibo. Any number of them turns the task into an edit or reference-guided render. Fibo only edits, so it needs at least 1: a request with none is a `400`. Each must be at most 15,147,008 pixels on Qwen and Fibo, 33,554,432 on Klein. |

Unknown fields are rejected. A boolean is not accepted where an integer is expected. An option a model does not take
is a `400` when set to `true`, for example `"transparent": true` on Klein; `false` is accepted on every model, so a
client can send the same body to all of them.

The response is `202`:

```json
{"data": {"task_id": "47c558c69037443bb66d388f7d655b73", "status": "CREATED", "position": 0}}
```

`position` is the number of tasks waiting ahead of this one. `0` means it runs next, or starts right away if nothing is
rendering.

| Status | When | Client action |
| --- | --- | --- |
| 400 | Invalid JSON, a malformed `Content-Length`, a field that breaks a rule, or a model that isn't configured. `error` says which. | Don't retry; fix the request. |
| 411 | `Content-Length` header missing. | Send it (HTTP clients do this by default). |
| 413 | Body over 64 MB. | Send fewer or smaller reference images. |
| 429 | The queue is full (`MAX_QUEUED` tasks waiting, default 64). Comes with a `Retry-After: 30` header. | Wait `Retry-After` seconds, then resubmit. |
| 503 | The server is shutting down. | Retry after a short wait. |

### POST /v1/models/{model}/tasks

This is `POST /v1/tasks` with the model taken from the URL. The body, responses and rules are the same, and the URL
wins over a `model` in the body. A model that is unknown or not configured gets a `404`
(`{"error": "model 'x' is not configured"}`), where `POST /v1/tasks` answers `400`.

### GET /v1/models

This lists what each configured model accepts, as `{"data": [...]}` in the order `qwen-base`, `qwen-turbo`,
`flux-klein`, `fibo-edit-turbo`. Unconfigured models are left out. It needs the key.

```json
{
  "data": [
    {
      "id": "flux-klein",
      "loaded": false,
      "precision": {"transformer": "bf16", "text_encoder": "bf16"},
      "max_input_pixels": 33554432,
      "max_reference_pixels": 10485760,
      "family": "flux",
      "ratios": {"1:1": [1024, 1024], "4:3": [1152, 864], "...": "..."},
      "min_side": 64,
      "max_side": 4096,
      "side_multiple": 16,
      "steps": 16,
      "max_steps": 16,
      "options": ["upsample"],
      "max_images": 10,
      "needs_image": false
    }
  ]
}
```

| Field | Meaning |
| --- | --- |
| `id` | The name to send as `model`. |
| `loaded` | Whether the model is in memory now. |
| `precision` | The weight precision of the transformer and text encoder, the same for every model of a family. See [`GET /health`](#get-health). |
| `max_images` | The most reference images a request can send: 10, or 4 on `fibo-edit-turbo`. |
| `needs_image` | `true` when a request must send at least one image, as `fibo-edit-turbo` must. |
| `max_input_pixels` | The largest reference image, in pixels. |
| `max_reference_pixels` | The total reference pixels of a request, the same for every model of a family. Past it every reference shrinks evenly. For `flux-klein` the cap only applies with several references: a single reference keeps the pipeline's own 4 MP limit. `fibo-edit-turbo` scales references back up, so the cap saves it no memory. See [Size](#size). |
| `family` | `qwen`, `flux` or `fibo`. One family stays loaded at a time (see [Memory](#memory)). |
| `ratios` | Each `ratio` and its `[width, height]`. |
| `min_side`, `max_side`, `side_multiple` | The rules for `width` and `height`. |
| `steps`, `max_steps` | The default when a request sends none, and the most that run. Turbo reports the Qwen values, `40` and `100`, but always runs 8. |
| `options` | The options the model takes. Klein lists `upsample` only when the server has an upsampler. |

### GET /v1/tasks/{task_id}

This polls a task. A `200` response looks like this:

```json
{
  "data": {
    "task_id": "47c558c69037443bb66d388f7d655b73",
    "status": "IN_PROGRESS",
    "model": "qwen-turbo",
    "seed": 3,
    "position": null,
    "step": 5,
    "steps": 8,
    "error": null,
    "caption": null,
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
| `step`, `steps` | Denoising progress while `IN_PROGRESS`. Both are `null` until the first step finishes, which includes the time spent loading the model (see [Memory](#memory)) and upsampling the prompt. |
| `error` | Why the task failed, set only when `FAILED`. Example: `"RuntimeError: MPS backend out of memory"`. |
| `caption` | The full caption the prompt upsampler wrote, once `COMPLETED` with `"upsample": true`, otherwise `null`. |
| `created_at`, `started_at`, `finished_at` | Unix timestamps in seconds, or `null`. |
| `elapsed_ms` | Time from start to finish, once finished. It doesn't include time spent waiting in the queue, but it does include the model load when this task had to load its model. |

A `404` (`{"error": "unknown task"}`) means the id never existed or has expired. Finished tasks are deleted
`TASK_TTL` seconds after they finish (default 3600).

### GET /v1/tasks/{task_id}/image

This downloads the result of a `COMPLETED` task. The `200` body is the PNG itself (`Content-Type: image/png`), with
the headers below. Add `?format=jpeg` for a compressed JPEG at the same pixel size (`Content-Type: image/jpeg`, quality
92), for example to send the image to a vision LLM. Transparency is flattened onto white, so keep the PNG if you need
the alpha channel. Each JPEG is encoded when you ask for it. Any other `format` gets a `400`.

| Header | Value |
| --- | --- |
| `X-Model` | `qwen-base`, `qwen-turbo`, `flux-klein` or `fibo-edit-turbo` |
| `X-Seed` | the seed used |
| `X-Steps` | the number of denoising steps actually run (always 8 for Turbo, at most 16 for Klein, at most 4 for Fibo) |
| `X-Elapsed-Ms` | the render time (same as `elapsed_ms`) |

Downloading doesn't delete anything, so a retry is safe until the task expires. Other responses:

| Status | When |
| --- | --- |
| 409 | The task is not `COMPLETED` yet. `error` says `task is CREATED` or `task is IN_PROGRESS`. If the task failed, `error` holds the failure message. |
| 404 | Unknown or expired task. |

Qwen output images are often RGBA even when `transparent` is false. Convert them if your storage needs RGB or JPEG.

### GET /health

This shows which models and upsampler are configured and loaded, and how busy the queue is.

```json
{
  "device": "mps",
  "precision": {
    "qwen": {"transformer": "int8", "text_encoder": "int8"},
    "flux": {"transformer": "bf16", "text_encoder": "bf16"},
    "fibo": {"transformer": "bf16", "text_encoder": "bf16"}
  },
  "models": {
    "qwen-base": {"configured": true, "loaded": true},
    "qwen-turbo": {"configured": true, "loaded": false},
    "flux-klein": {"configured": false, "loaded": false},
    "fibo-edit-turbo": {"configured": false, "loaded": false}
  },
  "upsampler": {"configured": false, "loaded": false},
  "queued": 1,
  "busy": true,
  "tasks": 7
}
```

`queued` is the number of tasks waiting, `busy` is whether the worker is rendering or loading a `PRELOAD` model, and `tasks` is the number of tasks
stored, including finished ones. `models` lists every model the server knows, so use `models.<name>.configured` or
`GET /v1/models` to decide which to offer. Use `upsampler.configured` to tell whether the server can upsample.
`precision` is the weight precision the server loads the transformer and text encoder in, by family: `bf16`, `int8`,
`fp8` or `fp8_dynamic` (transformer only). It lists every family the server knows, configured or not. It is per family
and not per model because `qwen-base` and `qwen-turbo` share one text encoder. `device` is one value for the whole
server.

## Size

The output size comes from these fields, in this order:

1. `ratio` picks one of these presets, about 4 MP for Qwen and about 1 MP for Klein and Fibo:

   | ratio | Qwen width × height | Klein and Fibo width × height |
   | --- | --- | --- |
   | 1:1 | 2048 × 2048 | 1024 × 1024 |
   | 4:3 | 2400 × 1792 | 1152 × 864 |
   | 3:4 | 1792 × 2400 | 864 × 1152 |
   | 3:2 | 2528 × 1696 | 1248 × 832 |
   | 2:3 | 1696 × 2528 | 832 × 1248 |
   | 16:9 | 2752 × 1536 | 1360 × 768 |
   | 9:16 | 1536 × 2752 | 768 × 1360 |

2. Without `ratio`, a request with no `images` uses `1:1`. A request with `images` depends on the family:
   - Qwen keeps the aspect of the **last** input image at about **1 MP** (for example 1376 × 768 for a 16:9 input), not
     the ~4 MP presets.
   - Klein keeps the size of the **first** input image, after it is fitted under 4 MP for one reference or 1 MP for
     several.
   - Fibo keeps the aspect of the **first** input image at about **1 MP**, the size it was trained at.
3. `width` and `height` each replace only their own side. The other side still comes from step 1 or 2, so sending
   one alone usually changes the aspect. On Fibo with no `ratio`, the other side is 1024 instead. Send **both** to get
   an exact size.

For any other aspect ratio or size, compute it yourself and send both `width` and `height`. On Qwen, round each to a
multiple of 32, and keep each between 256 and 2752. For example, 4:5 at about 1 MP is `896 × 1120`, and 21:9 at
about 2 MP is `2176 × 928`. The largest Qwen output is 2752 on a side, so there's no 4K output. Upscale on the client
if you need more. On Klein, round each to a multiple of 16, and keep each between 64 and 4096. On Fibo, round each to a
multiple of 16, and keep each between 64 and 2752; it was trained at about 1 MP. Larger sizes take much more memory and
time.

The server caps the total reference pixels of a request per family, with `MAX_REFERENCE_PIXELS` for every family or
`QWEN_MAX_REFERENCE_PIXELS`, `FLUX_MAX_REFERENCE_PIXELS` and `FIBO_MAX_REFERENCE_PIXELS` for one. `GET /v1/models` reports the cap as
`max_reference_pixels`. For `flux-klein` the cap only applies with several references: a single reference keeps the
pipeline's own 4 MP limit. When the cap leaves each reference less than 1 MP, a request with no size gets a smaller
output too: on Qwen with any number of references, on `flux-klein` with several.

Fibo prepares references itself: one reference is scaled to cover the output size and center-cropped to it, so an output
of another aspect loses the edges, and several are each cropped to about 1 MP at their own aspect. The cap applies with
several references, but they are then scaled back up, so a tight cap loses detail and saves no memory.

## Fibo

`fibo-edit-turbo` only edits, with at least 1 and at most 4 `images`. Its repo is gated behind Bria's access form, and
it is under CC BY-NC 4.0, so it is non-commercial: commercial use needs an agreement with Bria.

It was trained at about 1 MP. The server does not cap its size, but all 15 renders at about 4 MP drew two people in one
frame, so check larger outputs.

With one image, send a short edit instruction as `prompt`. With several, the instruction refers to them as `<image_1>`,
`<image_2>` and so on, in the order of `images`:

1. Put the scene first, as `<image_1>`, then the people, then the products.
2. Send `prompt` as a string holding a JSON object with two keys: `short_description`, a caption stating one single
   subject, its pose and the scene, and `edit_instruction`, a short instruction.

```json
{
  "short_description": "One single woman standing full-length, {facing}, wearing {outfit}, in this exact room: {background_notes} Exactly one person in the image.",
  "edit_instruction": "Add one woman to <image_1>, keeping the room exactly as it is, with every wall, the floor and the chair in place. She is the woman from <image_2>, which shows the same woman from six angles, so draw her only once, {facing}. Dress her in {outfit}, exactly as shown in <image_3> and <image_4>, replacing her black clothes. Full length, head to toe. Soft diffused daylight, real cloth with a visible weave, natural unretouched skin."
}
```

Measured at 1152 × 928, with a six-pose sheet as the model reference:

| Request | Result |
| --- | --- |
| A plain long instruction, people reference first | Several people |
| Scene first, the caption and short instruction above | One person, correct facing and the real room in 6 of 6 renders |
| Scene first, the same caption, a long instruction as `edit_instruction` | 5 of 6, and a floating outfit left in the image |
| A caption with no room | The model invented a different room |

So a sheet of several poses works as a reference when the caption and instruction say to draw one subject.

The checkpoint runs with guidance off, so there is no negative prompt. A `prompt` that is itself a JSON object must carry
an `edit_instruction` key: without one it passes validation, the pipeline refuses it and the task ends `FAILED`.

Text and images share 4,096 tokens, and 4 references can take up to about 2,300 of them. A longer prompt loses its end
with no error, so keep what must survive near the start, especially in non-Latin scripts, which take more tokens.

## Memory

One family stays loaded at a time: `qwen-base` and `qwen-turbo` together, `flux-klein` with its upsampler, or
`fibo-edit-turbo`. A request for another family unloads the current one and loads its own model first. The upsampler
loads on the first `"upsample": true` request after that, and leaves with `flux-klein`.

That load is part of the task. The task is `IN_PROGRESS` from `started_at`, and its `step` stays `null` until the load,
and any upsampling, is done, so longer than usual. The queue is strict first in, first out, so tasks alternating
between families pay a full load each time. Send a family's tasks together when you can. `loaded` in `GET /v1/models`
and `/health` shows what is in memory now.

On an Apple M-series Mac in `int8`, a swap took about 30 s when the weights had to come from disk and about 10 s when
they were still in the OS file cache. A swap has not been timed on a CUDA GPU.

## Timing

These were measured on an Apple M-series Mac (MPS) with 64 GB of memory. No GPU (CUDA) timing has been measured for
Klein. `fibo-edit-turbo` takes about 21 GB idle and peaks at about 25 GB at 1152 × 928 with 4 references, rendering in about
11 s on an RTX PRO 6000. At about 4 MP it drew two people in all 15 renders, and the server does not refuse that size.

| Render | Time |
| --- | --- |
| First request for a Qwen model | Adds about 10–15 s to load it (set `PRELOAD=qwen-base,qwen-turbo` to load at startup instead) |
| Turbo, 1024 × 1024 | ~25 s |
| Turbo, 1344 × 768 | ~28 s |
| Turbo, 2752 × 1536 | ~2 min |
| Base, 768 × 768, 12 steps | ~18 s |
| Base, ~2K, 40 steps | several minutes |
| First request for Klein | Adds a few seconds to load the model (set `PRELOAD=flux-klein` to load at startup instead) |
| Klein, text-to-image, 1024 × 1024, 16 steps | ~33 s |
| Klein, edit of a 1024 × 1024 reference, 16 steps | ~75 s |
| Klein, `"upsample": true` | Adds ~25 s for text-to-image, ~7 s for a one-image edit, on top of loading the upsampler |

These times are for `bf16`. A request that swaps families pays the load again, after the unload. With `int8` (see
`precision` in `/health`), the transformer takes about half the memory
and the text encoder about 57% (13 GB less with both, measured with Turbo), renders are about 15% slower, and images
change slightly for the same seed (fine texture only). `fp8` and `fp8_dynamic` should take the same memory as `int8`,
but no full render has been measured with them yet.

A task's total wait is its own render plus every render queued ahead of it. With many tasks in flight that can be
tens of minutes, so set client deadlines with that in mind (see below).

## Restarts

- Tasks still `CREATED` when the server stops stay queued, and they run in their original order after it starts again.
- A task that was `IN_PROGRESS` when the server stopped becomes `FAILED` with `"server restarted during render"`.
  Resubmit it if you still need it.
- A queued task whose model or upsampler is no longer configured becomes `FAILED` with its validation error when it
  comes up.
- Finished tasks stay downloadable until `TASK_TTL` runs out.
- While the server is down, requests get connection errors. Polling should treat those as temporary and retry.

## Integrating a client

1. **Submit once.** If a submit dies mid-flight (a read timeout after the body was sent), the task may have
   been created anyway. Resending it can queue a duplicate. Retry a POST only when the connection was never made.
2. **Poll with backoff.** Start at 1 s and grow to about 10 s. A poll costs the server almost nothing.
3. **Use two deadlines.**
   - A queue deadline while `CREATED`. `position` tells you how far back the task is.
   - A render deadline that starts at `started_at` (`IN_PROGRESS`). About 10 minutes covers `qwen-base` at full size.
     Allow for a full model load on top when the task may swap families.

   Don't use one deadline counted from submission: queue time can be arbitrarily long.

   If you also keep a stall deadline (no change in `step` for some time), don't apply it while `step` is `null`: a
   family swap, a first load or `"upsample": true` can hold it there far longer than any one step takes. Start the stall
   deadline at the first `step`, and give the time before it a limit of its own that covers a full load.
4. **Retry polls and downloads on temporary failures:** connection errors, `503`, and `5xx`. Both calls are
   read-only.
5. **On `FAILED`**, raise with `error` as the message.
6. **Download right away** after `COMPLETED`, and store the bytes on your side. The server only keeps results for
   `TASK_TTL` seconds.
7. **Reference images:** send at most the model's `max_images`, as base64, ordered by importance. If you send more,
   the server returns 400; it doesn't drop the extras itself.

### Python example (async, httpx)

```python
import asyncio
import base64
import os
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8740"
MAX_REFERENCES = 10
QUEUE_TIMEOUT = 3600.0   # s waiting as CREATED
RENDER_TIMEOUT = 900.0   # s from started_at to finished
HEADERS = {"Authorization": f"Bearer {os.environ['IMAGEGEN_API_KEY']}"}
TRANSIENT = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)


class ImageGenError(RuntimeError):
    pass


def size_for(aspect_ratio: str, megapixels: float) -> tuple[int, int]:
    """Width and height for "W:H" at about `megapixels`, each a multiple of 32 within 256..2752: fine for every model."""
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
    model: str = "qwen-turbo",
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
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS, timeout=httpx.Timeout(60.0, connect=10.0)) as client:
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
  -H "Authorization: Bearer $IMAGEGEN_API_KEY" \
  -d '{"prompt":"a ceramic mug on a wooden table, product photo","model":"qwen-turbo","width":1024,"height":1024}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"]["task_id"])')

until curl -s localhost:8740/v1/tasks/$TASK -H "Authorization: Bearer $IMAGEGEN_API_KEY" | grep -q '"status": "\(COMPLETED\|FAILED\)"'; do sleep 3; done

curl -s localhost:8740/v1/tasks/$TASK/image -H "Authorization: Bearer $IMAGEGEN_API_KEY" -o mug.png
```

When a request has reference images, write the body to a file rather than putting the base64 on the command line,
because shell argument limits are about 1 MB:

```bash
python3 -c 'import base64,json,sys; print(json.dumps({"prompt": sys.argv[1], "model": "qwen-turbo",
  "images": [base64.b64encode(open(p, "rb").read()).decode() for p in sys.argv[2:]]}))' \
  "the same mug in a forest" photo.png > req.json
curl -s localhost:8740/v1/tasks -H "Authorization: Bearer $IMAGEGEN_API_KEY" --data-binary @req.json
```
