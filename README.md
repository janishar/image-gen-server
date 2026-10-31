# image-gen-server

A small on-demand HTTP server for several image models behind one API: Qwen-Image-2.1 (`qwen-base`),
Qwen-Image-2.1-Turbo (`qwen-turbo`) and FLUX 3 Image Klein (`flux-klein`, the distilled 16-step checkpoint, not Klein
Base).
Jobs run one at a time in a FIFO queue on a single worker thread. A model loads the first time it is used and stays in
memory until a model of another family is asked for (see [Memory](#memory)). Clients submit a task, poll it and
download the PNG.

## Models

| Model | Family | `width`/`height` | `ratio` presets | `steps` | Options |
|---|---|---|---|---|---|
| `qwen-base` | Qwen | 256..2752, multiples of 32 | about 4 MP | default 40, max 100 | `transparent` |
| `qwen-turbo` | Qwen | 256..2752, multiples of 32 | about 4 MP | taken, but always runs its own 8-step schedule | `transparent` |
| `flux-klein` | Flux | 64..4096, multiples of 16 | about 1 MP | default 16, max 16 (a longer request is cut to 16) | `upsample`, with `FLUX_KLEIN_UPSAMPLER_PATH` |

Every model takes the ratios `1:1`, `4:3`, `3:4`, `3:2`, `2:3`, `16:9`, `9:16`, but the same ratio means a different
size per family: `16:9` is 2752×1536 on Qwen and 1360×768 on Klein. `GET /v1/models` lists all of this for the models
a server has configured.

Every model takes up to 10 reference `images`. With references and no size in the request, the output follows a
different reference per family: Qwen keeps the aspect of the **last** image at about 1 MP, Klein keeps the size of the
**first** image, fitted under 4 MP for one reference or 1 MP for several.

A request with no `model` gets `qwen-turbo` when configured, otherwise the first configured model.

## Setup

```sh
uv sync
cp .env.example .env   # then edit the paths and set the key
uv run --env-file .env image-gen-server
```

Klein's repo is gated: accept the license on [the gated repo](https://huggingface.co/black-forest-labs/flux-3-image-klein)
first, and log in with `hf auth login`. Klein takes about 20 GB of disk, and the optional upsampler about 19 GB more
(not gated).

```sh
hf download black-forest-labs/flux-3-image-klein --local-dir ~/models/flux-3-image-klein \
  --include "transformer/*" --include "text_encoder/*" --include "vae/*" --include "LICENSE.md" --include "NOTICE.md"
hf download black-forest-labs/flux-3-image-klein-pup --local-dir ~/models/flux-3-image-klein-pup   # optional
```

At least one of `QWEN_IMAGE_21_PATH`, `QWEN_IMAGE_21_TURBO_PATH` and `FLUX_KLEIN_PATH` is required. A model with no path
is off, and a path that is set must exist.

| Variable | Default | Meaning |
|---|---|---|
| `IMAGEGEN_API_KEY` | required | 16+ characters. Every route but `GET /health` needs `Authorization: Bearer <key>` |
| `QWEN_IMAGE_21_PATH` | unset | Qwen-Image-2.1 model directory (`qwen-base` off if unset) |
| `QWEN_IMAGE_21_TURBO_PATH` | unset | Qwen-Image-2.1-Turbo model directory (`qwen-turbo` off if unset) |
| `FLUX_KLEIN_PATH` | unset | FLUX 3 Image Klein model directory (`flux-klein` off if unset) |
| `FLUX_KLEIN_UPSAMPLER_PATH` | unset | Prompt upsampler directory, for requests that send `"upsample": true`. Unset, those are a 400 |
| `IMAGEGEN_DEVICE` | `auto` | `auto`, `mps`, `cuda` or `cpu` (auto: MPS, else CUDA, else CPU). Replaces `QWEN_IMAGE_21_DEVICE` and `KLEIN_DEVICE`, which are no longer read |
| `TRANSFORMER_PRECISION` | `bf16` | The default for every family. `bf16`, `int8`, `fp8` or `fp8_dynamic`: transformer weights, quantized while loading (no converted copy on disk). `fp8_dynamic` also quantizes activations and uses the FP8 matmul; it needs a CUDA GPU with FP8 support |
| `TEXT_ENCODER_PRECISION` | `bf16` | The default for every family. `bf16`, `int8` or `fp8`: text encoder weights, same as above (the VAE and the upsampler always stay `bf16`) |
| `GPU_MEMORY_FRACTION` | `0` (no cap) | One value for the whole server, CUDA only: the share of the card this server may hold, so an overrun fails one render instead of a process beside it. CUDA also tiles the Qwen VAE decode |
| `MAX_REFERENCE_PIXELS` | `10485760` | The default for every family: total reference pixels per request; past it every reference shrinks evenly. Each reference is about 1 MP inside a Qwen model. For Klein it only matters with several references, and only when the budget is tighter than 1 MP each. With no size in the request, shrunk references shrink the output too |
| `QWEN_TRANSFORMER_PRECISION`, `FLUX_TRANSFORMER_PRECISION` | `TRANSFORMER_PRECISION` | That family's transformer. Same values as the shared key |
| `QWEN_TEXT_ENCODER_PRECISION`, `FLUX_TEXT_ENCODER_PRECISION` | `TEXT_ENCODER_PRECISION` | That family's text encoder. Same values as the shared key, so no `fp8_dynamic` here either |
| `QWEN_MAX_REFERENCE_PIXELS`, `FLUX_MAX_REFERENCE_PIXELS` | `MAX_REFERENCE_PIXELS` | That family's reference budget |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8740` | Port |
| `PRELOAD` | empty | Comma list of models to load at startup, all of one family: `qwen-turbo`, `qwen-base,qwen-turbo` or `flux-klein`. Mixing families refuses to start. The upsampler is not preloaded |
| `MAX_QUEUED` | `64` | Waiting (`CREATED`) tasks allowed before a submit answers 429 |
| `TASK_TTL` | `3600` | Seconds a finished task and its PNG are kept before polls return 404 |
| `DB_PATH` | `data/imagegen.sqlite3` | SQLite file holding tasks and PNGs (relative to the working directory; created if missing) |

`TRANSFORMER_PRECISION`, `TEXT_ENCODER_PRECISION` and `MAX_REFERENCE_PIXELS` are the default for every family. A
`{FAMILY}_` key wins over the shared key for that family, and an empty one counts as unset. To run Klein's text encoder
in `bf16` while Qwen's stays `int8`, set `QWEN_TEXT_ENCODER_PRECISION=int8` and `FLUX_TEXT_ENCODER_PRECISION=bf16`.

These settings are per family and not per model because `qwen-base` and `qwen-turbo` share one text encoder.
`GPU_MEMORY_FRACTION` and `IMAGEGEN_DEVICE` stay shared: every family runs on the same device, under the same cap.

## API

Renders take 30 s to 10 min, so the API is asynchronous: submit, poll, download. Tasks, their requests and PNGs are
kept in the SQLite file `DB_PATH`, so they survive a restart (see below). [doc/API.md](doc/API.md) has the details.

- `POST /v1/tasks` takes JSON and answers `202 {"data": {"task_id", "status": "CREATED", "position"}}`
  (`position` 0 = runs next). Fields: `prompt` (required), `model`, `ratio`, `width`/`height`, `steps`, `seed` (default
  42), `transparent` (RGBA output, Qwen only), `upsample` (write the caption with the prompt upsampler, Klein only),
  `images` (up to 10 base64 images to edit). See [Models](#models) for what each model accepts; a task option a model
  does not take is a 400. 400 bad request, 429 `MAX_QUEUED` tasks waiting (with `Retry-After: 30`), 503 shutting down.
- `POST /v1/models/{model}/tasks` is the same with the model taken from the URL, which wins over a `model` in the body.
  404 when the model is unknown or not configured.
- `GET /v1/models` answers `{"data": [...]}`: what each configured model accepts, whether it is loaded, and its family's
  `precision` and `max_reference_pixels`.
- `GET /v1/tasks/{task_id}` answers `{"data": {...}}` with `task_id`, `status` (`CREATED` → `IN_PROGRESS` →
  `COMPLETED` or `FAILED`), `model`, `seed`, `position` (while `CREATED`), `step`/`steps` (denoising progress),
  `error` (when `FAILED`), `caption` (what the upsampler wrote), `created_at`/`started_at`/`finished_at` (unix seconds)
  and `elapsed_ms` (render time). 404 for an unknown id, or `TASK_TTL` seconds after the task finished.
- `GET /v1/tasks/{task_id}/image` returns the `image/png` once `COMPLETED`, with headers `X-Model`, `X-Seed`,
  `X-Steps`, `X-Elapsed-Ms`; 409 before that or if it failed. It can be fetched again until the task expires.
  `?format=jpeg` returns a compressed `image/jpeg` at the same pixel size, for example to send to a vision LLM.
  Transparency is flattened onto white.
- `GET /health` reports the device, the `precision` of each family, models, upsampler and queue: `queued` (waiting
  tasks), `busy`, `tasks` (retained).

Errors are JSON `{"error": ...}`. On SIGTERM/Ctrl+C the server stops taking tasks (503), finishes and stores the one
rendering, then exits; queued tasks stay `CREATED`. On the next start they run again in submit order (after any
`PRELOAD`), ahead of new tasks, and finished tasks can still be polled and downloaded until `TASK_TTL`. A task that was
rendering when the process died (crash, kill -9, a second Ctrl+C) is `FAILED` with `server restarted during render`.
A queued task whose model or upsampler is no longer configured fails with its validation error when it comes up. A
second server on the same `DB_PATH` refuses to start (it holds `<DB_PATH>.lock`).

```sh
# submit, poll, download
auth="Authorization: Bearer $IMAGEGEN_API_KEY"
id=$(curl -s localhost:8740/v1/tasks -H "$auth" -d '{"prompt": "A neon shop sign that reads QWEN", "ratio": "16:9"}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["task_id"])')
while :; do
  status=$(curl -s localhost:8740/v1/tasks/$id -H "$auth" | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["status"])')
  echo "$status"
  case $status in COMPLETED|FAILED) break;; esac
  sleep 2
done
curl -sf localhost:8740/v1/tasks/$id/image -H "$auth" -o base.png

# qwen-turbo
curl -s localhost:8740/v1/tasks -H "$auth" -d '{"prompt": "A corgi playing guitar in the rain", "model": "qwen-turbo"}'

# flux-klein, with the model in the URL
curl -s localhost:8740/v1/models/flux-klein/tasks -H "$auth" -d '{"prompt": "A neon shop sign that reads KLEIN", "ratio": "16:9"}'

# edit
python3 -c 'import base64, json, sys; json.dump({"prompt": "Make the sky a sunset", "images": [base64.b64encode(open(sys.argv[1], "rb").read()).decode()]}, sys.stdout)' photo.png > req.json
curl -s localhost:8740/v1/tasks -H "$auth" --data-binary @req.json

# what the configured models accept
curl -s localhost:8740/v1/models -H "$auth"

curl -s localhost:8740/health
```

The submit/poll/download shape and `{"data": {"task_id", "status"}}` envelope match Magnific's async API, so a provider
written for Magnific can follow the same submit/poll loop for every model here, with `POST /v1/models/{model}/tasks`
if it keeps one path per model. There are three differences:

- Tasks wait in a queue that renders one at a time, so queue time can exceed a timeout counted from submit. Start
  the deadline at `started_at` (status `IN_PROGRESS`) instead, and use `position` while `CREATED`.
- Download from `/v1/tasks/{task_id}/image` instead of `data.generated[0]`.
- Errors are `{"error": "..."}`, not Magnific's shape; read that key rather than printing the whole body.

## Prompts

Klein takes full scene captions, and a short prompt sent as typed gives generic output. Have an LLM write the caption
before submitting: use `t2i` from [doc/klein_system_prompts.json](doc/klein_system_prompts.json) as its system prompt,
or `i2i` with the reference images when editing. `prompt` goes to the model as sent.

With `FLUX_KLEIN_UPSAMPLER_PATH` set, a Klein request can send `"upsample": true` to have the server write the caption
itself with `flux-3-image-klein-pup`. The caption comes back in `caption` when polling the finished task.

## Memory

One family stays loaded at a time: `qwen-base` and `qwen-turbo` together, or `flux-klein` with its upsampler. A request
for the other family unloads the current one and loads its own model first, and the upsampler loads again on the next
`upsample` request. That load is part of the task, so its `step` stays `null` longer. The queue is strict FIFO, so
tasks alternating between families pay a full load each time; send a family's tasks together when you can.

A client with a stall deadline must not count the `null` stretch as a stall: start that deadline at the first `step`,
and give the time before it a limit of its own that covers a full load.

Base and Turbo share one text encoder and VAE, but each has its own transformer: both resident take about 44 GB,
which fits a 64 GB Mac. Configure only one of them on smaller machines. Klein takes about 20 GB of weights in `bf16`,
and the optional upsampler about 19 GB more.

## Adding a model

1. Add a `ModelSpec` entry to `MODELS` in `src/imagegen/schemas.py`, with what the model accepts. Name it maker first,
   like `qwen-turbo` and `flux-klein`.
2. For a new family, add an engine module in `src/imagegen/engine/` with `load()` and `render()`, like `flux.py`.
3. Add it to `ENGINES` in `src/imagegen/engine/registry.py`, keyed by `ModelSpec.family`.
4. Add its path in `src/imagegen/config.py`: a `Settings` field, read in `load_settings` and listed in `model_paths`.

A new model of an existing family, such as a later `flux-base`, skips steps 2 and 3, and stays in memory beside the
rest of its family. It shares that family's settings, so `flux-base` would take the `FLUX_` keys `flux-klein` does. It
does need `load()` and `render()` in its family's engine to handle it: today `flux.py` assumes the distilled checkpoint.

A new family needs no config work for its settings: `{FAMILY}_TRANSFORMER_PRECISION`, `{FAMILY}_TEXT_ENCODER_PRECISION`
and `{FAMILY}_MAX_REFERENCE_PIXELS` are read from its `ModelSpec.family`, uppercased, and it appears in `precision` on
its own.

## License

FLUX 3 Image Klein is under the FLUX Kommunity License, a non-commercial license with limited commercial use of outputs
for qualifying users. Check `LICENSE.md` in the weights before commercial use.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```
