# image-gen-server

A small on-demand HTTP server for Qwen-Image-2.1 (`base`) and Qwen-Image-2.1-Turbo (`turbo`).
Jobs run one at a time in a FIFO queue on a single worker thread. Each model loads the first time
it is used and stays in memory until the server stops. Clients submit a task, poll it and download the PNG.

## Setup

```sh
uv sync
cp .env.example .env   # then edit the paths
uv run --env-file .env image-gen-server
```

| Variable | Default | Meaning |
|---|---|---|
| `QWEN_IMAGE_21_PATH` | required | Qwen-Image-2.1 model directory |
| `QWEN_IMAGE_21_TURBO_PATH` | unset | Qwen-Image-2.1-Turbo model directory (turbo off if unset) |
| `QWEN_IMAGE_21_DEVICE` | `auto` | `auto`, `mps`, `cuda` or `cpu` (auto: MPS, else CUDA, else CPU) |
| `TRANSFORMER_PRECISION` | `bf16` | `bf16`, `int8`, `fp8` or `fp8_dynamic`: transformer weights, quantized while loading (no converted copy on disk). `fp8_dynamic` also quantizes activations and uses the FP8 matmul; it needs a CUDA GPU with FP8 support |
| `TEXT_ENCODER_PRECISION` | `bf16` | `bf16`, `int8` or `fp8`: text encoder weights, same as above (the VAE always stays `bf16`) |
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8740` | Port |
| `PRELOAD` | empty | Comma list of models to load at startup, e.g. `turbo` or `base,turbo` |
| `MAX_QUEUED` | `64` | Waiting (`CREATED`) tasks allowed before `POST /v1/tasks` answers 429 |
| `TASK_TTL` | `3600` | Seconds a finished task and its PNG are kept before polls return 404 |
| `DB_PATH` | `data/imagegen.sqlite3` | SQLite file holding tasks and PNGs (relative to the working directory; created if missing) |

## API

Renders take 30 s to 10 min, so the API is asynchronous: submit, poll, download. Tasks, their requests and PNGs are
kept in the SQLite file `DB_PATH`, so they survive a restart (see below).

- `POST /v1/tasks` takes JSON and answers `202 {"data": {"task_id", "status": "CREATED", "position"}}`
  (`position` 0 = runs next). Fields: `prompt` (required), `model` (`turbo`, the default when configured, or `base`), `ratio` (`1:1`, `4:3`, `3:4`,
  `3:2`, `2:3`, `16:9`, `9:16`), `width`/`height` (multiples of 32, 256..2752), `steps` (1..100, default 40; turbo
  always runs its own 8-step schedule), `seed` (default 42), `transparent` (RGBA output), `images` (up to 10 base64
  images to edit). 400 bad request, 429 `MAX_QUEUED` tasks waiting (with `Retry-After: 30`), 503 shutting down.
- `GET /v1/tasks/{task_id}` answers `{"data": {...}}` with `task_id`, `status` (`CREATED` → `IN_PROGRESS` →
  `COMPLETED` or `FAILED`), `model`, `seed`, `position` (while `CREATED`), `step`/`steps` (denoising progress),
  `error` (when `FAILED`), `created_at`/`started_at`/`finished_at` (unix seconds) and `elapsed_ms` (render time).
  404 for an unknown id, or `TASK_TTL` seconds after the task finished.
- `GET /v1/tasks/{task_id}/image` returns the `image/png` once `COMPLETED`, with headers `X-Model`, `X-Seed`,
  `X-Steps`, `X-Elapsed-Ms`; 409 before that or if it failed. It can be fetched again until the task expires.
  `?format=jpeg` returns a compressed `image/jpeg` at the same pixel size, for example to send to a vision LLM.
  Transparency is flattened onto white.
- `GET /health` reports the device, models and queue: `queued` (waiting tasks), `busy`, `tasks` (retained).

Errors are JSON `{"error": ...}`. On SIGTERM/Ctrl+C the server stops taking tasks (503), finishes and stores the one
rendering, then exits; queued tasks stay `CREATED`. On the next start they run again in submit order (after any
`PRELOAD`), ahead of new tasks, and finished tasks can still be polled and downloaded until `TASK_TTL`. A task that was
rendering when the process died (crash, kill -9, a second Ctrl+C) is `FAILED` with `server restarted during render`.
A queued task whose model is no longer configured fails with its validation error when it comes up. A second server on
the same `DB_PATH` refuses to start (it holds `<DB_PATH>.lock`).

```sh
# submit, poll, download
id=$(curl -s localhost:8740/v1/tasks -d '{"prompt": "A neon shop sign that reads QWEN", "ratio": "16:9"}' \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["task_id"])')
while :; do
  status=$(curl -s localhost:8740/v1/tasks/$id | python3 -c 'import json, sys; print(json.load(sys.stdin)["data"]["status"])')
  echo "$status"
  case $status in COMPLETED|FAILED) break;; esac
  sleep 2
done
curl -sf localhost:8740/v1/tasks/$id/image -o base.png

# turbo
curl -s localhost:8740/v1/tasks -d '{"prompt": "A corgi playing guitar in the rain", "model": "turbo"}'

# edit
python3 -c 'import base64, json, sys; json.dump({"prompt": "Make the sky a sunset", "images": [base64.b64encode(open(sys.argv[1], "rb").read()).decode()]}, sys.stdout)' photo.png > req.json
curl -s localhost:8740/v1/tasks --data-binary @req.json

curl -s localhost:8740/health
```

The submit/poll/download shape and `{"data": {"task_id", "status"}}` envelope match Magnific's async API, so a
`QwenProvider` in agents_service can follow `MagnificProvider`'s submit/poll loop, with three differences:

- Tasks wait in a queue that renders one at a time, so queue time can exceed Magnific's 600 s `_POLL_TIMEOUT` from
  submit. Start the deadline at `started_at` (status `IN_PROGRESS`) instead, and use `position` while `CREATED`.
- Download from `/v1/tasks/{task_id}/image` instead of `data.generated[0]`.
- Errors are `{"error": "..."}`, not Magnific's shape; read that key rather than printing the whole body.

## Memory

Base and Turbo share one text encoder and VAE, but each has its own transformer: both resident take about 44 GB,
which fits a 64 GB Mac. Models are never evicted; configure only `QWEN_IMAGE_21_PATH` on smaller machines.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```
