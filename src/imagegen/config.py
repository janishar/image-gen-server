"""Server settings from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DEVICES = ("auto", "mps", "cuda", "cpu")
PRECISIONS = ("bf16", "int8", "fp8", "fp8_dynamic")
TEXT_ENCODER_PRECISIONS = ("bf16", "int8", "fp8")  # fp8_dynamic's quantized activations would reach rendered text first
MIN_API_KEY = 16


@dataclass(frozen=True)
class Settings:
    base_path: str | None
    turbo_path: str | None
    device: str
    host: str
    port: int
    preload: tuple[str, ...]
    max_queued: int
    task_ttl: int
    db_path: str
    transformer_precision: str
    text_encoder_precision: str
    gpu_memory_fraction: float
    max_reference_pixels: int
    api_key: str = field(repr=False)

    @property
    def model_paths(self) -> dict[str, str]:
        """The configured models, by name."""
        paths = {"base": self.base_path, "turbo": self.turbo_path}
        return {name: path for name, path in paths.items() if path}


def _model_dir(env, key: str, required: bool) -> str | None:
    value = env.get(key, "").strip()
    if not value:
        if required:
            raise ValueError(f"{key} is required: the Qwen-Image-2.1 model directory")
        return None
    path = os.path.expanduser(value)
    if not os.path.isdir(path):
        raise ValueError(f"{key}: model directory not found: {path}")
    return path


def _positive(env, key: str, default: int) -> int:
    text = env.get(key, "").strip() or str(default)
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got {text!r}") from None
    if value < 1:
        raise ValueError(f"{key} must be at least 1, got {value}")
    return value


def _precision(env, key: str, choices: tuple[str, ...]) -> str:
    value = env.get(key, "").strip() or "bf16"
    if value not in choices:
        raise ValueError(f"{key} must be one of {', '.join(choices)}, got {value!r}")
    return value


def _fraction(env, key: str) -> float:
    text = env.get(key, "").strip() or "0"
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {text!r}") from None
    if not 0 <= value <= 1:
        raise ValueError(f"{key} must be between 0 and 1, got {value}")
    return value


def _api_key(env) -> str:
    value = env.get("IMAGEGEN_API_KEY", "").strip()
    if len(value) < MIN_API_KEY:
        raise ValueError(f"IMAGEGEN_API_KEY is required: a random string of {MIN_API_KEY}+ characters")
    return value


def load_settings(env=os.environ) -> Settings:
    """Read and check the settings, raising ValueError on anything wrong."""
    base = _model_dir(env, "QWEN_IMAGE_21_PATH", required=True)
    turbo = _model_dir(env, "QWEN_IMAGE_21_TURBO_PATH", required=False)

    device = env.get("QWEN_IMAGE_21_DEVICE", "").strip() or "auto"
    if device not in DEVICES:
        raise ValueError(f"QWEN_IMAGE_21_DEVICE must be one of {', '.join(DEVICES)}, got {device!r}")

    port_text = env.get("PORT", "").strip() or "8740"
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"PORT must be an integer, got {port_text!r}") from None
    if not 0 <= port <= 65535:
        raise ValueError(f"PORT out of range: {port}")

    settings = Settings(
        base_path=base,
        turbo_path=turbo,
        device=device,
        host=env.get("HOST", "").strip() or "127.0.0.1",
        port=port,
        preload=tuple(n.strip() for n in env.get("PRELOAD", "").split(",") if n.strip()),
        max_queued=_positive(env, "MAX_QUEUED", 64),
        task_ttl=_positive(env, "TASK_TTL", 3600),
        db_path=os.path.abspath(os.path.expanduser(env.get("DB_PATH", "").strip() or "data/imagegen.sqlite3")),
        transformer_precision=_precision(env, "TRANSFORMER_PRECISION", PRECISIONS),
        text_encoder_precision=_precision(env, "TEXT_ENCODER_PRECISION", TEXT_ENCODER_PRECISIONS),
        gpu_memory_fraction=_fraction(env, "GPU_MEMORY_FRACTION"),
        max_reference_pixels=_positive(env, "MAX_REFERENCE_PIXELS", 10 * 1024**2),
        api_key=_api_key(env),
    )
    for name in settings.preload:
        if name not in settings.model_paths:
            raise ValueError(f"PRELOAD: model {name!r} is not configured (configured: {', '.join(settings.model_paths)})")
    return settings
