"""Server settings from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .schemas import MODELS

DEVICES = ("auto", "mps", "cuda", "cpu")
PRECISIONS = ("bf16", "int8", "fp8", "fp8_dynamic")
TEXT_ENCODER_PRECISIONS = ("bf16", "int8", "fp8")  # fp8_dynamic's quantized activations would reach rendered text first
MIN_API_KEY = 16


@dataclass(frozen=True)
class FamilySettings:
    """Per ModelSpec.family, not per model: qwen-base and qwen-turbo share one text encoder."""

    transformer_precision: str = "bf16"
    text_encoder_precision: str = "bf16"
    max_reference_pixels: int = 10 * 1024**2


@dataclass(frozen=True)
class Settings:
    qwen_base_path: str | None
    qwen_turbo_path: str | None
    flux_klein_path: str | None
    fibo_edit_turbo_path: str | None
    upsampler_path: str | None
    device: str
    host: str
    port: int
    preload: tuple[str, ...]
    max_queued: int
    task_ttl: int
    db_path: str
    families: dict[str, FamilySettings]
    gpu_memory_fraction: float
    api_key: str = field(repr=False)

    @property
    def model_paths(self) -> dict[str, str]:
        """The configured models, by name."""
        paths = {
            "qwen-base": self.qwen_base_path,
            "qwen-turbo": self.qwen_turbo_path,
            "flux-klein": self.flux_klein_path,
            "fibo-edit-turbo": self.fibo_edit_turbo_path,
        }
        return {name: path for name, path in paths.items() if path}


def _model_dir(env, key: str) -> str | None:
    value = env.get(key, "").strip()
    if not value:
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


def _family(env, family: str) -> FamilySettings:
    """A family's settings: QWEN_X when set, else X, which covers every family."""

    def key(name: str) -> str:
        scoped = f"{family.upper()}_{name}"
        return scoped if env.get(scoped, "").strip() else name

    return FamilySettings(
        _precision(env, key("TRANSFORMER_PRECISION"), PRECISIONS),
        _precision(env, key("TEXT_ENCODER_PRECISION"), TEXT_ENCODER_PRECISIONS),
        _positive(env, key("MAX_REFERENCE_PIXELS"), FamilySettings.max_reference_pixels),
    )


def _api_key(env) -> str:
    value = env.get("IMAGEGEN_API_KEY", "").strip()
    if len(value) < MIN_API_KEY:
        raise ValueError(f"IMAGEGEN_API_KEY is required: a random string of {MIN_API_KEY}+ characters")
    return value


def load_settings(env=os.environ) -> Settings:
    """Read and check the settings, raising ValueError on anything wrong."""
    qwen_base = _model_dir(env, "QWEN_IMAGE_21_PATH")
    qwen_turbo = _model_dir(env, "QWEN_IMAGE_21_TURBO_PATH")
    flux_klein = _model_dir(env, "FLUX_KLEIN_PATH")
    fibo_edit_turbo = _model_dir(env, "FIBO_EDIT_TURBO_PATH")
    upsampler = _model_dir(env, "FLUX_KLEIN_UPSAMPLER_PATH")

    device = env.get("IMAGEGEN_DEVICE", "").strip() or "auto"
    if device not in DEVICES:
        raise ValueError(f"IMAGEGEN_DEVICE must be one of {', '.join(DEVICES)}, got {device!r}")

    port_text = env.get("PORT", "").strip() or "8740"
    try:
        port = int(port_text)
    except ValueError:
        raise ValueError(f"PORT must be an integer, got {port_text!r}") from None
    if not 0 <= port <= 65535:
        raise ValueError(f"PORT out of range: {port}")

    settings = Settings(
        qwen_base_path=qwen_base,
        qwen_turbo_path=qwen_turbo,
        flux_klein_path=flux_klein,
        fibo_edit_turbo_path=fibo_edit_turbo,
        upsampler_path=upsampler,
        device=device,
        host=env.get("HOST", "").strip() or "127.0.0.1",
        port=port,
        preload=tuple(n.strip() for n in env.get("PRELOAD", "").split(",") if n.strip()),
        max_queued=_positive(env, "MAX_QUEUED", 64),
        task_ttl=_positive(env, "TASK_TTL", 3600),
        db_path=os.path.abspath(os.path.expanduser(env.get("DB_PATH", "").strip() or "data/imagegen.sqlite3")),
        families={family: _family(env, family) for family in dict.fromkeys(m.family for m in MODELS.values())},
        gpu_memory_fraction=_fraction(env, "GPU_MEMORY_FRACTION"),
        api_key=_api_key(env),
    )
    if not settings.model_paths:
        raise ValueError("no model configured: set QWEN_IMAGE_21_PATH, QWEN_IMAGE_21_TURBO_PATH, FLUX_KLEIN_PATH or FIBO_EDIT_TURBO_PATH")
    for name in settings.preload:
        if name not in settings.model_paths:
            raise ValueError(f"PRELOAD: model {name!r} is not configured (configured: {', '.join(settings.model_paths)})")
    if len({MODELS[name].family for name in settings.preload}) > 1:
        raise ValueError("PRELOAD: one family stays loaded at a time, so name models of one family")
    return settings
