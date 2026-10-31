"""The task request: its fields, defaults and what each model accepts."""

from __future__ import annotations

import base64
import binascii
import io
import struct
from collections.abc import Collection
from dataclasses import dataclass, fields, replace

from PIL import Image

QWEN_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

FLUX_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1360, 768),
    "9:16": (768, 1360),
}

MAX_IMAGES = 10
OPTIONS = ("transparent", "upsample")


@dataclass(frozen=True)
class ModelSpec:
    """What one model accepts. Models of a family share memory, and one family stays loaded at a time."""

    family: str
    ratios: dict[str, tuple[int, int]]
    min_side: int
    max_side: int
    side_multiple: int
    steps: int  # when a request sends none
    max_steps: int  # a longer request is cut to this, so a client can send one number to every model
    options: tuple[str, ...] = ()  # the OPTIONS this model takes

    @property
    def max_input_pixels(self) -> int:
        return 2 * self.max_side**2  # twice the largest output; a 12 MP phone photo fits


_QWEN = ModelSpec("qwen", QWEN_RATIOS, 256, 2752, 32, steps=40, max_steps=100, options=("transparent",))
MODELS = {
    "qwen-base": _QWEN,
    "qwen-turbo": _QWEN,  # takes `steps` but always runs its own 8-step schedule
    "flux-klein": ModelSpec("flux", FLUX_RATIOS, 64, 4096, 16, steps=16, max_steps=16, options=("upsample",)),
}


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    model: str = "qwen-turbo"
    ratio: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int = 40
    seed: int = 42
    transparent: bool = False
    upsample: bool = False
    images: tuple = ()  # raw image bytes once parsed (small while queued); PIL images after load_images


_KEYS = {f.name for f in fields(GenerateRequest)}


def _int(data: dict, key: str, lo: int, hi: int, multiple: int = 1) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if type(value) is not int or not lo <= value <= hi or value % multiple:
        extra = f", a multiple of {multiple}" if multiple > 1 else ""
        raise ValidationError(f"{key} must be an integer in {lo}..{hi}{extra}")
    return value


def _decode_image(i: int, text: object, max_pixels: int) -> bytes:
    if not isinstance(text, str):
        raise ValidationError(f"images[{i}] must be a base64 string")
    try:
        raw = base64.b64decode(text, validate=True)
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > max_pixels:
            raise ValidationError(f"images[{i}] is {img.width}x{img.height}, over {max_pixels} pixels")
        img.load()  # decoded only to validate; the worker decodes again in load_images
    except ValidationError:
        raise
    except (binascii.Error, OSError, ValueError, SyntaxError, IndexError, TypeError, struct.error,
            Image.DecompressionBombError) as e:
        raise ValidationError(f"images[{i}] is not a valid image: {e}") from None
    return raw


def load_images(req: GenerateRequest) -> GenerateRequest:
    """`req` with its validated image bytes decoded, for the render."""
    images = []
    for raw in req.images:
        img = Image.open(io.BytesIO(raw))
        img.load()
        images.append(img)
    return replace(req, images=tuple(images))


def dump_request(req: GenerateRequest) -> dict:
    """`req` as a JSON-able dict that parse_request turns back into it (images as base64)."""
    data = {f.name: getattr(req, f.name) for f in fields(GenerateRequest)}
    data["images"] = [base64.b64encode(raw).decode() for raw in req.images]
    return data


def parse_request(data: object, available_models: Collection[str], can_upsample: bool = True) -> GenerateRequest:
    """Check a decoded JSON body and turn it into a GenerateRequest, raising ValidationError on anything wrong."""
    if not isinstance(data, dict):
        raise ValidationError("request body must be a JSON object")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise ValidationError(f"unknown field(s): {', '.join(unknown)}")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("prompt must be a non-empty string")

    model = data.get("model", "qwen-turbo" if "qwen-turbo" in available_models else next(iter(available_models)))
    if not isinstance(model, str) or model not in MODELS:
        raise ValidationError(f"model must be one of {', '.join(MODELS)}")
    if model not in available_models:
        raise ValidationError(f"model {model!r} is not configured")
    spec = MODELS[model]

    ratio = data.get("ratio")
    if ratio is not None and (not isinstance(ratio, str) or ratio not in spec.ratios):
        raise ValidationError(f"ratio must be one of {', '.join(spec.ratios)}")

    for option in OPTIONS:
        value = data.get(option, False)
        if not isinstance(value, bool):
            raise ValidationError(f"{option} must be a boolean")
        if value and option not in spec.options:
            raise ValidationError(f"model {model!r} has no {option} option")
    if data.get("upsample") and not can_upsample:
        raise ValidationError("the prompt upsampler is not configured")

    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValidationError("images must be a list of base64 strings")
    if len(images) > MAX_IMAGES:
        raise ValidationError(f"at most {MAX_IMAGES} images")

    steps = _int(data, "steps", 1, 100)
    seed = _int(data, "seed", 0, 2**63 - 1)
    return GenerateRequest(
        prompt=prompt,
        model=model,
        ratio=ratio,
        width=_int(data, "width", spec.min_side, spec.max_side, spec.side_multiple),
        height=_int(data, "height", spec.min_side, spec.max_side, spec.side_multiple),
        steps=spec.steps if steps is None else min(steps, spec.max_steps),
        seed=42 if seed is None else seed,
        transparent=data.get("transparent", False),
        upsample=data.get("upsample", False),
        images=tuple(_decode_image(i, text, spec.max_input_pixels) for i, text in enumerate(images)),
    )
