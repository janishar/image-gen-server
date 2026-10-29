"""The /generate request: its fields, defaults and validation."""

from __future__ import annotations

import base64
import binascii
import io
import struct
from collections.abc import Collection
from dataclasses import dataclass, fields, replace

from PIL import Image

ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

MODELS = ("base", "turbo")
MAX_IMAGES = 10
MAX_INPUT_PIXELS = 2 * 2752 * 2752  # twice the largest output; a 12 MP phone photo fits


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    model: str = "base"
    ratio: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int = 40
    seed: int = 42
    transparent: bool = False
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


def _decode_image(i: int, text: object) -> bytes:
    if not isinstance(text, str):
        raise ValidationError(f"images[{i}] must be a base64 string")
    try:
        raw = base64.b64decode(text, validate=True)
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_INPUT_PIXELS:
            raise ValidationError(f"images[{i}] is {img.width}x{img.height}, over {MAX_INPUT_PIXELS} pixels")
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


def parse_request(data: object, available_models: Collection[str]) -> GenerateRequest:
    """Check a decoded JSON body and turn it into a GenerateRequest, raising ValidationError on anything wrong."""
    if not isinstance(data, dict):
        raise ValidationError("request body must be a JSON object")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise ValidationError(f"unknown field(s): {', '.join(unknown)}")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("prompt must be a non-empty string")

    model = data.get("model", "base")
    if model not in MODELS:
        raise ValidationError(f"model must be one of {', '.join(MODELS)}")
    if model not in available_models:
        raise ValidationError(f"model {model!r} is not configured")

    ratio = data.get("ratio")
    if ratio is not None and (not isinstance(ratio, str) or ratio not in ASPECT_RATIOS):
        raise ValidationError(f"ratio must be one of {', '.join(ASPECT_RATIOS)}")

    transparent = data.get("transparent", False)
    if not isinstance(transparent, bool):
        raise ValidationError("transparent must be a boolean")

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
        width=_int(data, "width", 256, 2752, 32),
        height=_int(data, "height", 256, 2752, 32),
        steps=40 if steps is None else steps,
        seed=42 if seed is None else seed,
        transparent=transparent,
        images=tuple(_decode_image(i, text) for i, text in enumerate(images)),
    )
