"""Load Qwen-Image-2.1 pipelines and render one request."""

from __future__ import annotations

import hashlib

import torch
from PIL import Image

from ..schemas import ASPECT_RATIOS, GenerateRequest

SHARED = ("text_encoder", "vae", "processor")


def noise_seed(seed: int, images: list[Image.Image] | None) -> int:
    """The seed for the starting noise: `seed` itself, or for an edit, `seed` mixed with the inputs' pixels.

    Editing an image with the seed and size it was generated with would start from the exact noise that made it,
    and the model then over-sharpens it into a crunchy, HDR-like texture instead of editing. Mixing in the inputs
    gives an edit its own noise, while the same seed and inputs still reproduce the same take.
    """
    if not images:
        return seed
    digest = hashlib.sha256(str(seed).encode())
    for img in images:
        digest.update(f"{img.mode}{img.size}".encode())
        digest.update(img.tobytes())
    return int.from_bytes(digest.digest()[:8], "big") >> 1


def load_pipeline(path: str, device: str, shared: dict | None = None):
    """A QwenImage21Pipeline from `path`, reusing already-loaded `shared` components instead of loading them again."""
    from diffusers import QwenImage21Pipeline  # type: ignore[import-not-found]

    # device_map loads each weight straight to the GPU; passed components are used as-is, so only the rest load.
    # ponytail: assumes both checkpoints ship the same text_encoder/vae (file sizes match); load separately if not
    return QwenImage21Pipeline.from_pretrained(path, dtype=torch.bfloat16, device_map=device, **(shared or {}))


def shared_components(pipe) -> dict:
    """The components another Qwen-Image-2.1 pipeline can reuse."""
    return {name: getattr(pipe, name) for name in SHARED}


def render(pipe, req: GenerateRequest, on_step=None) -> tuple[Image.Image, int]:
    """Run `req` through `pipe`, calling on_step(done, total) after each step; returns the image and steps run."""

    def on_step_end(p, i: int, t, kwargs: dict) -> dict:
        print(f"@step {i + 1}/{p.num_timesteps}", flush=True)
        if on_step:
            on_step(i + 1, p.num_timesteps)
        return kwargs

    images = list(req.images) or None
    ratio = req.ratio or (None if images else "1:1")
    width, height = ASPECT_RATIOS[ratio] if ratio else (None, None)
    width, height = req.width or width, req.height or height
    prompt = req.prompt
    if req.transparent:
        prompt = (
            "This is an RGBA image with transparency. "
            f"{prompt} The image has alpha channel and the background is transparent."
        )
    sigmas = pipe.config.get("sample_sigmas")
    out = pipe(
        prompt=prompt,
        image=images,
        width=width,
        height=height,
        num_inference_steps=req.steps,
        generator=torch.Generator(device="cpu").manual_seed(noise_seed(req.seed, images)),
        callback_on_step_end=on_step_end,
    ).images[0]
    return out, len(sigmas) if sigmas else req.steps
