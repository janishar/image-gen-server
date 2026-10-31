"""The fibo family's engine: load the Bria Fibo 2.0 Edit Turbo pipeline and render one request."""

from __future__ import annotations

import torch
from PIL import Image

from ..schemas import MODELS, GenerateRequest
from .common import fit_references, pipeline_quantization, step_callback


def load(path: str, device: str, loaded: list, transformer: str, text_encoder: str):
    """The pipeline at `path`. The family has one model, so `loaded` is always empty."""
    from diffusers import BriaFibo2Pipeline  # type: ignore[import-not-found]

    quantization = pipeline_quantization(transformer, text_encoder)
    return BriaFibo2Pipeline.from_pretrained(path, dtype=torch.bfloat16, device_map=device, **quantization)


def render(pipe, req: GenerateRequest, on_step=None,
           max_reference_pixels: int = 10 * 1024**2) -> tuple[Image.Image, int, None]:
    """Run `req` through `pipe`, calling on_step(done, total) after each step; returns the image, the steps run and
    no caption (only Klein writes one). With no size in `req` the pipeline takes it from the first image."""
    width, height = MODELS[req.model].ratios[req.ratio] if req.ratio else (None, None)
    out = pipe(
        prompt=req.prompt,
        image=fit_references(list(req.images), max_reference_pixels),
        width=req.width or width,
        height=req.height or height,
        num_inference_steps=req.steps,
        guidance_scale=1.0,  # the turbo checkpoint runs without guidance, so a negative prompt would do nothing
        generator=torch.Generator(device="cpu").manual_seed(req.seed),
        callback_on_step_end=step_callback(on_step),
    ).images[0]
    return out, req.steps, None
