"""The flux family's engine: load the FLUX 3 Image Klein pipeline and its prompt upsampler, and render one request."""

from __future__ import annotations

import math

import torch
from PIL import Image

from ..schemas import MODELS, GenerateRequest
from .common import fit_references, quant_config


def load_pipeline(path: str, device: str, transformer: str = "bf16", text_encoder: str = "bf16"):
    """The distilled pipeline from `path`, without the upsampling step: render runs that itself, per request."""
    import diffusers
    import transformers
    from diffusers import AutoencoderKLFlux2, Flux3DistilledAutoBlocks, Flux3Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    blocks = Flux3DistilledAutoBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline()
    # loaded by hand: the repo's modular_model_index.json points at the Hub, not at `path`
    common = {"dtype": torch.bfloat16, "device_map": device}
    pipe.update_components(
        transformer=Flux3Transformer2DModel.from_pretrained(
            path, subfolder="transformer", quantization_config=quant_config(diffusers, transformer), **common
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            f"{path}/text_encoder", quantization_config=quant_config(transformers, text_encoder), **common
        ),
        tokenizer=Qwen3VLProcessor.from_pretrained(f"{path}/text_encoder"),
        vae=AutoencoderKLFlux2.from_pretrained(path, subfolder="vae", **common),
    )
    return pipe


def load(path: str, device: str, loaded: list, transformer: str, text_encoder: str):
    """The pipeline at `path`. The family has one model, so `loaded` is always empty."""
    return load_pipeline(path, device, transformer, text_encoder)


def load_upsampler(path: str, device: str) -> tuple:
    """(model, processor) of the prompt upsampler at `path`."""
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
    return model.eval(), AutoProcessor.from_pretrained(path)


def canvas_line(width: int, height: int) -> str:
    """The first request line telling the text-to-image upsampler what shape to compose for."""
    divisor = math.gcd(width, height)
    return f"TARGET CANVAS: {width // divisor}:{height // divisor}\n"


def upsample(upsampler: tuple, prompt: str, images: list[Image.Image] | None, width: int | None, height: int | None) -> str:
    """The full scene caption the model takes, written from the short request `prompt` and any references."""
    from diffusers.modular_pipelines.flux3.prompt_upsampler import (
        MAX_NEW_TOKENS,
        prepare_references,
        upsample_prompt,
    )

    model, processor = upsampler
    request = prompt if images or not (width and height) else canvas_line(width, height) + prompt
    with torch.no_grad():
        return upsample_prompt(
            model,
            processor,
            request=request,
            images=prepare_references(images) if images else None,
            max_new_tokens=MAX_NEW_TOKENS,
        )


def render(pipe, req: GenerateRequest, on_step=None, max_reference_pixels: int = 10 * 1024**2,
           upsampler: tuple | None = None) -> tuple[Image.Image, int, str | None]:
    """Run `req` through `pipe`, calling on_step(done, total) after each step.

    Returns the image, the steps run and the caption the upsampler wrote (None when it did not run).
    """
    done = 0

    def on_forward(*_) -> None:  # the distilled denoise loop has no step callback, and runs the transformer once a step
        nonlocal done
        done += 1
        print(f"@step {done}/{req.steps}", flush=True)
        if on_step:
            on_step(done, req.steps)

    images = list(req.images) or None
    ratio = req.ratio or (None if images else "1:1")
    width, height = MODELS[req.model].ratios[ratio] if ratio else (None, None)
    width, height = req.width or width, req.height or height
    caption = upsample(upsampler, req.prompt, images, width, height) if req.upsample else None
    inputs = {"image": fit_references(images, max_reference_pixels)} if images else {}
    hook = pipe.transformer.register_forward_hook(on_forward)
    try:
        out = pipe(
            prompt=caption or req.prompt,
            width=width,
            height=height,
            num_inference_steps=req.steps,
            generator=torch.Generator(device="cpu").manual_seed(req.seed),
            output="images",
            **inputs,
        )[0]
    finally:
        hook.remove()
    return out, req.steps, caption
