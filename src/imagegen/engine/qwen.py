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


def load_pipeline(path: str, device: str, shared: dict | None = None, transformer: str = "bf16", text_encoder: str = "bf16"):
    """A QwenImage21Pipeline from `path`, reusing already-loaded `shared` components instead of loading them again.

    `transformer`/`text_encoder` are "bf16", "int8", "fp8" or "fp8_dynamic"; each quantizes the Linear weights as they
    load (nothing saved). int8 and fp8 run in bf16; fp8_dynamic also quantizes activations and uses the FP8 matmul.
    """
    from diffusers import QwenImage21Pipeline  # type: ignore[import-not-found]

    kwargs = dict(shared or {})
    mapping = {}
    if (transformer, text_encoder) != ("bf16", "bf16"):
        import diffusers
        import transformers
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int8WeightOnlyConfig,
        )

        configs = {
            "int8": Int8WeightOnlyConfig,
            "fp8": Float8WeightOnlyConfig,
            "fp8_dynamic": Float8DynamicActivationFloat8WeightConfig,
        }
        if transformer != "bf16":
            mapping["transformer"] = diffusers.TorchAoConfig(configs[transformer]())
        if text_encoder != "bf16" and "text_encoder" not in kwargs:  # a shared one is already loaded as it is
            mapping["text_encoder"] = transformers.TorchAoConfig(configs[text_encoder]())
    if mapping:
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(quant_mapping=mapping)
    # device_map loads each weight straight to the GPU; passed components are used as-is, so only the rest load.
    # ponytail: assumes both checkpoints ship the same text_encoder/vae (file sizes match); load separately if not
    return QwenImage21Pipeline.from_pretrained(path, dtype=torch.bfloat16, device_map=device, **kwargs)


def shared_components(pipe) -> dict:
    """The components another Qwen-Image-2.1 pipeline can reuse."""
    return {name: getattr(pipe, name) for name in SHARED}


def trim_vae(vae) -> None:
    """Hold a decode near 3.7 GiB at any size: tile it, and drop the feature cache a single frame never reads."""
    # ponytail: patches private VAE methods, recheck on a diffusers bump
    clear_cache, encode = vae.clear_cache, vae._encode

    def clear_and_drop():
        clear_cache()
        vae._feat_map = vae._enc_feat_map = None

    def encode_untiled(x):  # tiles are for the decode; a reference stays whole
        vae.use_tiling = False
        try:
            return encode(x)
        finally:
            vae.use_tiling = True

    vae.clear_cache, vae._encode = clear_and_drop, encode_untiled
    vae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024,
                      tile_sample_stride_height=768, tile_sample_stride_width=768)


def reference_side(count: int, max_pixels: int) -> int:
    """The pipeline's output_resolution for `count` references: 1024 unless their pixels would pass `max_pixels`."""
    # ponytail: 4+ references with no size also shrink the output; ORA always sends a size
    return min(1024, int((max_pixels / max(count, 1)) ** 0.5) // 32 * 32)


def render(pipe, req: GenerateRequest, on_step=None, max_reference_pixels: int = 10 * 1024**2) -> tuple[Image.Image, int]:
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
        output_resolution=reference_side(len(images or ()), max_reference_pixels),
    ).images[0]
    return out, len(sigmas) if sigmas else req.steps
