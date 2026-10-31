"""What the engines share: quantizing weights while they load, the budget for reference pixels, and step reports."""

from __future__ import annotations

from PIL import Image


def quant_config(library, precision: str):
    """`library.TorchAoConfig` for "int8", "fp8" or "fp8_dynamic"; None for "bf16". Quantizes the Linear weights as
    they load (nothing saved). int8 and fp8 run in bf16; fp8_dynamic also quantizes activations and uses the FP8 matmul.
    """
    if precision == "bf16":
        return None
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
    return library.TorchAoConfig(configs[precision]())


def pipeline_quantization(transformer: str, text_encoder: str) -> dict:
    """The `quantization_config` keyword for a pipeline's from_pretrained, empty when both parts stay "bf16"."""
    mapping = {}
    if (transformer, text_encoder) != ("bf16", "bf16"):
        import diffusers
        import transformers

        if transformer != "bf16":
            mapping["transformer"] = quant_config(diffusers, transformer)
        if text_encoder != "bf16":
            mapping["text_encoder"] = quant_config(transformers, text_encoder)
    if not mapping:
        return {}
    from diffusers.quantizers import PipelineQuantizationConfig

    return {"quantization_config": PipelineQuantizationConfig(quant_mapping=mapping)}


def reference_side(count: int, max_pixels: int) -> int:
    """The side each of `count` references gets: 1024 unless their pixels would pass `max_pixels`."""
    return min(1024, int((max_pixels / max(count, 1)) ** 0.5) // 32 * 32)


def fit_references(images: list[Image.Image], max_pixels: int) -> list[Image.Image]:
    """`images` shrunk evenly to stay under `max_pixels` together. The pipelines fit several references under about 1 MP
    each and a single one themselves, so only a budget tighter than that changes anything here."""
    side = reference_side(len(images), max_pixels)
    if len(images) < 2 or side >= 1024:
        return images

    def shrink(img: Image.Image) -> Image.Image:
        scale = side / (img.width * img.height) ** 0.5
        size = (round(img.width * scale), round(img.height * scale))
        return img if scale >= 1 else img.resize(size, Image.Resampling.LANCZOS)

    return [shrink(img) for img in images]


def step_callback(on_step):
    """A `callback_on_step_end` that prints each step and calls on_step(done, total)."""

    def on_step_end(p, i: int, t, kwargs: dict) -> dict:
        print(f"@step {i + 1}/{p.num_timesteps}", flush=True)
        if on_step:
            on_step(i + 1, p.num_timesteps)
        return kwargs

    return on_step_end
