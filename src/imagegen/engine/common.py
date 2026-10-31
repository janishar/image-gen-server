"""What the engines share: quantizing weights while they load, and the budget for reference pixels."""

from __future__ import annotations


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


def reference_side(count: int, max_pixels: int) -> int:
    """The side each of `count` references gets: 1024 unless their pixels would pass `max_pixels`."""
    return min(1024, int((max_pixels / max(count, 1)) ** 0.5) // 32 * 32)
