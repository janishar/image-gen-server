"""Lazily loaded, resident pipelines keyed by model name."""

from __future__ import annotations

import torch

from ..schemas import MODELS, GenerateRequest
from . import qwen
from .device import patch_mps_empty_cache


class ModelRegistry:
    """Loads each configured model on first use and keeps it until the process exits.

    ponytail: both models resident ≈44 GB of a 64 GB Mac (text_encoder/vae shared); eviction left out,
    add an LRU drop + device.empty_cache if a third model or a smaller machine shows up.
    """

    def __init__(self, model_paths: dict[str, str], device: str, transformer: str = "bf16", text_encoder: str = "bf16",
                 memory_fraction: float = 0, max_reference_pixels: int = 10 * 1024**2):
        self.model_paths = dict(model_paths)
        self.device = device
        self.precision = {"transformer": transformer, "text_encoder": text_encoder}
        self.max_reference_pixels = max_reference_pixels
        self._pipes: dict[str, object] = {}
        if device == "mps":
            patch_mps_empty_cache()
        if device == "cuda" and memory_fraction:  # an overrun fails this render, not vLLM beside it
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

    @property
    def configured(self) -> tuple[str, ...]:
        return tuple(name for name in MODELS if name in self.model_paths)

    def status(self) -> dict[str, dict]:
        return {
            name: {"configured": name in self.model_paths, "loaded": name in self._pipes}
            for name in MODELS
        }

    def get(self, name: str):
        """The pipeline for `name`, loading it now if needed. Worker thread only."""
        if name not in self._pipes:
            other = next(iter(self._pipes.values()), None)
            shared = qwen.shared_components(other) if other is not None else None
            print(f"Loading {name} from {self.model_paths[name]}", flush=True)
            self._pipes[name] = qwen.load_pipeline(self.model_paths[name], self.device, shared, **self.precision)
            if other is None and self.device == "cuda":  # the VAE is shared, so once
                qwen.trim_vae(self._pipes[name].vae)
        return self._pipes[name]

    def render(self, req: GenerateRequest, on_step=None):
        try:
            return qwen.render(self.get(req.model), req, on_step, self.max_reference_pixels)
        finally:
            if self.device == "cuda":  # a vLLM restart needs this memory free
                torch.cuda.empty_cache()
