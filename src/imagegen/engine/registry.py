"""Lazily loaded pipelines keyed by model name, one model family in memory at a time."""

from __future__ import annotations

import gc

import torch

from ..config import FamilySettings
from ..schemas import MODELS, GenerateRequest
from . import flux, qwen
from .device import empty_cache, patch_mps_empty_cache

ENGINES = {"qwen": qwen, "flux": flux}  # by ModelSpec.family: each has load() and render()


class ModelRegistry:
    """Loads each configured model on first use and keeps it until a model of another family is asked for.

    ponytail: every family swap costs a full load, and tasks alternating between families pay it each time; run a
    family's queued tasks together if that hurts. Base and Turbo stay resident together, ≈44 GB in bf16 of a 64 GB Mac.
    """

    def __init__(self, model_paths: dict[str, str], device: str, families: dict[str, FamilySettings],
                 memory_fraction: float = 0, upsampler_path: str | None = None):
        self.model_paths = dict(model_paths)
        self.device = device
        self.families = dict(families)
        self.upsampler_path = upsampler_path
        self._pipes: dict[str, object] = {}
        self._upsampler = None
        if device == "mps":
            patch_mps_empty_cache()
        if device == "cuda" and memory_fraction:  # an overrun fails this render, not vLLM beside it
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

    @property
    def configured(self) -> tuple[str, ...]:
        return tuple(name for name in MODELS if name in self.model_paths)

    @property
    def precision(self) -> dict[str, dict]:
        return {
            family: {"transformer": s.transformer_precision, "text_encoder": s.text_encoder_precision}
            for family, s in self.families.items()
        }

    @property
    def can_upsample(self) -> bool:
        return self.upsampler_path is not None and "flux-klein" in self.model_paths

    def status(self) -> dict[str, dict]:
        return {
            name: {"configured": name in self.model_paths, "loaded": name in self._pipes}
            for name in MODELS
        }

    def upsampler_status(self) -> dict:
        return {"configured": self.can_upsample, "loaded": self._upsampler is not None}

    def _evict(self, family: str) -> None:
        """Free every model outside `family`, with Klein's upsampler, so `family` has the memory to load."""
        others = [name for name in self._pipes if MODELS[name].family != family]
        if not others:
            return
        print(f"Unloading {', '.join(others)}", flush=True)
        for name in others:
            del self._pipes[name]
        self._upsampler = None
        gc.collect()  # the patched VAE refers to itself, so only the collector frees it
        empty_cache(self.device)

    def get(self, name: str):
        """The pipeline for `name`, loading it now if needed. Worker thread only."""
        if name not in self._pipes:
            family = MODELS[name].family
            self._evict(family)
            print(f"Loading {name} from {self.model_paths[name]}", flush=True)
            loaded = list(self._pipes.values())
            precision = self.precision[family]
            self._pipes[name] = ENGINES[family].load(self.model_paths[name], self.device, loaded, **precision)
        return self._pipes[name]

    def upsampler(self) -> tuple:
        """The upsampler, loading it now if needed. Worker thread only."""
        if self._upsampler is None:
            print(f"Loading the upsampler from {self.upsampler_path}", flush=True)
            self._upsampler = flux.load_upsampler(self.upsampler_path, self.device)
        return self._upsampler

    def render(self, req: GenerateRequest, on_step=None):
        """(image, steps run, caption or None) for `req`."""
        try:
            pipe = self.get(req.model)
            extra = {"upsampler": self.upsampler()} if req.upsample else {}
            family = MODELS[req.model].family
            return ENGINES[family].render(pipe, req, on_step, self.families[family].max_reference_pixels, **extra)
        finally:
            if self.device == "cuda":  # a vLLM restart needs this memory free
                torch.cuda.empty_cache()
