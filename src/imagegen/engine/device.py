"""Torch device selection and cache helpers."""

from __future__ import annotations

import torch


def patch_mps_empty_cache() -> None:
    # torch 2.14: mps.empty_cache() waits on the GPU while holding the GIL, but finishing the
    # non_blocking weight copies device_map makes needs the GIL -> deadlock. synchronize() releases it.
    # ponytail: global monkeypatch, drop once torch's empty_cache releases the GIL itself
    empty_cache = torch.mps.empty_cache

    def safe_empty_cache() -> None:
        torch.mps.synchronize()
        empty_cache()

    torch.mps.empty_cache = safe_empty_cache


def pick_device(requested: str) -> str:
    """The torch device to run on: the one asked for, or with "auto" the first of MPS, CUDA, CPU present."""
    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        return next(d for d, ok in available.items() if ok)
    if not available[requested]:
        raise ValueError(f"{requested.upper()} is not available in this torch build")
    return requested


def empty_cache(device: str) -> None:
    """Hand freed memory back to the device, so the next model has room."""
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
