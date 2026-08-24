"""Device and precision resolution (CPU/MPS/CUDA), plus the platform snapshot smoke_check uses.

Single code path across tiers: the auto policies supply per-backend defaults,
and explicit ``infra.device`` / ``infra.precision`` dotlist values override them.
"""

from __future__ import annotations

import os
import platform

import torch
from omegaconf import DictConfig

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def resolve_device(cfg: DictConfig) -> torch.device:
    """Resolve cfg.infra.device (auto|cpu|mps|cuda); auto prefers cuda > mps > cpu.

    An explicitly requested backend that is unavailable raises RuntimeError
    rather than silently falling back — tier results must come from the tier's
    declared hardware. ``infra.multi_gpu`` has no consumer yet and refuses to
    silently no-op.
    """
    if cfg.infra.get("multi_gpu", False):
        raise NotImplementedError("infra.multi_gpu: accelerate path lands with the training loop")
    choice = str(cfg.infra.device).lower()
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "infra.device=cuda but CUDA is not available on this machine "
                "(torch.cuda.is_available() is False). Use infra.device=auto or "
                "run on the node tier."
            )
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "infra.device=mps but MPS is not available on this machine "
                "(torch.backends.mps.is_available() is False). Use "
                "infra.device=auto or run on the mac tier."
            )
        return torch.device("mps")
    raise ValueError(f"Unknown infra.device {choice!r}: expected auto|cpu|mps|cuda")


def resolve_precision(cfg: DictConfig, device: torch.device) -> torch.dtype:
    """Resolve cfg.infra.precision (auto|fp32|fp16|bf16) to a torch dtype.

    Auto policy: cpu -> fp32, mps -> fp32 (fp16 is explicit opt-in), cuda -> bf16.
    An explicit choice always wins over the per-backend default.
    """
    choice = str(cfg.infra.precision).lower()
    if choice == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if choice in _DTYPES:
        return _DTYPES[choice]
    raise ValueError(f"Unknown infra.precision {choice!r}: expected auto|fp32|fp16|bf16")


def describe_platform() -> dict:
    """Snapshot of os/python/torch, device availability, cpu count, and RAM (GB)."""
    try:
        ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (AttributeError, OSError, ValueError):  # non-POSIX or sysconf key missing
        ram_gb = None
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
        "ram_gb": ram_gb,
    }
