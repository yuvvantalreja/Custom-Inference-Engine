from __future__ import annotations

import torch


def require_cuda(device: str | torch.device | None = None) -> torch.device:
    """Return a CUDA device, raising if CUDA is unavailable or device is not cuda."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    if device is None:
        return torch.device("cuda")
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        raise ValueError(f"GPU-only engine: expected cuda device, got {dev}")
    return dev
