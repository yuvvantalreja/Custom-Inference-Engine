from __future__ import annotations

import torch


def activation_dtype(device: torch.device) -> torch.dtype:
    """FP16 on CUDA, FP32 on CPU. T4 has no bf16 tensor cores."""
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def kv_cache_dtype(device: torch.device) -> torch.dtype:
    """Match the activation dtype so attention doesn't mix FP32 K/V with FP16 Q."""
    return activation_dtype(device)


def accumulation_dtype() -> torch.dtype:
    return torch.float32


def weight_storage_dtype() -> torch.dtype:
    return torch.int8


def as_device(device: str | torch.device) -> torch.device:
    return torch.device(device) if isinstance(device, str) else device


def validate_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    if dtype is torch.bfloat16:
        raise ValueError("bfloat16 is not supported on sm_75 (T4)")
    return dtype
