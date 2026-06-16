"""Custom CUDA kernels backed by Triton.

Reference implementations live in kernels.ref for correctness checks.
Production kernel entry points require CUDA at runtime.
"""

from __future__ import annotations

import importlib

import torch


def require_triton() -> None:
    """Verify CUDA and Triton are available before running a kernel."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available")
    try:
        importlib.import_module("triton")
    except ImportError as e:
        raise RuntimeError(
            "Triton is required. Install with: pip install triton"
        ) from e
