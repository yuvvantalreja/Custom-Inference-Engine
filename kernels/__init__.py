"""Custom kernels.

Triton kernels are only importable when CUDA + Triton are available.
The reference implementations in ``kernels.ref`` are always importable.
"""

from __future__ import annotations

import importlib

import torch

_TRITON_AVAILABLE = False
try:
    if torch.cuda.is_available():
        importlib.import_module("triton")
        _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


def triton_available() -> bool:
    return _TRITON_AVAILABLE
