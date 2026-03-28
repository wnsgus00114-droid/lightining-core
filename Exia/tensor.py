from __future__ import annotations

from typing import Any

import numpy as np

from .backend import get_backend

try:
    import torch
except Exception:
    torch = None


def default_device() -> str:
    backend = get_backend()
    if backend == "torch" and torch is not None:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "metal"


def tensor(data: Any, dtype: Any | None = None, device: str | None = None):
    backend = get_backend()
    if backend == "torch":
        if torch is None:
            raise RuntimeError("torch backend is selected but torch is not installed. Install with: pip install Exia[torch]")
        selected = device or default_device()
        return torch.tensor(data, dtype=dtype, device=selected)
    return np.asarray(data, dtype=dtype)


def as_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)
