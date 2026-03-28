from __future__ import annotations

from typing import Any

import numpy as np


try:
    import lightning_core as _lc
except Exception:
    _lc = None


def lightning_vector_add(a: Any, b: Any, device: str = "metal") -> np.ndarray:
    if _lc is None:
        raise RuntimeError("lightning_core is not available. Install lightning-core first.")
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return _lc.vector_add(arr_a, arr_b, device)
