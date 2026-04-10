#!/usr/bin/env python3
"""Python smoke: model runner alpha + wrapper surfaces."""

from __future__ import annotations

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def main() -> None:
    np.random.seed(20260410)
    _require(hasattr(lc_api, "TinyTransformerRunner"), "TinyTransformerRunner must exist")
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, seed=20260410)
    x = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)

    y_eager = runner.run(x, mode="eager", device="cpu")
    y_graph = runner.run(x, mode="graph", device="cpu")
    _require(np.allclose(y_eager, y_graph, atol=1.0e-4, rtol=1.0e-4), "runner eager/graph parity failed")

    if _torch_available():
        y_interop = runner.run(x, mode="interop", device="cpu")
        _require(np.allclose(y_eager, y_interop, atol=3.0e-3, rtol=3.0e-3), "runner eager/interop parity failed")
        wrapper = lc_api.create_torch_module_wrapper(runner, mode="eager", device="cpu")
        _require(wrapper is not None, "torch wrapper creation failed")

    try:
        _ = lc_api.create_tf_keras_layer_wrapper(runner, mode="eager", device="cpu")
    except RuntimeError:
        # TensorFlow is optional in CI; runtime error is acceptable when unavailable.
        pass

    print("python model runner alpha smoke: ok")


if __name__ == "__main__":
    main()

