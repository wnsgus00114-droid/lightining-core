#!/usr/bin/env python3
"""Python smoke test: checkpoint IO v1 for integrated API objects."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    return bool(np.allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=rtol))


def main() -> None:
    np.random.seed(20260408)

    _require(hasattr(lc_api, "save_checkpoint"), "save_checkpoint must exist")
    _require(hasattr(lc_api, "load_checkpoint"), "load_checkpoint must exist")
    _require(hasattr(lc_api.Linear, "state_dict"), "Linear.state_dict must exist")
    _require(hasattr(lc_api.Linear, "load_state_dict"), "Linear.load_state_dict must exist")
    _require(hasattr(lc, "api") and hasattr(lc.api, "save_checkpoint"), "lc.api.save_checkpoint must be installed")
    _require(hasattr(lc.api, "load_checkpoint"), "lc.api.load_checkpoint must be installed")

    linear = lc_api.Linear(8, 4, bias=True)
    linear.weight = (np.arange(8 * 4, dtype=np.float32).reshape(8, 4) * 0.01) - 0.1
    linear.bias = np.asarray([0.1, -0.2, 0.3, -0.4], dtype=np.float32)

    x = np.linspace(-1.0, 1.0, num=16, dtype=np.float32).reshape(2, 8)
    ref = linear(x)

    with tempfile.TemporaryDirectory(prefix="lc_ckpt_") as td:
        ckpt_path = Path(td) / "linear_v1.npz"
        saved = lc_api.save_checkpoint(
            ckpt_path,
            linear,
            metadata={"suite": "checkpoint_smoke", "kind": "linear"},
            compressed=True,
        )
        _require(Path(saved).exists(), "checkpoint file must exist after save")

        linear.weight.fill(0.0)
        linear.bias.fill(0.0)
        changed = linear(x)
        _require(not _allclose(ref, changed), "modified parameters should change output")

        load_out = lc_api.load_checkpoint(ckpt_path, into=linear, strict=True)
        _require(isinstance(load_out, dict), "load_checkpoint should return dict payload")
        _require(isinstance(load_out.get("meta", {}), dict), "checkpoint meta must be dict")
        _require(load_out.get("meta", {}).get("format") == "lc_checkpoint_v1", "checkpoint format mismatch")

        restored = linear(x)
        _require(_allclose(ref, restored), "restored linear output must match reference")

        # Conv resident block state round-trip.
        conv_w = np.random.rand(16, 3, 3, 3).astype(np.float32)
        conv_b = np.random.rand(16).astype(np.float32)
        block_src = lc_api.ConvReLUResidentBlock(conv_w, conv_b, stride_h=1, stride_w=1, pad_h=1, pad_w=1, device="cpu")
        block_dst = lc_api.ConvReLUResidentBlock(np.zeros_like(conv_w), np.zeros_like(conv_b), stride_h=1, stride_w=1, pad_h=1, pad_w=1, device="cpu")

        block_ckpt = Path(td) / "conv_block_v1.npz"
        lc_api.save_checkpoint(block_ckpt, block_src.state_dict(prefix="conv"), metadata={"kind": "conv_block"})
        loaded_block = lc_api.load_checkpoint(block_ckpt)
        block_dst.load_state_dict(loaded_block["state"], strict=True, prefix="conv")

        _require(_allclose(block_src.w, block_dst.w), "conv block weight restore mismatch")
        _require(block_src.b is not None and block_dst.b is not None, "conv block bias must exist")
        _require(_allclose(block_src.b, block_dst.b), "conv block bias restore mismatch")

    print("python checkpoint IO smoke: ok")


if __name__ == "__main__":
    main()
