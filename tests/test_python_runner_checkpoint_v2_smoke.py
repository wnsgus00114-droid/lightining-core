#!/usr/bin/env python3
"""Python smoke: checkpoint IO v2 for runner/model manifest and compat loads."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    _require(hasattr(lc_api, "save_runner_checkpoint_v2"), "save_runner_checkpoint_v2 must exist")
    _require(hasattr(lc_api, "load_runner_checkpoint"), "load_runner_checkpoint must exist")
    _require(hasattr(lc_api, "runner_checkpoint_compatibility_matrix"), "runner_checkpoint_compatibility_matrix must exist")

    np.random.seed(20260411)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=20260411)
    x_tokens = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)
    ref = np.asarray(runner.run_tokens(x_tokens, mode="eager", device="cpu"), dtype=np.float32)
    optimizer_state = {
        "lr": np.asarray([1.0e-2], dtype=np.float32),
        "step": np.asarray([7], dtype=np.int32),
        "momentum": {"wq": np.zeros_like(runner.wq, dtype=np.float32)},
    }

    with tempfile.TemporaryDirectory(prefix="lc_runner_ckpt_v2_") as td:
        td_p = Path(td)
        ckpt_v2 = td_p / "runner_v2.npz"
        lc_api.save_runner_checkpoint_v2(
            ckpt_v2,
            runner,
            optimizer_state=optimizer_state,
            metadata={"suite": "runner_checkpoint_v2_smoke"},
        )
        valid = lc_api.validate_checkpoint(ckpt_v2, strict=True)
        _require(bool(valid.get("ok", False)), "v2 checkpoint must validate")
        _require(valid.get("meta", {}).get("format") == "lc_checkpoint_v2", "v2 checkpoint format mismatch")

        runner_loaded = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=1)
        opt_loaded: dict[str, np.ndarray] = {}
        payload = lc_api.load_runner_checkpoint(
            ckpt_v2,
            into_runner=runner_loaded,
            optimizer_into=opt_loaded,
            strict=True,
            validate=True,
        )
        _require(payload.get("format") == "lc_checkpoint_v2", "load_runner_checkpoint format mismatch")
        _require(bool(payload.get("manifest", {}).get("optimizer_present", False)), "optimizer manifest flag mismatch")
        out_loaded = np.asarray(runner_loaded.run_tokens(x_tokens, mode="eager", device="cpu"), dtype=np.float32)
        _require(np.allclose(ref, out_loaded, atol=1.0e-6, rtol=1.0e-6), "v2 load parity failed")
        _require(bool(opt_loaded), "optimizer state must be loaded into dict target")

        # Cross-version loader smoke: old v1.2 model checkpoint should still load into runner path.
        ckpt_v12 = td_p / "runner_v12.npz"
        lc_api.save_model_checkpoint(ckpt_v12, runner, metadata={"suite": "runner_checkpoint_v2_smoke"})
        runner_legacy = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=2)
        payload_legacy = lc_api.load_runner_checkpoint(ckpt_v12, into_runner=runner_legacy, strict=True, validate=True)
        _require(
            str(payload_legacy.get("format", "")) in {"lc_checkpoint_v1", "lc_checkpoint_v1_1", "lc_checkpoint_v1_2"},
            "legacy format should be accepted by runner checkpoint loader",
        )
        out_legacy = np.asarray(runner_legacy.run_tokens(x_tokens, mode="eager", device="cpu"), dtype=np.float32)
        _require(np.allclose(ref, out_legacy, atol=1.0e-6, rtol=1.0e-6), "legacy load parity failed")

    matrix = lc_api.runner_checkpoint_compatibility_matrix()
    rows = list(matrix.get("rows", []))
    _require(bool(rows), "runner checkpoint matrix rows must not be empty")
    row_v2 = next((r for r in rows if str(r.get("format", "")) == "lc_checkpoint_v2"), None)
    _require(row_v2 is not None, "runner checkpoint matrix must include v2 format")
    _require(bool(row_v2.get("save_runner_checkpoint_supported", False)), "v2 save support flag must be true")

    print("python runner checkpoint v2 smoke: ok")


if __name__ == "__main__":
    main()

