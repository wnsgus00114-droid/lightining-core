#!/usr/bin/env python3
"""Python smoke: checkpoint IO v1.2 integrity contract."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    np.random.seed(20260410)
    _require(hasattr(lc_api, "validate_checkpoint"), "validate_checkpoint must exist")
    _require(hasattr(lc_api, "checkpoint_conversion_diagnostics"), "checkpoint_conversion_diagnostics must exist")

    model = lc_api.TinyMLPModel(8, 16, 4)
    with tempfile.TemporaryDirectory(prefix="lc_ckpt_v12_") as td:
        td_p = Path(td)
        ckpt = td_p / "tiny_v12.npz"
        lc_api.save_model_checkpoint(ckpt, model, metadata={"suite": "ckpt_v12_smoke"})
        valid = lc_api.validate_checkpoint(ckpt, strict=True)
        _require(bool(valid.get("ok", False)), "checkpoint should validate")
        _require(valid.get("meta", {}).get("format") == "lc_checkpoint_v1_2", "v1.2 format expected")
        _require(
            valid.get("meta", {}).get("integrity_signature") == "lc_integrity_sha256_v1",
            "integrity signature mismatch",
        )

        payload = lc_api.load_checkpoint(ckpt, validate=True)
        _require(payload.get("meta", {}).get("format") == "lc_checkpoint_v1_2", "load_checkpoint format mismatch")

        # Corrupt checkpoint tensor payload while keeping metadata unchanged.
        corrupt = td_p / "tiny_v12_corrupt.npz"
        with np.load(ckpt, allow_pickle=False) as data:
            save_dict = {k: np.asarray(data[k]).copy() for k in data.files}
        tensor_keys = [k for k in save_dict.keys() if k != "__lc_checkpoint_meta__"]
        _require(bool(tensor_keys), "checkpoint must contain tensor keys")
        first = tensor_keys[0]
        arr = np.asarray(save_dict[first]).reshape(-1)
        if arr.size > 0:
            arr[0] = arr[0] + np.float32(1.0)
        save_dict[first] = arr.reshape(np.asarray(save_dict[first]).shape)
        np.savez_compressed(corrupt, **save_dict)

        invalid = lc_api.validate_checkpoint(corrupt, strict=False)
        _require(not bool(invalid.get("ok", True)), "corrupted checkpoint must fail validation")
        _require(
            str(invalid.get("code", "")).startswith("checkpoint_"),
            "invalid checkpoint should return structured checkpoint_* code",
        )
        try:
            lc_api.validate_checkpoint(corrupt, strict=True)
            raise AssertionError("strict validate_checkpoint must raise on corrupted payload")
        except lc_api.CheckpointValidationError:
            pass

        diag = lc_api.checkpoint_conversion_diagnostics(
            {"a": np.zeros((2, 2), dtype=np.float32), "bad": object()},
            source_format="torch_state_dict",
        )
        _require(diag.get("source_format") == "torch_state_dict", "source_format override mismatch")
        _require(int(diag.get("total_entries", 0)) == 2, "diagnostic total_entries mismatch")
        _require(int(diag.get("convertible_entries", 0)) == 1, "diagnostic convertible_entries mismatch")
        _require(not bool(diag.get("convertible", True)), "diagnostic convertible should be false")

    print("python checkpoint integrity v1.2 smoke: ok")


if __name__ == "__main__":
    main()

