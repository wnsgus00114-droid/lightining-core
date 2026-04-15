#!/usr/bin/env python3
"""Smoke test for coreml_roundtrip_beta_report."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> int:
    runner = lc_api.TinyTransformerRunner(seq_len=16, d_model=16, d_ff=32, vocab_size=64, seed=20260415)
    x = np.asarray(np.arange(16) % 64, dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix="lc_coreml_roundtrip_") as td:
        out = lc_api.coreml_roundtrip_beta_report(
            runner,
            x,
            bundle_dir=Path(td),
            mode="eager",
            device="cpu",
            route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
            warmup=1,
            iters=2,
        )
        _require(out.get("schema_version") == "coreml_roundtrip_beta_v1", "schema mismatch")
        _require(Path(str(out.get("checkpoint_path", ""))).exists(), "checkpoint not saved")
        _require(Path(str(out.get("manifest_path", ""))).exists(), "manifest not saved")
        _require("parity_pass" in out, "parity field missing")

    print("python coreml roundtrip beta smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
