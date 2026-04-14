#!/usr/bin/env python3
"""Python smoke: tune-cache v2 header format regression guard (matmul/attention)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


HEADER = "#lc_tune_cache"


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _read_lines(path: Path) -> list[str]:
    return [ln.rstrip("\n") for ln in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    if not bool(lc.metal_available()):
        print("python tune cache format smoke: skipped (metal unavailable)")
        return

    np.random.seed(20260414)

    with tempfile.TemporaryDirectory(prefix="lc_tune_cache_fmt_") as td:
        root = Path(td)
        matmul_cache = root / ".lightning_core_matmul_tune_cache.csv"
        attn_cache = root / ".lightning_core_attn_tune_cache.csv"

        # Seed with v1-compatible headers to ensure backward-compat read path
        # and then verify v2 rewrite header after a fresh tuning key is observed.
        matmul_cache.write_text(
            "\n".join(
                [
                    "#lc_tune_cache,matmul,1",
                    "64,64,64,16,1,0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        attn_cache.write_text(
            "\n".join(
                [
                    "#lc_tune_cache,attention,1",
                    "fwd,8,8,0,0,0,0",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        prev = Path.cwd()
        os.chdir(root)
        try:
            a = (np.random.standard_normal((16, 16)) * 0.2).astype(np.float32)
            b = (np.random.standard_normal((16, 16)) * 0.2).astype(np.float32)
            _ = lc_api.lightning_matmul(a, b, device="metal")

            q = (np.random.standard_normal((16, 16)) * 0.2).astype(np.float32)
            k = (np.random.standard_normal((16, 16)) * 0.2).astype(np.float32)
            v = (np.random.standard_normal((16, 16)) * 0.2).astype(np.float32)
            _ = lc_api.lightning_attention(q, k, v, seq=16, head_dim=16, device="metal", causal=False)
        finally:
            os.chdir(prev)

        _require(matmul_cache.exists(), "matmul tune cache file missing")
        _require(attn_cache.exists(), "attention tune cache file missing")

        matmul_lines = _read_lines(matmul_cache)
        attn_lines = _read_lines(attn_cache)
        _require(len(matmul_lines) >= 2, "matmul tune cache lines too short")
        _require(len(attn_lines) >= 2, "attention tune cache lines too short")

        _require(matmul_lines[0] == f"{HEADER},matmul,2", f"unexpected matmul cache header: {matmul_lines[0]}")
        _require(attn_lines[0] == f"{HEADER},attention,2", f"unexpected attention cache header: {attn_lines[0]}")
        _require(matmul_lines[1].startswith("#columns,"), "matmul cache missing #columns line")
        _require(attn_lines[1].startswith("#columns,"), "attention cache missing #columns line")

    print("python tune cache format smoke: ok")


if __name__ == "__main__":
    main()
