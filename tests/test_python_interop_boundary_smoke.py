#!/usr/bin/env python3
"""Python smoke: interop boundary hardening contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    _require(hasattr(lc_api, "validate_route_policy"), "validate_route_policy must exist")
    ok = lc_api.validate_route_policy({"conv": "auto", "attention": "auto", "graph": "auto"})
    _require(bool(ok.get("ok", False)), "default route_policy should validate")

    bad = lc_api.validate_route_policy({"conv": "invalid_engine"}, strict=False)
    _require(not bool(bad.get("ok", True)), "invalid policy should fail")
    _require(str(bad.get("code", "")).startswith("interop_route_policy_"), "structured route error code expected")

    x = np.random.rand(1, 3, 8, 8).astype(np.float32)
    w = np.random.rand(16, 3, 3, 3).astype(np.float32)
    b = np.random.rand(16).astype(np.float32)
    report = lc.api.conv_attention_torchstrong_nchw_route_report(
        x, w, b, 96, 48, 1, 1, 1, 1, "cpu", "eager", {"conv": "auto", "attention": "auto", "graph": "auto"}
    )
    _require(isinstance(report, dict), "route_report must return dict")
    for key in (
        "boundary_switch_count",
        "boundary_copy_mode",
        "boundary_reason_code",
        "boundary_copy_bytes_estimate",
        "boundary_overhead_est_ns",
        "zero_copy_eligible",
    ):
        _require(key in report, f"route_report missing key: {key}")

    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks/python/engine_split_bench.py"
    with tempfile.TemporaryDirectory(prefix="lc_interop_boundary_") as td:
        out_dir = Path(td)
        cmd = [
            sys.executable,
            str(script),
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--trace-iters",
            "1",
            "--out-dir",
            str(out_dir),
            "--pure-csv",
            "pure.csv",
            "--pure-json",
            "pure.json",
            "--interop-csv",
            "interop.csv",
            "--interop-json",
            "interop.json",
            "--require-max-boundary-overhead-ms",
            "--max-boundary-overhead-ms",
            "5.0",
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        _require(proc.returncode == 0, f"engine_split boundary gate failed: {proc.stdout}\n{proc.stderr}")
        payload = json.loads((out_dir / "interop.json").read_text(encoding="utf-8"))
        rows = list(payload.get("rows", []))
        _require(bool(rows), "interop rows must exist")
        conv_rows = [r for r in rows if r.get("bench") == "conv_attention_torchstrong_nchw"]
        _require(bool(conv_rows), "conv_attention rows must exist in interop report")

    print("python interop boundary smoke: ok")


if __name__ == "__main__":
    main()

