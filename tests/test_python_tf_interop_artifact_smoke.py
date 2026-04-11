#!/usr/bin/env python3
"""Python smoke: TF interop benchmark artifact schema regression."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks/python/tf_interop_bench.py"
    _require(script.exists(), "tf_interop_bench.py must exist")

    with tempfile.TemporaryDirectory(prefix="lc_tf_interop_") as td:
        out_dir = Path(td)
        cmd = [
            sys.executable,
            str(script),
            "--device",
            "cpu",
            "--mode",
            "graph",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--force-graph-policy-fallback",
            "--require-reason-coverage",
            "--min-reason-coverage-pct",
            "100.0",
            "--out-dir",
            str(out_dir),
            "--csv",
            "tf_interop.csv",
            "--json",
            "tf_interop.json",
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        _require(proc.returncode == 0, f"tf_interop benchmark failed: {proc.stdout}\n{proc.stderr}")

        payload = json.loads((out_dir / "tf_interop.json").read_text(encoding="utf-8"))
        _require(str(payload.get("artifact_schema_version", "")) == "tf_interop_artifact_v1", "schema version mismatch")
        _require("rows" in payload and isinstance(payload["rows"], list) and payload["rows"], "rows must exist")
        row = dict(payload["rows"][0])
        for key in (
            "suite",
            "bench",
            "shape",
            "status",
            "lc_api_lightning_ms",
            "lc_api_tf_ms",
            "route_boundary_reason_code",
            "route_boundary_copy_mode",
            "route_boundary_overhead_est_ms",
            "tf_runtime",
            "tf_runtime_available",
        ):
            _require(key in row, f"missing row key: {key}")
        _require(str(row.get("route_boundary_reason_code", "")).strip() != "", "reason code should not be empty")

    print("python tf interop artifact smoke: ok")


if __name__ == "__main__":
    main()
