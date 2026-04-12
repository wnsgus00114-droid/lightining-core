#!/usr/bin/env python3
"""Python smoke: model-runner beta artifact schema + replay gate."""

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
    script = repo_root / "benchmarks/python/model_runner_alpha_bench.py"
    with tempfile.TemporaryDirectory(prefix="lc_runner_beta_") as td:
        out_dir = Path(td)
        csv_path = out_dir / "runner.csv"
        json_path = out_dir / "runner.json"
        md_path = out_dir / "runner.md"
        cmd = [
            sys.executable,
            str(script),
            "--device",
            "cpu",
            "--warmup",
            "1",
            "--iters",
            "2",
            "--replay-iters",
            "3",
            "--require-replay-match",
            "--validate-artifact-schema",
            "--csv",
            str(csv_path),
            "--json",
            str(json_path),
            "--md",
            str(md_path),
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        _require(proc.returncode == 0, f"model_runner bench failed:\n{proc.stdout}\n{proc.stderr}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        required_top = {
            "generated_at_utc",
            "suite",
            "artifact_schema_version",
            "artifact_schema",
            "backend_name",
            "device",
            "warmup",
            "iters",
            "replay_iters",
            "seed",
            "runner_config_schema",
            "runner_contract_manifest",
            "checkpoint_compatibility_matrix",
            "runner_checkpoint_compatibility_matrix",
            "rows",
        }
        for k in required_top:
            _require(k in payload, f"missing top-level key: {k}")

        rows = list(payload.get("rows", []))
        _require(bool(rows), "rows should not be empty")
        _require(str(payload.get("artifact_schema_version", "")) == "model_runner_artifact_v3", "artifact schema v3 required")
        _require(
            str(payload.get("runner_contract_manifest", {}).get("freeze_id", "")) == "v0.4.0-rc0",
            "runner contract freeze id mismatch",
        )
        row_fields = set(payload.get("artifact_schema", {}).get("row_fields", []))
        for i, row in enumerate(rows):
            missing = sorted([k for k in row_fields if k not in row])
            _require(not missing, f"row[{i}] missing fields: {missing}")
            if str(row.get("status", "")).lower() == "ok":
                _require(bool(row.get("replay_deterministic", False)), f"row[{i}] replay_deterministic must be true")
                _require(str(row.get("fallback_reason_code", "")) != "", f"row[{i}] fallback reason code missing")
                _require(str(row.get("runner_contract_schema_hash", "")) != "", f"row[{i}] runner contract hash missing")

        matrix_rows = list(payload.get("checkpoint_compatibility_matrix", {}).get("rows", []))
        _require(bool(matrix_rows), "checkpoint compatibility matrix rows must not be empty")
        runner_matrix_rows = list(payload.get("runner_checkpoint_compatibility_matrix", {}).get("rows", []))
        _require(bool(runner_matrix_rows), "runner checkpoint compatibility matrix rows must not be empty")

    print("python model runner beta artifact smoke: ok")


if __name__ == "__main__":
    main()
