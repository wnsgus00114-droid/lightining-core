#!/usr/bin/env python3
"""Python smoke: Phase E exit audit bundle generation + hard gate pass."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _run(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    _require(proc.returncode == 0, f"command failed: {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    engine_split = repo_root / "benchmarks/python/engine_split_bench.py"
    coreml_bench = repo_root / "benchmarks/python/coreml_runner_adapter_bench.py"
    mlx_bench = repo_root / "benchmarks/python/mlx_runner_adapter_bench.py"
    perf_gate = repo_root / "benchmarks/python/interop_perf_gate_v2.py"
    audit = repo_root / "benchmarks/python/phase_e_exit_audit.py"
    for script in (engine_split, coreml_bench, mlx_bench, perf_gate, audit):
        _require(script.exists(), f"missing script: {script}")

    with tempfile.TemporaryDirectory(prefix="lc_phase_e_audit_") as td:
        out = Path(td)
        engine_interop_json = out / "engine_split_interop.json"
        coreml_json = out / "coreml_runner_adapter.json"
        mlx_json = out / "mlx_runner_adapter.json"
        gate_json = out / "interop_perf_gate_v2.json"
        gate_md = out / "interop_perf_gate_v2.md"
        audit_json = out / "phase_e_exit_audit.json"
        audit_md = out / "phase_e_exit_audit.md"
        bundle_json = out / "phase_e_exit_candidate_bundle.json"

        _run(
            [
                sys.executable,
                str(engine_split),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--trace-iters",
                "1",
                "--out-dir",
                str(out),
                "--pure-csv",
                "engine_split_pure_lc.csv",
                "--pure-json",
                "engine_split_pure_lc.json",
                "--interop-csv",
                "engine_split_interop.csv",
                "--interop-json",
                engine_interop_json.name,
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(coreml_bench),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--out-dir",
                str(out),
                "--csv",
                "coreml_runner_adapter.csv",
                "--json",
                coreml_json.name,
                "--md",
                "coreml_runner_adapter.md",
                "--require-reason-coverage",
                "--min-reason-coverage-pct",
                "100.0",
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(mlx_bench),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--out-dir",
                str(out),
                "--csv",
                "mlx_runner_adapter.csv",
                "--json",
                mlx_json.name,
                "--md",
                "mlx_runner_adapter.md",
                "--require-reason-coverage",
                "--min-reason-coverage-pct",
                "100.0",
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(perf_gate),
                "--engine-interop-json",
                str(engine_interop_json),
                "--coreml-adapter-json",
                str(coreml_json),
                "--mlx-adapter-json",
                str(mlx_json),
                "--out-json",
                str(gate_json),
                "--out-md",
                str(gate_md),
                "--require-reason-coverage",
                "--require-perf-explain-coverage",
                "--require-max-boundary-overhead-ms",
                "--max-boundary-overhead-ms",
                "6.0",
                "--require-pass",
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(audit),
                "--interop-gate-json",
                str(gate_json),
                "--coreml-adapter-json",
                str(coreml_json),
                "--mlx-adapter-json",
                str(mlx_json),
                "--contract-json",
                "docs/engine_federation_contract.json",
                "--matrix-json",
                "docs/import_export_compatibility_matrix.json",
                "--out-json",
                str(audit_json),
                "--out-md",
                str(audit_md),
                "--bundle-json",
                str(bundle_json),
                "--require-artifacts",
                "--artifact",
                str(gate_json),
                "--artifact",
                str(coreml_json),
                "--artifact",
                str(mlx_json),
                "--artifact",
                "docs/import_export_compatibility_matrix.json",
                "--require-pass",
            ],
            cwd=repo_root,
        )

        payload = _load(audit_json)
        _require(bool(payload.get("overall_pass", False)), "phase e audit should pass")
        metrics = dict(payload.get("metrics", {}))
        _require(bool(metrics.get("coreml_reason_coverage_pct", {}).get("pass", False)), "coreml coverage must pass")
        _require(bool(metrics.get("mlx_reason_coverage_pct", {}).get("pass", False)), "mlx coverage must pass")

    print("python phase e exit audit smoke: ok")


if __name__ == "__main__":
    main()
