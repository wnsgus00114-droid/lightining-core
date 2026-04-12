#!/usr/bin/env python3
"""Python smoke: Phase D exit audit bundle generation + hard gate pass."""

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
    torch_bench = repo_root / "benchmarks/python/torch_runner_adapter_bench.py"
    tf_bench = repo_root / "benchmarks/python/tf_runner_adapter_bench.py"
    var_gate = repo_root / "benchmarks/python/model_runner_variance_gate.py"
    audit = repo_root / "benchmarks/python/phase_d_exit_audit.py"
    for script in (torch_bench, tf_bench, var_gate, audit):
        _require(script.exists(), f"missing script: {script}")

    with tempfile.TemporaryDirectory(prefix="lc_phase_d_audit_") as td:
        out = Path(td)
        torch_json = out / "torch_runner_adapter.json"
        tf_json = out / "tf_runner_adapter.json"
        var_json = out / "runner_variance.json"
        var_md = out / "runner_variance.md"
        var_runs = out / "runner_variance_runs.csv"
        audit_json = out / "phase_d_exit_audit.json"
        audit_md = out / "phase_d_exit_audit.md"
        bundle_json = out / "phase_d_exit_candidate_bundle.json"

        _run(
            [
                sys.executable,
                str(torch_bench),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--allow-fake-torch",
                "--require-reason-coverage",
                "--min-reason-coverage-pct",
                "100.0",
                "--require-budget-gate",
                "--max-boundary-overhead-ms",
                "5.0",
                "--out-dir",
                str(out),
                "--json",
                torch_json.name,
                "--csv",
                "torch_runner_adapter.csv",
                "--md",
                "torch_runner_adapter.md",
            ],
            cwd=repo_root,
        )

        _run(
            [
                sys.executable,
                str(tf_bench),
                "--device",
                "cpu",
                "--warmup",
                "1",
                "--iters",
                "2",
                "--require-reason-coverage",
                "--min-reason-coverage-pct",
                "100.0",
                "--require-artifact-schema",
                "--out-dir",
                str(out),
                "--json",
                tf_json.name,
                "--csv",
                "tf_runner_adapter.csv",
                "--md",
                "tf_runner_adapter.md",
            ],
            cwd=repo_root,
        )

        _require(var_gate.exists(), f"missing script: {var_gate}")
        var_runs.write_text("run_idx,mode,input_kind,status,latency_ms\\n0,eager,token_ids,ok,1.0\\n", encoding="utf-8")
        var_md.write_text("## runner variance smoke\\n", encoding="utf-8")
        var_json.write_text(
            json.dumps(
                {
                    "generated_at_utc": "2026-04-11T00:00:00Z",
                    "gate": {
                        "overall_pass": True,
                        "threshold_cv_pct": 2.0,
                        "suite_total_trimmed_cv_pct": 1.0,
                    },
                    "case_count": 1,
                    "cases": [
                        {
                            "mode": "eager",
                            "input_kind": "token_ids",
                            "samples": 3,
                            "mean_ms": 1.0,
                            "stdev_ms": 0.01,
                            "cv_pct": 1.0,
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        _run(
            [
                sys.executable,
                str(audit),
                "--torch-adapter-json",
                str(torch_json),
                "--tf-runner-json",
                str(tf_json),
                "--runner-variance-json",
                str(var_json),
                "--contract-json",
                "docs/phase_d_runner_contract.json",
                "--quickstart",
                "docs/quickstart.md",
                "--out-json",
                str(audit_json),
                "--out-md",
                str(audit_md),
                "--bundle-json",
                str(bundle_json),
                "--require-artifacts",
                "--artifact",
                str(torch_json),
                "--artifact",
                str(tf_json),
                "--artifact",
                str(var_json),
                "--require-pass",
            ],
            cwd=repo_root,
        )

        payload = _load(audit_json)
        _require(bool(payload.get("overall_pass", False)), "phase d audit should pass")
        _require(bool(payload.get("tf_both_runtime_paths_ok", False)), "tf both-runtime gate should pass")
        _require(bool(payload.get("tf_artifact_schema_pass", False)), "tf artifact schema gate should pass")

    print("python phase d exit audit smoke: ok")


if __name__ == "__main__":
    main()
