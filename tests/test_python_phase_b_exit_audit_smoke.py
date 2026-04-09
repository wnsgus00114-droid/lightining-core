#!/usr/bin/env python3
"""Python smoke: Phase B exit audit script output contract."""

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
    script = repo_root / "benchmarks/python/phase_b_exit_audit.py"
    _require(script.exists(), "phase_b_exit_audit.py must exist")

    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)

        graph_json = td / "graph_eager_ab.json"
        contract_json = td / "phase_b_graph_contract.json"
        out_json = td / "phase_b_exit_audit.json"
        out_md = td / "phase_b_exit_audit.md"
        bundle_json = td / "phase_b_exit_candidate_bundle.json"

        readme = td / "README.md"
        roadmap = td / "ROADMAP.md"
        readme.write_text("Current public release: v0.2.7\n", encoding="utf-8")
        roadmap.write_text("Version context: v0.2.7\n", encoding="utf-8")

        graph_payload = {
            "backend_name": "cpu",
            "device": "cpu",
            "summary": {
                "total_cases": 3,
                "ok_cases": 3,
                "allclose_ok_cases": 3,
                "host_dispatch_reduction_rate_pct": 50.0,
                "chain_latency_gate_applicable": False,
                "representative_chain_latency_reduction_pct": -1.0,
            },
            "rows": [
                {"bench": "matmul_matrix_sub", "status": "ok", "graph_fallback_per_iter": 0.0, "graph_plan_fallback_groups": 0, "graph_plan_fallback_reason_codes": "none"},
                {"bench": "conv_attention_torchstrong_nchw", "status": "ok", "graph_fallback_per_iter": 0.0, "graph_plan_fallback_groups": 0, "graph_plan_fallback_reason_codes": "none"},
                {"bench": "matmul_bias_relu_fusion_path", "status": "ok", "graph_fallback_per_iter": 0.0, "graph_plan_fallback_groups": 0, "graph_plan_fallback_reason_codes": "none"},
            ],
        }
        graph_json.write_text(json.dumps(graph_payload, indent=2), encoding="utf-8")

        contract_payload = {
            "ci_constants": {
                "graph_eager_ab": {
                    "min_host_dispatch_reduction_rate_pct": 25.0,
                    "min_chained_latency_reduction_pct": 15.0,
                },
                "phase_b_exit_audit": {
                    "min_graph_pipeline_adoption_rate_pct": 100.0,
                    "allow_chain_latency_not_applicable": True,
                    "required_graph_benches": [
                        "matmul_matrix_sub",
                        "conv_attention_torchstrong_nchw",
                        "matmul_bias_relu_fusion_path",
                    ],
                },
            }
        }
        contract_json.write_text(json.dumps(contract_payload, indent=2), encoding="utf-8")

        dummy_artifact = td / "dummy.txt"
        dummy_artifact.write_text("ok\n", encoding="utf-8")

        cmd = [
            sys.executable,
            str(script),
            "--graph-json",
            str(graph_json),
            "--contract-json",
            str(contract_json),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--bundle-json",
            str(bundle_json),
            "--readme",
            str(readme),
            "--roadmap",
            str(roadmap),
            "--expected-version",
            "v0.2.7",
            "--require-doc-sync",
            "--require-artifacts",
            "--artifact",
            str(dummy_artifact),
            "--require-pass",
        ]

        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        _require(proc.returncode == 0, f"phase_b_exit_audit smoke failed: {proc.stdout}\n{proc.stderr}")

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        _require(bool(payload.get("phase_b_success_metrics_pass", False)), "phase B success metrics must pass")
        _require(bool(payload.get("overall_pass", False)), "overall_pass must be true in smoke")

        metrics = dict(payload.get("metrics", {}))
        adoption = dict(metrics.get("graph_pipeline_adoption_rate_pct", {}))
        _require(float(adoption.get("observed", 0.0)) >= 100.0, "graph pipeline adoption must be 100%")

    print("python phase_b_exit_audit smoke: ok")


if __name__ == "__main__":
    main()
