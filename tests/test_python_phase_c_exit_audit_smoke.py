#!/usr/bin/env python3
"""Python smoke: Phase C exit audit script output contract."""

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
    script = repo_root / "benchmarks/python/phase_c_exit_audit.py"
    _require(script.exists(), "phase_c_exit_audit.py must exist")

    with tempfile.TemporaryDirectory(prefix="lc_phase_c_") as td:
        td_p = Path(td)
        graph_json = td_p / "graph.json"
        fusion_json = td_p / "fusion.json"
        interop_json = td_p / "interop.json"
        tf_interop_json = td_p / "tf_interop.json"
        model_json = td_p / "model_runner.json"
        cost_json = td_p / "cost.json"
        contract_json = td_p / "phase_c_contract.json"
        out_json = td_p / "phase_c_exit_audit.json"
        out_md = td_p / "phase_c_exit_audit.md"
        bundle_json = td_p / "phase_c_exit_candidate_bundle.json"
        prior_audit_json = td_p / "phase_c_exit_audit_prior.json"
        readme = td_p / "README.md"
        roadmap = td_p / "ROADMAP.md"

        readme.write_text("Current public release: v0.3.0-rc0\n", encoding="utf-8")
        roadmap.write_text("Version context: v0.3.0-rc0\n", encoding="utf-8")

        graph_json.write_text(
            json.dumps(
                {
                    "summary": {"host_dispatch_reduction_rate_pct": 50.0},
                    "rows": [
                        {
                            "bench": "matmul_matrix_sub",
                            "status": "ok",
                            "allclose": True,
                            "dispatch_delta_per_iter": -0.25,
                            "graph_fallback_per_iter": 0.0,
                            "graph_plan_fallback_groups": 0,
                            "graph_plan_fallback_reason_codes": "none",
                        },
                        {
                            "bench": "conv_chain",
                            "status": "ok",
                            "allclose": True,
                            "dispatch_delta_per_iter": 0.0,
                            "graph_fallback_per_iter": 0.0,
                            "graph_plan_fallback_groups": 0,
                            "graph_plan_fallback_reason_codes": "none",
                        }
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        fusion_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "bench": "conv_relu_eligible",
                            "status": "ok",
                            "pattern": "conv_relu_v1",
                            "fusion_applied": True,
                            "fusion_reason": "cost_model_accept",
                            "fusion_reason_code": "cost_model_accept",
                            "fusion_disabled_reason_code": "fusion_disabled_not_requested",
                            "cost_model_reject_reason_code": "cost_model_not_requested",
                            "fused_ms": 0.80,
                            "unfused_ms": 1.00,
                            "allclose": True,
                        },
                        {
                            "bench": "attention_proj_eligible",
                            "status": "ok",
                            "pattern": "attention_proj_v1",
                            "fusion_applied": True,
                            "fusion_reason": "cost_model_accept",
                            "fusion_reason_code": "cost_model_accept",
                            "fusion_disabled_reason_code": "fusion_disabled_not_requested",
                            "cost_model_reject_reason_code": "cost_model_not_requested",
                            "fused_ms": 0.78,
                            "unfused_ms": 1.00,
                            "allclose": True,
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        interop_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {"bench": "attention", "status": "ok", "interop_over_pure": 1.05},
                        {
                            "bench": "conv_attention_torchstrong_nchw",
                            "status": "ok",
                            "interop_over_pure": 1.08,
                            "route_boundary_switch_count": 1,
                            "route_boundary_reason_code": "interop_engine_boundary_switch",
                            "route_boundary_overhead_est_ms": 0.10,
                        },
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        tf_interop_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "bench": "tiny_transformer_runner_tf_bridge",
                            "status": "ok",
                            "route_boundary_reason_code": "tf_runner_graph_policy_forced_eager",
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        model_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {"mode": "eager", "status": "ok", "allclose_vs_eager": True, "latency_ms": 1.00},
                        {"mode": "graph", "status": "ok", "allclose_vs_eager": True, "latency_ms": 0.95},
                        {"mode": "interop", "status": "ok", "allclose_vs_eager": True, "latency_ms": 1.07},
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        prior_audit_json.write_text(
            json.dumps(
                {
                    "metrics": {
                        "dispatch_overhead_p95_per_iter": {"observed": 0.05}
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        cost_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {
                            "case": "matmul_bias_relu",
                            "status": "ok",
                            "estimated_ns": 100000.0,
                            "measured_ns": 120000.0,
                        }
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        contract_json.write_text(
            json.dumps(
                {
                    "ci_constants": {
                        "phase_c_exit_audit": {
                            "min_fusion_coverage_pct": 50.0,
                            "min_cost_explain_coverage_pct": 80.0,
                            "min_host_dispatch_reduction_rate_pct": 25.0,
                            "max_dispatch_overhead_p95_per_iter": 0.0,
                            "require_dispatch_overhead_p95_trend_nonincreasing": True,
                            "min_accuracy_consistency_pct": 80.0,
                            "min_fallback_reason_coverage_pct": 100.0,
                            "min_conv_e2e_improvement_pct": 0.0,
                            "min_attn_e2e_improvement_pct": 0.0,
                            "min_ffn_e2e_improvement_pct": 0.0,
                            "max_median_interop_over_pure": 1.30,
                            "min_interop_boundary_reason_coverage_pct": 100.0,
                            "max_interop_boundary_overhead_ms": 0.35,
                            "min_model_runner_mode_success_rate_pct": 66.0,
                        }
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        dummy = td_p / "dummy.txt"
        dummy.write_text("ok\n", encoding="utf-8")

        cmd = [
            sys.executable,
            str(script),
            "--graph-json",
            str(graph_json),
            "--fusion-json",
            str(fusion_json),
            "--engine-interop-json",
            str(interop_json),
            "--tf-interop-json",
            str(tf_interop_json),
            "--model-runner-json",
            str(model_json),
            "--cost-calibration-json",
            str(cost_json),
            "--contract-json",
            str(contract_json),
            "--prior-audit-json",
            str(prior_audit_json),
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
            "v0.3.0-rc0",
            "--require-doc-sync",
            "--require-artifacts",
            "--artifact",
            str(dummy),
            "--artifact",
            str(tf_interop_json),
            "--require-pass",
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        _require(proc.returncode == 0, f"phase_c_exit_audit smoke failed: {proc.stdout}\n{proc.stderr}")
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        _require(bool(payload.get("phase_c_success_metrics_pass", False)), "phase C success metrics must pass")
        _require(bool(payload.get("overall_pass", False)), "overall_pass must be true")
        _require(str(payload.get("artifact_bundle", {}).get("artifact_manifest_sha256", "")) != "", "manifest hash missing")

    print("python phase_c_exit_audit smoke: ok")


if __name__ == "__main__":
    main()
