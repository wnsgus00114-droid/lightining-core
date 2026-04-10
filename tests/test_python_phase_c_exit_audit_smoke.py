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
        model_json = td_p / "model_runner.json"
        cost_json = td_p / "cost.json"
        contract_json = td_p / "phase_c_contract.json"
        out_json = td_p / "phase_c_exit_audit.json"
        out_md = td_p / "phase_c_exit_audit.md"
        bundle_json = td_p / "phase_c_exit_candidate_bundle.json"
        readme = td_p / "README.md"
        roadmap = td_p / "ROADMAP.md"

        readme.write_text("Current public release: v0.3.0-rc0\n", encoding="utf-8")
        roadmap.write_text("Version context: v0.3.0-rc0\n", encoding="utf-8")

        graph_json.write_text(
            json.dumps({"summary": {"host_dispatch_reduction_rate_pct": 50.0}, "rows": []}, indent=2),
            encoding="utf-8",
        )
        fusion_json.write_text(
            json.dumps(
                {
                    "rows": [
                        {"bench": "conv_relu_eligible", "status": "ok", "fusion_applied": True, "fusion_reason": "cost_model_accept"},
                        {"bench": "matmul_bias_relu_eligible", "status": "ok", "fusion_applied": True, "fusion_reason": "cost_model_accept"},
                    ]
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        interop_json.write_text(
            json.dumps({"rows": [{"bench": "attention", "status": "ok", "interop_over_pure": 1.05}]}, indent=2),
            encoding="utf-8",
        )
        model_json.write_text(
            json.dumps(
                {"rows": [{"mode": "eager", "status": "ok"}, {"mode": "graph", "status": "ok"}, {"mode": "interop", "status": "ok"}]},
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
                            "max_median_interop_over_pure": 1.30,
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
            "--model-runner-json",
            str(model_json),
            "--cost-calibration-json",
            str(cost_json),
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
            "v0.3.0-rc0",
            "--require-doc-sync",
            "--require-artifacts",
            "--artifact",
            str(dummy),
            "--require-pass",
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        _require(proc.returncode == 0, f"phase_c_exit_audit smoke failed: {proc.stdout}\n{proc.stderr}")
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        _require(bool(payload.get("phase_c_success_metrics_pass", False)), "phase C success metrics must pass")
        _require(bool(payload.get("overall_pass", False)), "overall_pass must be true")

    print("python phase_c_exit_audit smoke: ok")


if __name__ == "__main__":
    main()

