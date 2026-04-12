#!/usr/bin/env python3
"""Check Phase C contract constants are wired into CI/release workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _require_contains(text: str, needle: str, *, file_label: str) -> None:
    _require(needle in text, f"[phase_c_contract_sync] missing '{needle}' in {file_label}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_c_engine_contract.json"))
    p.add_argument(
        "--workflow",
        action="append",
        default=[
            Path(".github/workflows/benchmark-artifacts.yml"),
            Path(".github/workflows/python-wheel-publish.yml"),
        ],
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / args.contract_json).read_text(encoding="utf-8"))
    constants = dict(contract.get("ci_constants", {}).get("phase_c_exit_audit", {}))
    required_constants = (
        "min_fusion_coverage_pct",
        "min_cost_explain_coverage_pct",
        "min_fusion_reason_code_coverage_pct",
        "min_host_dispatch_reduction_rate_pct",
        "max_dispatch_overhead_p95_per_iter",
        "require_dispatch_overhead_p95_trend_nonincreasing",
        "min_accuracy_consistency_pct",
        "min_fallback_reason_coverage_pct",
        "min_conv_e2e_improvement_pct",
        "min_attn_e2e_improvement_pct",
        "min_ffn_e2e_improvement_pct",
        "max_median_interop_over_pure",
        "min_model_runner_mode_success_rate_pct",
    )
    for key in required_constants:
        _require(key in constants, f"[phase_c_contract_sync] missing ci constant '{key}' in {args.contract_json}")

    for wf in args.workflow:
        wf_path = root / wf
        text = wf_path.read_text(encoding="utf-8")
        label = str(wf)
        _require_contains(text, "phase_c_exit_audit.py", file_label=label)
        _require_contains(text, "docs/phase_c_engine_contract.json", file_label=label)
        _require_contains(text, "LC_PHASE_C_CONTRACT_JSON", file_label=label)
        _require_contains(text, "phase_c_exit_audit.json", file_label=label)
        _require_contains(text, "phase_c_exit_candidate_bundle.json", file_label=label)
        _require_contains(text, "--require-artifacts", file_label=label)
        _require_contains(text, "--require-pass", file_label=label)
        if bool(constants.get("require_dispatch_overhead_p95_trend_nonincreasing", False)):
            _require_contains(text, "--prior-audit-json", file_label=label)

    print(
        json.dumps(
            {
                "ok": True,
                "contract": str((root / args.contract_json).resolve()),
                "workflows_checked": [str(w) for w in args.workflow],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
