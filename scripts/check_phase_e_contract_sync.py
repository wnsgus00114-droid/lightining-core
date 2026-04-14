#!/usr/bin/env python3
"""Check Phase E engine-federation contract constants are wired into workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _require_contains(text: str, needle: str, *, file_label: str) -> None:
    _require(needle in text, f"[phase_e_contract_sync] missing '{needle}' in {file_label}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract-json", type=Path, default=Path("docs/engine_federation_contract.json"))
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
    constants = dict(contract.get("ci_constants", {}).get("phase_e_exit_audit", {}))
    required_constants = (
        "min_torch_reason_coverage_pct",
        "min_tf_reason_coverage_pct",
        "min_coreml_reason_coverage_pct",
        "min_mlx_reason_coverage_pct",
        "min_federation_reason_coverage_pct",
        "max_boundary_overhead_ms",
        "require_import_export_matrix_sync",
        "require_roundtrip_artifacts",
        "min_perf_explain_coverage_pct",
    )
    for key in required_constants:
        _require(key in constants, f"[phase_e_contract_sync] missing ci constant '{key}' in {args.contract_json}")

    for wf in args.workflow:
        wf_path = root / wf
        text = wf_path.read_text(encoding="utf-8")
        label = str(wf)
        _require_contains(text, "engine_federation_contract.json", file_label=label)
        _require_contains(text, "LC_PHASE_E_CONTRACT_JSON", file_label=label)
        _require_contains(text, "phase_e_exit_audit.py", file_label=label)
        _require_contains(text, "coreml_runner_adapter", file_label=label)
        _require_contains(text, "mlx_runner_adapter", file_label=label)
        _require_contains(text, "interop_perf_gate_v2", file_label=label)

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
