#!/usr/bin/env python3
"""Check Phase D contract constants are wired into CI/release workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _require_contains(text: str, needle: str, *, file_label: str) -> None:
    _require(needle in text, f"[phase_d_contract_sync] missing '{needle}' in {file_label}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_d_runner_contract.json"))
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
    constants = dict(contract.get("ci_constants", {}).get("phase_d_exit_audit", {}))
    required_constants = (
        "min_torch_reason_coverage_pct",
        "min_torch_budget_pass_rate_pct",
        "min_tf_reason_coverage_pct",
        "require_tf_both_runtime_paths",
        "max_runner_variance_cv_pct",
        "max_quickstart_python_lines",
        "require_inference_example",
        "require_training_example",
    )
    for key in required_constants:
        _require(key in constants, f"[phase_d_contract_sync] missing ci constant '{key}' in {args.contract_json}")

    for wf in args.workflow:
        wf_path = root / wf
        text = wf_path.read_text(encoding="utf-8")
        label = str(wf)
        _require_contains(text, "phase_d_exit_audit.py", file_label=label)
        _require_contains(text, "docs/phase_d_runner_contract.json", file_label=label)
        _require_contains(text, "LC_PHASE_D_CONTRACT_JSON", file_label=label)
        _require_contains(text, "runner_variance", file_label=label)
        _require_contains(text, "torch_runner_adapter", file_label=label)
        _require_contains(text, "tf_runner_adapter", file_label=label)

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
