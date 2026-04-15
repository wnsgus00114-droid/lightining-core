#!/usr/bin/env python3
"""Check Phase F contract constants and workflow wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def _require_contains(text: str, needle: str, *, file_label: str) -> None:
    _require(needle in text, f"[phase_f_contract_sync] missing '{needle}' in {file_label}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_f_framework_contract.json"))
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
    contract_path = root / args.contract_json
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    constants = dict(contract.get("ci_constants", {}).get("phase_f_rc_audit", {}))
    required_constants = (
        "require_semver_sync",
        "require_deprecation_policy_section",
        "require_phase_f_docs_sync",
        "require_phase_e_exit_bundle",
        "require_test_matrix_sync",
    )
    for key in required_constants:
        _require(key in constants, f"[phase_f_contract_sync] missing ci constant '{key}' in {args.contract_json}")

    for wf in args.workflow:
        wf_path = root / wf
        text = wf_path.read_text(encoding="utf-8")
        label = str(wf)
        _require_contains(text, "phase_f_framework_contract.json", file_label=label)
        _require_contains(text, "phase_f_rc_audit", file_label=label)

    print(
        json.dumps(
            {
                "ok": True,
                "contract": str(contract_path.resolve()),
                "workflows_checked": [str(w) for w in args.workflow],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
