#!/usr/bin/env python3
"""Phase F RC audit gate for v0.6.0-rc0 entry."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_check(repo_root: Path, args: list[str]) -> tuple[bool, str]:
    cmd = ["python3", *args]
    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    ok = proc.returncode == 0
    detail = (proc.stdout or proc.stderr or "").strip()
    return ok, detail


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_f_framework_contract.json"))
    p.add_argument("--phase-e-audit-json", type=Path, default=Path("benchmarks/reports/ci/phase_e_exit_audit.json"))
    p.add_argument("--test-matrix-json", type=Path, default=Path("docs/test_matrix_contract.json"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--roadmap", type=Path, default=Path("ROADMAP.md"))
    p.add_argument("--expected-version", type=str, default="")
    p.add_argument("--out-json", type=Path, default=Path("benchmark_results/phase_f_rc_audit.json"))
    p.add_argument("--out-md", type=Path, default=Path("benchmark_results/phase_f_rc_audit.md"))
    p.add_argument("--require-pass", action="store_true")
    args = p.parse_args()

    root = args.repo_root.resolve()
    contract = _load_json(root / args.contract_json)
    constants = dict(contract.get("ci_constants", {}).get("phase_f_rc_audit", {}))

    phase_e_payload = _load_json(root / args.phase_e_audit_json)
    phase_e_pass = bool(phase_e_payload.get("overall_pass", phase_e_payload.get("pass", False)))

    semver_ok = True
    semver_detail = ""
    if bool(constants.get("require_semver_sync", True)):
        check_args = ["scripts/sync_release_metadata.py", "--check"]
        if str(args.expected_version).strip():
            check_args.extend(["--expected-version", str(args.expected_version).strip()])
        semver_ok, semver_detail = _run_check(root, check_args)

    phase_f_docs_ok = True
    phase_f_docs_detail = ""
    if bool(constants.get("require_phase_f_docs_sync", True)):
        phase_f_docs_ok, phase_f_docs_detail = _run_check(root, ["scripts/generate_phase_f_contract_docs.py", "--check"])

    matrix_ok = True
    matrix_detail = ""
    if bool(constants.get("require_test_matrix_sync", True)):
        matrix_ok, matrix_detail = _run_check(root, ["scripts/generate_test_matrix_docs.py", "--check"])

    readme_text = (root / args.readme).read_text(encoding="utf-8")
    roadmap_text = (root / args.roadmap).read_text(encoding="utf-8")
    deprecation_ok = (
        "deprecation" in readme_text.lower() or "deprecation" in roadmap_text.lower()
    ) if bool(constants.get("require_deprecation_policy_section", True)) else True

    matrix_contract = _load_json(root / args.test_matrix_json)
    matrix_constants = dict(matrix_contract.get("ci_constants", {}))
    tested_rows = json.loads((root / "docs/tested_environments.json").read_text(encoding="utf-8"))
    min_rows = int(matrix_constants.get("min_rows", 0))
    matrix_rows_ok = len(tested_rows) >= min_rows

    overall_pass = bool(
        phase_e_pass
        and semver_ok
        and phase_f_docs_ok
        and matrix_ok
        and deprecation_ok
        and matrix_rows_ok
    )

    out = {
        "schema_version": "phase_f_rc_audit_v1",
        "contract_version": str(contract.get("contract_version", "")),
        "phase_e_exit_pass": bool(phase_e_pass),
        "semver_sync_pass": bool(semver_ok),
        "phase_f_docs_sync_pass": bool(phase_f_docs_ok),
        "test_matrix_sync_pass": bool(matrix_ok),
        "deprecation_policy_present": bool(deprecation_ok),
        "tested_env_rows": int(len(tested_rows)),
        "tested_env_min_rows": int(min_rows),
        "tested_env_rows_pass": bool(matrix_rows_ok),
        "details": {
            "semver": semver_detail,
            "phase_f_docs": phase_f_docs_detail,
            "test_matrix": matrix_detail,
        },
        "overall_pass": bool(overall_pass),
    }

    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    out_md = root / args.out_md
    md_lines = [
        "### phase_f_rc_audit",
        "",
        f"- overall_pass: `{out['overall_pass']}`",
        f"- phase_e_exit_pass: `{out['phase_e_exit_pass']}`",
        f"- semver_sync_pass: `{out['semver_sync_pass']}`",
        f"- phase_f_docs_sync_pass: `{out['phase_f_docs_sync_pass']}`",
        f"- test_matrix_sync_pass: `{out['test_matrix_sync_pass']}`",
        f"- deprecation_policy_present: `{out['deprecation_policy_present']}`",
        f"- tested_env_rows: `{out['tested_env_rows']}` (min `{out['tested_env_min_rows']}`)",
    ]
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"overall_pass": overall_pass, "json": str(out_json), "md": str(out_md)}))
    if args.require_pass and not overall_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
