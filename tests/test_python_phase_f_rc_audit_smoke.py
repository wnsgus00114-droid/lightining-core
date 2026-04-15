#!/usr/bin/env python3
"""Smoke test for phase_f_rc_audit script."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, check=False)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "benchmarks/python/phase_f_rc_audit.py"

    with tempfile.TemporaryDirectory(prefix="lc_phase_f_audit_") as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        phase_e_json = out_dir / "phase_e_exit_audit.json"
        out_json = out_dir / "phase_f_rc_audit.json"
        out_md = out_dir / "phase_f_rc_audit.md"
        phase_e_json.write_text(
            json.dumps({"overall_pass": True, "schema_version": "phase_e_exit_audit_v1"}, ensure_ascii=False),
            encoding="utf-8",
        )

        proc = _run(
            [
                "python3",
                str(script),
                "--repo-root",
                str(repo_root),
                "--phase-e-audit-json",
                str(phase_e_json),
                "--out-json",
                str(out_json),
                "--out-md",
                str(out_md),
            ],
            repo_root,
        )
        _require(proc.returncode == 0, f"phase_f_rc_audit failed: {proc.stdout}\n{proc.stderr}")
        _require(out_json.exists(), "phase_f_rc_audit json not created")
        _require(out_md.exists(), "phase_f_rc_audit md not created")

        payload = json.loads(out_json.read_text(encoding="utf-8"))
        _require(payload.get("schema_version") == "phase_f_rc_audit_v1", "unexpected schema version")
        _require("overall_pass" in payload, "overall_pass missing")

    print("python phase f rc audit smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
