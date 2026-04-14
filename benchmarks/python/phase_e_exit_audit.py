#!/usr/bin/env python3
"""Phase E exit audit for engine federation + bridge interoperability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be object: {path}")
    return payload


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_pct(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.2f}%"


def _fmt_ms(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.6f}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _artifact_entries(paths: list[Path]) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for p in paths:
        exists = p.exists()
        rows.append(
            {
                "path": str(p),
                "exists": bool(exists),
                "size_bytes": p.stat().st_size if exists else 0,
                "sha256": _sha256(p) if exists else "",
            }
        )
        if not exists:
            missing.append(str(p))
    return rows, missing


def _manifest_sha256(entries: list[dict]) -> str:
    lines: list[str] = []
    for e in sorted(entries, key=lambda x: str(x.get("path", ""))):
        lines.append(
            "|".join(
                [
                    str(e.get("path", "")),
                    "1" if bool(e.get("exists", False)) else "0",
                    str(int(_as_float(e.get("size_bytes", 0.0), 0.0))),
                    str(e.get("sha256", "")),
                ]
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _metric(name: str, observed: float, target: float, *, greater_is_better: bool = True, applicable: bool = True) -> dict:
    if not applicable:
        return {
            "name": name,
            "observed": observed,
            "target": target,
            "applicable": False,
            "pass": True,
        }
    if greater_is_better:
        ok = math.isfinite(observed) and observed >= target
    else:
        ok = math.isfinite(observed) and observed <= target
    return {
        "name": name,
        "observed": observed,
        "target": target,
        "applicable": True,
        "pass": bool(ok),
    }


def _render_md(payload: dict) -> str:
    metrics = dict(payload.get("metrics", {}))
    order = [
        "torch_reason_coverage_pct",
        "tf_reason_coverage_pct",
        "coreml_reason_coverage_pct",
        "mlx_reason_coverage_pct",
        "federation_reason_coverage_pct",
        "federation_max_boundary_overhead_ms",
        "perf_explain_coverage_pct",
    ]
    lines = [
        "## Phase E Exit Audit",
        "",
        f"- status: `{'pass' if payload.get('overall_pass', False) else 'fail'}`",
        f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`",
        "",
        "| Metric | Observed | Target | Pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for key in order:
        if key not in metrics:
            continue
        row = dict(metrics[key])
        observed = _as_float(row.get("observed"))
        target = _as_float(row.get("target"))
        if key.endswith("_ms"):
            obs_s = _fmt_ms(observed)
            tgt_s = _fmt_ms(target)
        else:
            obs_s = _fmt_pct(observed)
            tgt_s = _fmt_pct(target)
        lines.append(f"| {key} | {obs_s} | {tgt_s} | {bool(row.get('pass', False))} |")

    lines.append("")
    lines.append(f"- matrix_sync_pass: `{bool(payload.get('matrix_sync', {}).get('pass', False))}`")
    lines.append(f"- docs_sync_pass: `{bool(payload.get('docs_sync', {}).get('pass', False))}`")
    lines.append(f"- required_artifacts_missing: `{len(payload.get('artifact_bundle', {}).get('missing_required', []))}`")
    lines.append(f"- artifact_manifest_sha256: `{payload.get('artifact_bundle', {}).get('artifact_manifest_sha256', '')}`")
    return "\n".join(lines)


def _run_check_script(repo_root: Path, args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, *args], cwd=str(repo_root), capture_output=True, text=True)
    ok = proc.returncode == 0
    details = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return ok, details.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interop-gate-json", type=Path, required=True)
    p.add_argument("--coreml-adapter-json", type=Path, required=True)
    p.add_argument("--mlx-adapter-json", type=Path, required=True)
    p.add_argument("--matrix-json", type=Path, default=Path("docs/import_export_compatibility_matrix.json"))
    p.add_argument("--contract-json", type=Path, default=Path("docs/engine_federation_contract.json"))
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--bundle-json", type=Path, required=True)
    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--require-artifacts", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--roadmap", type=Path, default=Path("ROADMAP.md"))
    p.add_argument("--expected-version", type=str, default="")
    p.add_argument("--require-doc-sync", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    contract = _load_json(args.contract_json)
    constants = dict(contract.get("ci_constants", {}).get("phase_e_exit_audit", {}))

    interop_gate = _load_json(args.interop_gate_json)
    coreml_payload = _load_json(args.coreml_adapter_json)
    mlx_payload = _load_json(args.mlx_adapter_json)
    matrix_payload = _load_json(args.matrix_json) if args.matrix_json.exists() else {}

    bridge_rows = list(interop_gate.get("bridges", []))
    bridge_reason = {
        str(r.get("bridge", "")).strip().lower(): _as_float(r.get("reason_coverage_pct"), float("nan"))
        for r in bridge_rows
        if isinstance(r, dict)
    }

    torch_reason = _as_float(bridge_reason.get("torch"), float("nan"))
    tf_reason = _as_float(bridge_reason.get("tf"), float("nan"))
    coreml_reason = _as_float(coreml_payload.get("reason_coverage_pct"), float("nan"))
    mlx_reason = _as_float(mlx_payload.get("reason_coverage_pct"), float("nan"))
    federation_reason = _as_float(interop_gate.get("federation_reason_coverage_pct"), float("nan"))
    federation_overhead = _as_float(interop_gate.get("federation_max_boundary_overhead_ms"), float("nan"))
    perf_explain = _as_float(interop_gate.get("federation_explain_coverage_pct"), float("nan"))

    metrics = {
        "torch_reason_coverage_pct": _metric(
            "torch_reason_coverage_pct",
            torch_reason,
            float(constants.get("min_torch_reason_coverage_pct", 100.0)),
            greater_is_better=True,
            applicable=math.isfinite(torch_reason),
        ),
        "tf_reason_coverage_pct": _metric(
            "tf_reason_coverage_pct",
            tf_reason,
            float(constants.get("min_tf_reason_coverage_pct", 100.0)),
            greater_is_better=True,
            applicable=math.isfinite(tf_reason),
        ),
        "coreml_reason_coverage_pct": _metric(
            "coreml_reason_coverage_pct",
            coreml_reason,
            float(constants.get("min_coreml_reason_coverage_pct", 100.0)),
            greater_is_better=True,
        ),
        "mlx_reason_coverage_pct": _metric(
            "mlx_reason_coverage_pct",
            mlx_reason,
            float(constants.get("min_mlx_reason_coverage_pct", 100.0)),
            greater_is_better=True,
        ),
        "federation_reason_coverage_pct": _metric(
            "federation_reason_coverage_pct",
            federation_reason,
            float(constants.get("min_federation_reason_coverage_pct", 100.0)),
            greater_is_better=True,
        ),
        "federation_max_boundary_overhead_ms": _metric(
            "federation_max_boundary_overhead_ms",
            federation_overhead,
            float(constants.get("max_boundary_overhead_ms", 6.0)),
            greater_is_better=False,
        ),
        "perf_explain_coverage_pct": _metric(
            "perf_explain_coverage_pct",
            perf_explain,
            float(constants.get("min_perf_explain_coverage_pct", 100.0)),
            greater_is_better=True,
        ),
    }

    # Import/export matrix sanity + docs sync check
    matrix_rows = list(matrix_payload.get("rows", [])) if isinstance(matrix_payload, dict) else []
    matrix_has_required_bridges = {"lightning", "torch", "tensorflow", "coreml", "mlx"}.issubset(
        {str(r.get("bridge", "")).strip().lower() for r in matrix_rows if isinstance(r, dict)}
    )
    matrix_sync = {
        "pass": bool(matrix_has_required_bridges),
        "details": [],
    }
    if not matrix_has_required_bridges:
        matrix_sync["details"].append("import/export matrix missing one or more bridge rows")

    require_matrix_sync = bool(constants.get("require_import_export_matrix_sync", True))
    if require_matrix_sync:
        ok_phase_e_docs, detail_phase_e_docs = _run_check_script(
            repo_root,
            ["scripts/generate_phase_e_contract_docs.py", "--check"],
        )
        ok_matrix_docs, detail_matrix_docs = _run_check_script(
            repo_root,
            ["scripts/generate_import_export_matrix_docs.py", "--check"],
        )
        if not ok_phase_e_docs:
            matrix_sync["pass"] = False
            matrix_sync["details"].append("phase-e contract docs out of sync")
            if detail_phase_e_docs:
                matrix_sync["details"].append(detail_phase_e_docs)
        if not ok_matrix_docs:
            matrix_sync["pass"] = False
            matrix_sync["details"].append("import/export matrix docs out of sync")
            if detail_matrix_docs:
                matrix_sync["details"].append(detail_matrix_docs)

    docs_sync = {"pass": True, "expected_version": str(args.expected_version), "details": []}
    if str(args.expected_version).strip():
        readme_text = args.readme.read_text(encoding="utf-8") if args.readme.exists() else ""
        roadmap_text = args.roadmap.read_text(encoding="utf-8") if args.roadmap.exists() else ""
        if args.expected_version not in readme_text:
            docs_sync["pass"] = False
            docs_sync["details"].append(f"missing version in README: {args.expected_version}")
        if args.expected_version not in roadmap_text:
            docs_sync["pass"] = False
            docs_sync["details"].append(f"missing version in ROADMAP: {args.expected_version}")

    artifact_paths = [Path(p) for p in list(args.artifact)]
    entries, missing = _artifact_entries(artifact_paths)

    hard_flags = [bool(metrics[k].get("pass", False)) for k in metrics.keys()]
    if require_matrix_sync:
        hard_flags.append(bool(matrix_sync.get("pass", False)))
    if bool(constants.get("require_roundtrip_artifacts", True)) and args.require_artifacts:
        hard_flags.append(len(missing) == 0)
    if args.require_doc_sync:
        hard_flags.append(bool(docs_sync.get("pass", False)))

    overall_pass = all(hard_flags)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": str(contract.get("contract_version", "phase_e_v0.5.x_lock")),
        "metrics": metrics,
        "matrix_sync": matrix_sync,
        "docs_sync": docs_sync,
        "interop_gate": {
            "federation_reason_coverage_pct": federation_reason,
            "federation_explain_coverage_pct": perf_explain,
            "federation_max_boundary_overhead_ms": federation_overhead,
        },
        "artifact_bundle": {
            "required_artifacts": [str(p) for p in artifact_paths],
            "missing_required": missing,
            "entries": entries,
            "artifact_manifest_sha256": _manifest_sha256(entries),
        },
        "overall_pass": bool(overall_pass),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_md(payload), encoding="utf-8")
    args.bundle_json.write_text(
        json.dumps(
            {
                "generated_at_utc": payload["generated_at_utc"],
                "contract_version": payload["contract_version"],
                "overall_pass": payload["overall_pass"],
                "artifact_manifest_sha256": payload["artifact_bundle"]["artifact_manifest_sha256"],
                "artifact_entries": payload["artifact_bundle"]["entries"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"phase_e_exit_audit status={'pass' if overall_pass else 'fail'} "
        f"coreml_reason={coreml_reason:.2f}% mlx_reason={mlx_reason:.2f}% "
        f"fed_reason={federation_reason:.2f}% max_overhead_ms={federation_overhead:.6f}"
    )
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.bundle_json}")

    if args.require_pass and (not overall_pass):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
