#!/usr/bin/env python3
"""Phase D exit audit for runner/adapter/CLI readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _as_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_pct(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.2f}%"


def _fmt_ms(v: float) -> str:
    return "n/a" if not math.isfinite(v) else f"{v:.6f}"


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be object: {path}")
    return payload


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


def _extract_python_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    in_block = False
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not in_block and line.strip().startswith("```python"):
            in_block = True
            lines = []
            continue
        if in_block and line.strip().startswith("```"):
            in_block = False
            blocks.append(lines)
            lines = []
            continue
        if in_block:
            lines.append(line)
    return blocks


def _render_md(payload: dict) -> str:
    m = dict(payload.get("metrics", {}))
    keys = [
        "torch_reason_coverage_pct",
        "torch_budget_pass_rate_pct",
        "tf_reason_coverage_pct",
        "runner_variance_cv_pct",
        "quickstart_example_line_count",
    ]
    lines = [
        "## Phase D Exit Audit",
        "",
        f"- status: `{'pass' if payload.get('overall_pass', False) else 'fail'}`",
        f"- generated_at_utc: `{payload.get('generated_at_utc', '')}`",
        "",
        "| Metric | Observed | Target | Pass |",
        "| --- | ---: | ---: | --- |",
    ]
    for k in keys:
        if k not in m:
            continue
        row = dict(m[k])
        obs = _as_float(row.get("observed"))
        tgt = _as_float(row.get("target"))
        if k.endswith("line_count"):
            obs_s = "n/a" if not math.isfinite(obs) else f"{obs:.0f}"
            tgt_s = "n/a" if not math.isfinite(tgt) else f"{tgt:.0f}"
        elif k.endswith("cv_pct"):
            obs_s = _fmt_pct(obs)
            tgt_s = _fmt_pct(tgt)
        else:
            obs_s = _fmt_pct(obs)
            tgt_s = _fmt_pct(tgt)
        lines.append(f"| {k} | {obs_s} | {tgt_s} | {bool(row.get('pass', False))} |")

    lines.append("")
    lines.append(f"- tf_both_runtime_paths_ok: `{bool(payload.get('tf_both_runtime_paths_ok', False))}`")
    lines.append(f"- tf_artifact_schema_pass: `{bool(payload.get('tf_artifact_schema_pass', False))}`")
    lines.append(f"- quickstart_inference_example_present: `{bool(payload.get('quickstart_inference_example_present', False))}`")
    lines.append(f"- quickstart_training_example_present: `{bool(payload.get('quickstart_training_example_present', False))}`")
    lines.append(f"- docs_sync: `{bool(payload.get('docs_sync', {}).get('pass', True))}`")
    lines.append(f"- missing_artifacts: `{len(payload.get('artifact_bundle', {}).get('missing_required', []))}`")
    lines.append(f"- artifact_manifest_sha256: `{payload.get('artifact_bundle', {}).get('artifact_manifest_sha256', '')}`")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torch-adapter-json", type=Path, required=True)
    p.add_argument("--tf-runner-json", type=Path, required=True)
    p.add_argument("--runner-variance-json", type=Path, required=True)
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_d_runner_contract.json"))
    p.add_argument("--quickstart", type=Path, default=Path("docs/quickstart.md"))
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

    contract = _load_json(args.contract_json) if args.contract_json.exists() else {}
    constants = dict(contract.get("ci_constants", {}).get("phase_d_exit_audit", {}))

    torch_payload = _load_json(args.torch_adapter_json)
    tf_payload = _load_json(args.tf_runner_json)
    runner_var_payload = _load_json(args.runner_variance_json)

    torch_reason = _as_float(torch_payload.get("reason_coverage_pct"), float("nan"))
    torch_budget = _as_float(torch_payload.get("budget_pass_rate_pct"), float("nan"))
    tf_reason = _as_float(tf_payload.get("reason_coverage_pct"), float("nan"))
    runner_cv = _as_float(dict(runner_var_payload.get("gate", {})).get("suite_total_trimmed_cv_pct"), float("nan"))

    quickstart_text = args.quickstart.read_text(encoding="utf-8") if args.quickstart.exists() else ""
    py_blocks = _extract_python_blocks(quickstart_text)
    target_blocks = [b for b in py_blocks if any("TinyTransformerRunner" in line for line in b)]
    if not target_blocks:
        target_blocks = py_blocks
    quickstart_line_count = float(min((len([ln for ln in b if ln.strip()]) for b in target_blocks), default=float("nan")))

    quick_infer_present = ("run_tokens(" in quickstart_text) or ("runner.run(" in quickstart_text) or ("infer(" in quickstart_text)
    quick_train_present = ("autograd_train_loop(" in quickstart_text) or ("train_step(" in quickstart_text)

    tf_both_runtime_paths_ok = bool(tf_payload.get("both_runtime_paths_ok", False))
    tf_artifact_schema_pass = bool(tf_payload.get("artifact_schema_pass", False))

    metrics = {
        "torch_reason_coverage_pct": _metric(
            "torch_reason_coverage_pct",
            torch_reason,
            float(constants.get("min_torch_reason_coverage_pct", 100.0)),
            greater_is_better=True,
        ),
        "torch_budget_pass_rate_pct": _metric(
            "torch_budget_pass_rate_pct",
            torch_budget,
            float(constants.get("min_torch_budget_pass_rate_pct", 100.0)),
            greater_is_better=True,
        ),
        "tf_reason_coverage_pct": _metric(
            "tf_reason_coverage_pct",
            tf_reason,
            float(constants.get("min_tf_reason_coverage_pct", 100.0)),
            greater_is_better=True,
        ),
        "runner_variance_cv_pct": _metric(
            "runner_variance_cv_pct",
            runner_cv,
            float(constants.get("max_runner_variance_cv_pct", 2.0)),
            greater_is_better=False,
        ),
        "quickstart_example_line_count": _metric(
            "quickstart_example_line_count",
            quickstart_line_count,
            float(constants.get("max_quickstart_python_lines", 50.0)),
            greater_is_better=False,
            applicable=math.isfinite(quickstart_line_count),
        ),
    }

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

    hard_flags = [
        bool(metrics[k].get("pass", False)) for k in metrics.keys()
    ]
    hard_flags.append(tf_artifact_schema_pass)
    hard_flags.append(bool(tf_both_runtime_paths_ok) or (not bool(constants.get("require_tf_both_runtime_paths", True))))
    hard_flags.append(bool(quick_infer_present) or (not bool(constants.get("require_inference_example", True))))
    hard_flags.append(bool(quick_train_present) or (not bool(constants.get("require_training_example", True))))

    if args.require_artifacts:
        hard_flags.append(len(missing) == 0)
    if args.require_doc_sync:
        hard_flags.append(bool(docs_sync.get("pass", False)))

    overall_pass = all(hard_flags)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_version": str(contract.get("contract_version", "phase_d_v0.4.x")),
        "metrics": metrics,
        "tf_both_runtime_paths_ok": bool(tf_both_runtime_paths_ok),
        "tf_artifact_schema_pass": bool(tf_artifact_schema_pass),
        "quickstart_inference_example_present": bool(quick_infer_present),
        "quickstart_training_example_present": bool(quick_train_present),
        "docs_sync": docs_sync,
        "artifact_bundle": {
            "required": [str(p) for p in artifact_paths],
            "missing_required": missing,
            "entries": entries,
            "artifact_manifest_sha256": _manifest_sha256(entries),
        },
        "overall_pass": bool(overall_pass),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_md(payload), encoding="utf-8")

    bundle = {
        "phase": "phase_d_exit",
        "generated_at_utc": payload["generated_at_utc"],
        "overall_pass": bool(payload["overall_pass"]),
        "audit_json": str(args.out_json),
        "audit_md": str(args.out_md),
        "artifact_manifest_sha256": payload["artifact_bundle"]["artifact_manifest_sha256"],
        "artifacts": payload["artifact_bundle"]["entries"],
    }
    args.bundle_json.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.bundle_json}")

    if args.require_pass and (not bool(payload.get("overall_pass", False))):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
