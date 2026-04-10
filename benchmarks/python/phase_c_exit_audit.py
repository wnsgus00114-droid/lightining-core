#!/usr/bin/env python3
"""Phase C exit audit and v0.3.0-rc0 lock helper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _pct(numer: int, denom: int) -> float:
    if denom <= 0:
        return float("nan")
    return (float(numer) / float(denom)) * 100.0


def _fmt_pct(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.2f}%"


def _fmt_ratio(v: float) -> str:
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.3f}x"


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
    entries: list[dict] = []
    missing: list[str] = []
    for p in paths:
        exists = p.exists()
        entries.append(
            {
                "path": str(p),
                "exists": exists,
                "size_bytes": p.stat().st_size if exists else 0,
                "sha256": _sha256(p) if exists else "",
            }
        )
        if not exists:
            missing.append(str(p))
    return entries, missing


def _metric(name: str, observed: float, target: float, *, greater_is_better: bool = True, applicable: bool = True) -> dict:
    if not applicable:
        return {"name": name, "observed": observed, "target": target, "applicable": False, "pass": True}
    if greater_is_better:
        ok = math.isfinite(observed) and observed >= target
    else:
        ok = math.isfinite(observed) and observed <= target
    return {"name": name, "observed": observed, "target": target, "applicable": True, "pass": bool(ok)}


def _render_md(payload: dict) -> str:
    m = payload["metrics"]
    lines = []
    lines.append("## Phase C Exit Audit")
    lines.append("")
    lines.append(f"- status: `{'pass' if payload['overall_pass'] else 'fail'}`")
    lines.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append("")
    lines.append("| Metric | Observed | Target | Pass |")
    lines.append("| --- | ---: | ---: | --- |")
    lines.append(
        f"| fusion_coverage_pct | {_fmt_pct(_as_float(m['fusion_coverage_pct']['observed']))} | {_fmt_pct(_as_float(m['fusion_coverage_pct']['target']))} | {m['fusion_coverage_pct']['pass']} |"
    )
    lines.append(
        f"| cost_explain_coverage_pct | {_fmt_pct(_as_float(m['cost_explain_coverage_pct']['observed']))} | {_fmt_pct(_as_float(m['cost_explain_coverage_pct']['target']))} | {m['cost_explain_coverage_pct']['pass']} |"
    )
    lines.append(
        f"| host_dispatch_reduction_rate_pct | {_fmt_pct(_as_float(m['host_dispatch_reduction_rate_pct']['observed']))} | {_fmt_pct(_as_float(m['host_dispatch_reduction_rate_pct']['target']))} | {m['host_dispatch_reduction_rate_pct']['pass']} |"
    )
    lines.append(
        f"| accuracy_consistency_pct | {_fmt_pct(_as_float(m['accuracy_consistency_pct']['observed']))} | {_fmt_pct(_as_float(m['accuracy_consistency_pct']['target']))} | {m['accuracy_consistency_pct']['pass']} |"
    )
    lines.append(
        f"| fallback_reason_coverage_pct | {_fmt_pct(_as_float(m['fallback_reason_coverage_pct']['observed']))} | {_fmt_pct(_as_float(m['fallback_reason_coverage_pct']['target']))} | {m['fallback_reason_coverage_pct']['pass']} |"
    )
    lines.append(
        f"| median_interop_over_pure | {_fmt_ratio(_as_float(m['median_interop_over_pure']['observed']))} | {_fmt_ratio(_as_float(m['median_interop_over_pure']['target']))} | {m['median_interop_over_pure']['pass']} |"
    )
    lines.append(
        f"| interop_boundary_reason_coverage_pct | {_fmt_pct(_as_float(m['interop_boundary_reason_coverage_pct']['observed']))} | {_fmt_pct(_as_float(m['interop_boundary_reason_coverage_pct']['target']))} | {m['interop_boundary_reason_coverage_pct']['pass']} |"
    )
    lines.append(
        f"| interop_boundary_max_overhead_ms | {_as_float(m['interop_boundary_max_overhead_ms']['observed']):.6f} | {_as_float(m['interop_boundary_max_overhead_ms']['target']):.6f} | {m['interop_boundary_max_overhead_ms']['pass']} |"
    )
    lines.append(
        f"| model_runner_mode_success_rate_pct | {_fmt_pct(_as_float(m['model_runner_mode_success_rate_pct']['observed']))} | {_fmt_pct(_as_float(m['model_runner_mode_success_rate_pct']['target']))} | {m['model_runner_mode_success_rate_pct']['pass']} |"
    )
    lines.append("")
    lines.append(f"- docs_sync: `{payload['docs_sync']['pass']}`")
    lines.append(f"- required_artifacts_missing: `{len(payload['artifact_bundle']['missing_required'])}`")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-json", type=Path, required=True)
    p.add_argument("--fusion-json", type=Path, required=True)
    p.add_argument("--engine-interop-json", type=Path, required=True)
    p.add_argument("--model-runner-json", type=Path, default=Path(""))
    p.add_argument("--cost-calibration-json", type=Path, default=Path(""))
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_c_engine_contract.json"))
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
    cc = dict(contract.get("ci_constants", {})).get("phase_c_exit_audit", {})
    cc = dict(cc) if isinstance(cc, dict) else {}

    fusion_payload = _load_json(args.fusion_json)
    graph_payload = _load_json(args.graph_json)
    interop_payload = _load_json(args.engine_interop_json)
    model_runner_payload = _load_json(args.model_runner_json) if args.model_runner_json and args.model_runner_json.exists() else {}
    cost_payload = _load_json(args.cost_calibration_json) if args.cost_calibration_json and args.cost_calibration_json.exists() else {}

    fusion_rows = list(fusion_payload.get("rows", []))
    eligible = [r for r in fusion_rows if str(r.get("bench", "")).endswith("_eligible") and str(r.get("status", "")).lower() == "ok"]
    applied = [r for r in eligible if bool(r.get("fusion_applied", False))]
    fusion_coverage_pct = _pct(len(applied), len(eligible))
    fusion_ok_rows = [r for r in fusion_rows if str(r.get("status", "")).lower() == "ok"]
    fusion_allclose_rows = [r for r in fusion_ok_rows if bool(r.get("allclose", False))]
    fusion_accuracy_pct = _pct(len(fusion_allclose_rows), len(fusion_ok_rows))

    explain_candidates = [r for r in fusion_rows if str(r.get("status", "")).lower() == "ok"]
    explained = [r for r in explain_candidates if str(r.get("fusion_reason", "")).strip().lower() not in {"", "n/a"}]
    cost_rows = list(cost_payload.get("rows", []))
    if cost_rows:
        explain_candidates.extend(cost_rows)
        explained.extend(
            [
                r
                for r in cost_rows
                if math.isfinite(_as_float(r.get("estimated_ns")))
                and math.isfinite(_as_float(r.get("measured_ns")))
                and str(r.get("status", "ok")) == "ok"
            ]
        )
    cost_explain_coverage_pct = _pct(len(explained), len(explain_candidates))

    graph_summary = dict(graph_payload.get("summary", {}))
    host_dispatch_reduction_rate_pct = _as_float(graph_summary.get("host_dispatch_reduction_rate_pct"), float("nan"))
    graph_rows = list(graph_payload.get("rows", []))
    graph_ok_rows = [r for r in graph_rows if str(r.get("status", "")).lower() == "ok"]
    graph_allclose_rows = [r for r in graph_ok_rows if bool(r.get("allclose", False))]
    graph_accuracy_pct = _pct(len(graph_allclose_rows), len(graph_ok_rows))

    fallback_rows = []
    for r in graph_rows:
        status = str(r.get("status", "")).lower()
        fallback_per_iter = _as_float(r.get("graph_fallback_per_iter"), 0.0)
        fallback_groups = int(_as_float(r.get("graph_plan_fallback_groups"), 0))
        is_fallback = (status != "ok") or (fallback_per_iter > 0.0) or (fallback_groups > 0)
        if is_fallback:
            fallback_rows.append(r)
    if fallback_rows:
        fallback_with_reason = [
            r
            for r in fallback_rows
            if str(r.get("graph_plan_fallback_reason_codes", "")).strip().lower() not in {"", "none", "n/a", "unsupported"}
        ]
        fallback_reason_coverage_pct = _pct(len(fallback_with_reason), len(fallback_rows))
    else:
        fallback_reason_coverage_pct = 100.0

    interop_rows = [r for r in list(interop_payload.get("rows", [])) if str(r.get("status", "")).lower() == "ok"]
    interop_ratios = [_as_float(r.get("interop_over_pure"), float("nan")) for r in interop_rows]
    interop_ratios = [v for v in interop_ratios if math.isfinite(v)]
    if interop_ratios:
        interop_ratios_sorted = sorted(interop_ratios)
        mid = len(interop_ratios_sorted) // 2
        if len(interop_ratios_sorted) % 2 == 1:
            median_interop = float(interop_ratios_sorted[mid])
        else:
            median_interop = float((interop_ratios_sorted[mid - 1] + interop_ratios_sorted[mid]) / 2.0)
    else:
        median_interop = float("nan")

    interop_boundary_rows = [
        r
        for r in interop_rows
        if str(r.get("bench", "")).strip().lower() == "conv_attention_torchstrong_nchw"
        and math.isfinite(_as_float(r.get("route_boundary_switch_count"), float("nan")))
    ]
    if interop_boundary_rows:
        interop_reason_rows = [
            r
            for r in interop_boundary_rows
            if str(r.get("route_boundary_reason_code", "")).strip().lower() not in {"", "n/a"}
        ]
        interop_boundary_reason_coverage_pct = _pct(len(interop_reason_rows), len(interop_boundary_rows))
        interop_boundary_max_overhead_ms = max(
            _as_float(r.get("route_boundary_overhead_est_ms"), float("nan")) for r in interop_boundary_rows
        )
    else:
        interop_boundary_reason_coverage_pct = 100.0
        interop_boundary_max_overhead_ms = 0.0

    runner_rows = list(model_runner_payload.get("rows", []))
    runner_total = len(runner_rows)
    runner_ok = len([r for r in runner_rows if str(r.get("status", "")).lower() == "ok"])
    runner_success_pct = _pct(runner_ok, runner_total)
    runner_allclose_rows = [r for r in runner_rows if bool(r.get("allclose_vs_eager", False))]
    runner_accuracy_pct = _pct(len(runner_allclose_rows), len(runner_rows))

    accuracy_values = [v for v in [fusion_accuracy_pct, graph_accuracy_pct, runner_accuracy_pct] if math.isfinite(v)]
    if accuracy_values:
        accuracy_consistency_pct = float(sum(accuracy_values) / len(accuracy_values))
    else:
        accuracy_consistency_pct = float("nan")

    min_fusion = _as_float(cc.get("min_fusion_coverage_pct"), 60.0)
    min_explain = _as_float(cc.get("min_cost_explain_coverage_pct"), 80.0)
    min_dispatch = _as_float(cc.get("min_host_dispatch_reduction_rate_pct"), 25.0)
    min_accuracy = _as_float(cc.get("min_accuracy_consistency_pct"), 90.0)
    min_fallback_reason = _as_float(cc.get("min_fallback_reason_coverage_pct"), 100.0)
    max_interop = _as_float(cc.get("max_median_interop_over_pure"), 1.35)
    min_interop_reason = _as_float(cc.get("min_interop_boundary_reason_coverage_pct"), 100.0)
    max_interop_boundary_overhead_ms = _as_float(cc.get("max_interop_boundary_overhead_ms"), 0.35)
    min_runner = _as_float(cc.get("min_model_runner_mode_success_rate_pct"), 66.0)

    metrics = {
        "fusion_coverage_pct": _metric("fusion_coverage_pct", fusion_coverage_pct, min_fusion, greater_is_better=True),
        "cost_explain_coverage_pct": _metric(
            "cost_explain_coverage_pct", cost_explain_coverage_pct, min_explain, greater_is_better=True
        ),
        "host_dispatch_reduction_rate_pct": _metric(
            "host_dispatch_reduction_rate_pct", host_dispatch_reduction_rate_pct, min_dispatch, greater_is_better=True
        ),
        "accuracy_consistency_pct": _metric(
            "accuracy_consistency_pct", accuracy_consistency_pct, min_accuracy, greater_is_better=True
        ),
        "fallback_reason_coverage_pct": _metric(
            "fallback_reason_coverage_pct", fallback_reason_coverage_pct, min_fallback_reason, greater_is_better=True
        ),
        "median_interop_over_pure": _metric(
            "median_interop_over_pure", median_interop, max_interop, greater_is_better=False
        ),
        "interop_boundary_reason_coverage_pct": _metric(
            "interop_boundary_reason_coverage_pct",
            interop_boundary_reason_coverage_pct,
            min_interop_reason,
            greater_is_better=True,
        ),
        "interop_boundary_max_overhead_ms": _metric(
            "interop_boundary_max_overhead_ms",
            interop_boundary_max_overhead_ms,
            max_interop_boundary_overhead_ms,
            greater_is_better=False,
        ),
        "model_runner_mode_success_rate_pct": _metric(
            "model_runner_mode_success_rate_pct",
            runner_success_pct,
            min_runner,
            greater_is_better=True,
            applicable=runner_total > 0,
        ),
    }

    expected_version = str(args.expected_version).strip()
    docs_sync = {
        "expected_version": expected_version,
        "readme_contains_expected_version": True,
        "roadmap_contains_expected_version": True,
        "pass": True,
    }
    if expected_version:
        readme_text = args.readme.read_text(encoding="utf-8") if args.readme.exists() else ""
        roadmap_text = args.roadmap.read_text(encoding="utf-8") if args.roadmap.exists() else ""
        docs_sync["readme_contains_expected_version"] = expected_version in readme_text
        docs_sync["roadmap_contains_expected_version"] = expected_version in roadmap_text
        docs_sync["pass"] = bool(docs_sync["readme_contains_expected_version"] and docs_sync["roadmap_contains_expected_version"])

    artifact_paths = [Path(x) for x in args.artifact]
    artifact_entries, missing_required = _artifact_entries(artifact_paths)
    artifact_bundle = {
        "required": bool(args.require_artifacts),
        "artifacts": artifact_entries,
        "missing_required": missing_required if args.require_artifacts else [],
        "pass": (len(missing_required) == 0) if args.require_artifacts else True,
    }

    core_metric_pass = all(bool(v.get("pass", False)) for v in metrics.values() if bool(v.get("applicable", True)))
    docs_pass = docs_sync["pass"] if args.require_doc_sync else True
    overall_pass = bool(core_metric_pass and docs_pass and artifact_bundle["pass"])

    payload = {
        "artifact_kind": "phase_c_exit_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_pass": overall_pass,
        "phase_c_success_metrics_pass": core_metric_pass,
        "metrics": metrics,
        "inputs": {
            "graph_json": str(args.graph_json),
            "fusion_json": str(args.fusion_json),
            "engine_interop_json": str(args.engine_interop_json),
            "model_runner_json": str(args.model_runner_json) if args.model_runner_json else "",
            "cost_calibration_json": str(args.cost_calibration_json) if args.cost_calibration_json else "",
            "contract_json": str(args.contract_json),
        },
        "docs_sync": docs_sync,
        "artifact_bundle": artifact_bundle,
    }

    bundle = {
        "bundle_kind": "phase_c_exit_candidate_bundle",
        "generated_at_utc": payload["generated_at_utc"],
        "overall_pass": overall_pass,
        "phase_c_success_metrics_pass": core_metric_pass,
        "metric_snapshot": {k: dict(v) for k, v in metrics.items()},
        "docs_sync": docs_sync,
        "artifact_bundle": artifact_bundle,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.bundle_json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_md(payload), encoding="utf-8")

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.bundle_json}")
    print(f"phase_c_success_metrics_pass={core_metric_pass} overall_pass={overall_pass}")

    if args.require_pass and not overall_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
