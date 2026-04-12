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


def _manifest_sha256(entries: list[dict]) -> str:
    # Stable hash over path/exists/size/sha tuples for release-audit evidence lock.
    lines: list[str] = []
    for e in sorted(entries, key=lambda item: str(item.get("path", ""))):
        lines.append(
            "|".join(
                [
                    str(e.get("path", "")),
                    "1" if bool(e.get("exists", False)) else "0",
                    str(int(_as_float(e.get("size_bytes", 0), 0.0))),
                    str(e.get("sha256", "")),
                ]
            )
        )
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return float(vals[0])
    rank = max(0.0, min(1.0, q)) * float(len(vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(vals[lo])
    w = rank - float(lo)
    return float(vals[lo] * (1.0 - w) + vals[hi] * w)


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
    metric_order = [
        "fusion_coverage_pct",
        "cost_explain_coverage_pct",
        "fusion_reason_code_coverage_pct",
        "host_dispatch_reduction_rate_pct",
        "dispatch_overhead_p95_per_iter",
        "dispatch_overhead_p95_trend_nonincreasing_pct",
        "accuracy_consistency_pct",
        "fallback_reason_coverage_pct",
        "median_interop_over_pure",
        "interop_boundary_reason_coverage_pct",
        "interop_boundary_max_overhead_ms",
        "interop_boundary_max_upload_overhead_ms",
        "interop_boundary_max_engine_switch_overhead_ms",
        "interop_boundary_max_copy_overhead_ms",
        "interop_boundary_max_sync_overhead_ms",
        "zero_copy_fallback_reason_coverage_pct",
        "tf_boundary_reason_coverage_pct",
        "conv_e2e_improvement_pct",
        "attn_e2e_improvement_pct",
        "ffn_e2e_improvement_pct",
        "model_runner_mode_success_rate_pct",
    ]
    lines = []
    lines.append("## Phase C Exit Audit")
    lines.append("")
    lines.append(f"- status: `{'pass' if payload['overall_pass'] else 'fail'}`")
    lines.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append("")
    lines.append("| Metric | Observed | Target | Pass |")
    lines.append("| --- | ---: | ---: | --- |")
    for key in metric_order:
        if key not in m:
            continue
        item = dict(m[key])
        observed = _as_float(item.get("observed"))
        target = _as_float(item.get("target"))
        if key.endswith("_ms") or key.endswith("_per_iter"):
            obs_s = "n/a" if not math.isfinite(observed) else f"{observed:.6f}"
            tgt_s = "n/a" if not math.isfinite(target) else f"{target:.6f}"
        elif key.endswith("_over_pure"):
            obs_s = _fmt_ratio(observed)
            tgt_s = _fmt_ratio(target)
        else:
            obs_s = _fmt_pct(observed)
            tgt_s = _fmt_pct(target)
        lines.append(f"| {key} | {obs_s} | {tgt_s} | {item.get('pass', False)} |")
    lines.append("")
    lines.append(f"- docs_sync: `{payload['docs_sync']['pass']}`")
    lines.append(f"- required_artifacts_missing: `{len(payload['artifact_bundle']['missing_required'])}`")
    lines.append(f"- artifact_manifest_sha256: `{payload['artifact_bundle'].get('artifact_manifest_sha256', '')}`")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-json", type=Path, required=True)
    p.add_argument("--fusion-json", type=Path, required=True)
    p.add_argument("--engine-interop-json", type=Path, required=True)
    p.add_argument("--tf-interop-json", type=Path, default=Path(""))
    p.add_argument("--model-runner-json", type=Path, default=Path(""))
    p.add_argument("--cost-calibration-json", type=Path, default=Path(""))
    p.add_argument("--prior-audit-json", action="append", default=[])
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
    tf_interop_payload = _load_json(args.tf_interop_json) if args.tf_interop_json and args.tf_interop_json.exists() else {}
    model_runner_payload = _load_json(args.model_runner_json) if args.model_runner_json and args.model_runner_json.exists() else {}
    cost_payload = _load_json(args.cost_calibration_json) if args.cost_calibration_json and args.cost_calibration_json.exists() else {}
    prior_audits = [
        _load_json(Path(p))
        for p in args.prior_audit_json
        if str(p).strip() and Path(p).exists()
    ]

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

    fusion_reason_candidates = [r for r in fusion_rows if str(r.get("status", "")).lower() == "ok"]

    def _reason_present(token: object) -> bool:
        t = str(token).strip().lower()
        return t not in {"", "n/a", "none", "fusion_reason_missing", "fusion_disabled_reason_missing", "cost_model_reject_reason_missing"}

    fusion_reason_rows = [
        r
        for r in fusion_reason_candidates
        if _reason_present(r.get("fusion_reason_code", r.get("fusion_reason", "")))
        and _reason_present(
            r.get("fusion_disabled_reason_code", r.get("fusion_disabled_reason", "fusion_disabled_not_requested"))
        )
        and _reason_present(
            r.get("cost_model_reject_reason_code", r.get("cost_model_reject_reason", "cost_model_not_requested"))
        )
    ]
    fusion_reason_code_coverage_pct = _pct(len(fusion_reason_rows), len(fusion_reason_candidates))

    def _e2e_improvement_from_rows(rows: list[dict]) -> float:
        vals = []
        for r in rows:
            fused_ms = _as_float(r.get("fused_ms"), float("nan"))
            unfused_ms = _as_float(r.get("unfused_ms"), float("nan"))
            if not (math.isfinite(fused_ms) and math.isfinite(unfused_ms) and unfused_ms > 0.0):
                continue
            vals.append(((unfused_ms - fused_ms) / unfused_ms) * 100.0)
        if not vals:
            return float("nan")
        return float(sum(vals) / float(len(vals)))

    conv_rows = [r for r in fusion_reason_candidates if str(r.get("pattern", "")) == "conv_relu_v1"]
    attn_rows = [
        r
        for r in fusion_reason_candidates
        if str(r.get("pattern", "")) in {"attention_proj_v1", "attention_qkv_proj_v1"}
    ]
    conv_e2e_improvement_pct = _e2e_improvement_from_rows(conv_rows)
    attn_e2e_improvement_pct = _e2e_improvement_from_rows(attn_rows)

    graph_summary = dict(graph_payload.get("summary", {}))
    host_dispatch_reduction_rate_pct = _as_float(graph_summary.get("host_dispatch_reduction_rate_pct"), float("nan"))
    graph_rows = list(graph_payload.get("rows", []))
    dispatch_overhead_samples = [
        _as_float(r.get("dispatch_delta_per_iter"), float("nan"))
        for r in graph_rows
        if str(r.get("status", "")).lower() == "ok"
    ]
    dispatch_overhead_p95_per_iter = _percentile(dispatch_overhead_samples, 0.95)
    prior_dispatch_p95 = []
    for pa in prior_audits:
        prior_metrics = dict(pa.get("metrics", {}))
        if "dispatch_overhead_p95_per_iter" in prior_metrics:
            prior_dispatch_p95.append(_as_float(prior_metrics["dispatch_overhead_p95_per_iter"].get("observed"), float("nan")))
    dispatch_series = [v for v in (prior_dispatch_p95 + [dispatch_overhead_p95_per_iter]) if math.isfinite(v)]
    if len(dispatch_series) >= 2:
        trend_nonincreasing = all(dispatch_series[i] <= (dispatch_series[i - 1] + 1.0e-9) for i in range(1, len(dispatch_series)))
        dispatch_overhead_p95_trend_nonincreasing_pct = 100.0 if trend_nonincreasing else 0.0
    else:
        dispatch_overhead_p95_trend_nonincreasing_pct = float("nan")

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
        def _finite_or_zero(value: object) -> float:
            v = _as_float(value, 0.0)
            return v if math.isfinite(v) else 0.0

        interop_reason_rows = [
            r
            for r in interop_boundary_rows
            if str(r.get("route_boundary_reason_code", "")).strip().lower() not in {"", "n/a"}
        ]
        interop_boundary_reason_coverage_pct = _pct(len(interop_reason_rows), len(interop_boundary_rows))
        interop_boundary_max_overhead_ms = max(
            [_finite_or_zero(r.get("route_boundary_overhead_est_ms", 0.0)) for r in interop_boundary_rows] or [0.0]
        )
        interop_boundary_max_upload_overhead_ms = max(
            [_finite_or_zero(r.get("route_boundary_upload_overhead_est_ms", 0.0)) for r in interop_boundary_rows] or [0.0]
        )
        interop_boundary_max_engine_switch_overhead_ms = max(
            [_finite_or_zero(r.get("route_boundary_engine_switch_overhead_est_ms", 0.0)) for r in interop_boundary_rows] or [0.0]
        )
        interop_boundary_max_copy_overhead_ms = max(
            [_finite_or_zero(r.get("route_boundary_copy_overhead_est_ms", 0.0)) for r in interop_boundary_rows] or [0.0]
        )
        interop_boundary_max_sync_overhead_ms = max(
            [_finite_or_zero(r.get("route_boundary_sync_overhead_est_ms", 0.0)) for r in interop_boundary_rows] or [0.0]
        )
        zero_copy_fallback_rows = [
            r
            for r in interop_boundary_rows
            if bool(r.get("route_zero_copy_eligible", False))
            and str(r.get("route_boundary_copy_mode", "")).strip().lower() != "zero_copy"
        ]
        zero_copy_fallback_reason_rows = [
            r
            for r in zero_copy_fallback_rows
            if str(r.get("route_boundary_reason_code", "")).strip().lower() not in {"", "n/a", "none"}
        ]
        if zero_copy_fallback_rows:
            zero_copy_fallback_reason_coverage_pct = _pct(
                len(zero_copy_fallback_reason_rows), len(zero_copy_fallback_rows)
            )
        else:
            zero_copy_fallback_reason_coverage_pct = 100.0
    else:
        interop_boundary_reason_coverage_pct = 100.0
        interop_boundary_max_overhead_ms = 0.0
        interop_boundary_max_upload_overhead_ms = 0.0
        interop_boundary_max_engine_switch_overhead_ms = 0.0
        interop_boundary_max_copy_overhead_ms = 0.0
        interop_boundary_max_sync_overhead_ms = 0.0
        zero_copy_fallback_reason_coverage_pct = 100.0

    tf_rows = [r for r in list(tf_interop_payload.get("rows", [])) if str(r.get("status", "")).lower() == "ok"]
    tf_reason_rows = [
        r
        for r in tf_rows
        if str(r.get("route_boundary_reason_code", "")).strip().lower() not in {"", "n/a", "none"}
    ]
    tf_boundary_reason_coverage_pct = _pct(len(tf_reason_rows), len(tf_rows)) if tf_rows else float("nan")

    runner_rows = list(model_runner_payload.get("rows", []))
    runner_total = len(runner_rows)
    runner_ok = len([r for r in runner_rows if str(r.get("status", "")).lower() == "ok"])
    runner_success_pct = _pct(runner_ok, runner_total)
    runner_allclose_rows = [r for r in runner_rows if bool(r.get("allclose_vs_eager", False))]
    runner_accuracy_pct = _pct(len(runner_allclose_rows), len(runner_rows))
    eager_row = next((r for r in runner_rows if str(r.get("mode", "")) == "eager" and str(r.get("status", "")).lower() == "ok"), None)
    graph_row = next((r for r in runner_rows if str(r.get("mode", "")) == "graph" and str(r.get("status", "")).lower() == "ok"), None)
    ffn_e2e_improvement_pct = float("nan")
    if eager_row and graph_row:
        eager_ms = _as_float(eager_row.get("latency_ms"), float("nan"))
        graph_ms = _as_float(graph_row.get("latency_ms"), float("nan"))
        if math.isfinite(eager_ms) and math.isfinite(graph_ms) and eager_ms > 0.0:
            ffn_e2e_improvement_pct = ((eager_ms - graph_ms) / eager_ms) * 100.0

    accuracy_values = [v for v in [fusion_accuracy_pct, graph_accuracy_pct, runner_accuracy_pct] if math.isfinite(v)]
    if accuracy_values:
        accuracy_consistency_pct = float(sum(accuracy_values) / len(accuracy_values))
    else:
        accuracy_consistency_pct = float("nan")

    min_fusion = _as_float(cc.get("min_fusion_coverage_pct"), 60.0)
    min_explain = _as_float(cc.get("min_cost_explain_coverage_pct"), 80.0)
    min_fusion_reason_code = _as_float(cc.get("min_fusion_reason_code_coverage_pct"), 100.0)
    min_dispatch = _as_float(cc.get("min_host_dispatch_reduction_rate_pct"), 25.0)
    max_dispatch_overhead_p95_per_iter = _as_float(cc.get("max_dispatch_overhead_p95_per_iter"), float("nan"))
    require_dispatch_p95_trend = bool(cc.get("require_dispatch_overhead_p95_trend_nonincreasing", False))
    min_accuracy = _as_float(cc.get("min_accuracy_consistency_pct"), 90.0)
    min_fallback_reason = _as_float(cc.get("min_fallback_reason_coverage_pct"), 100.0)
    max_interop = _as_float(cc.get("max_median_interop_over_pure"), 1.35)
    min_interop_reason = _as_float(cc.get("min_interop_boundary_reason_coverage_pct"), 100.0)
    max_interop_boundary_overhead_ms = _as_float(cc.get("max_interop_boundary_overhead_ms"), 0.35)
    max_interop_boundary_upload_overhead_ms = _as_float(cc.get("max_interop_boundary_upload_overhead_ms"), 0.12)
    max_interop_boundary_engine_switch_overhead_ms = _as_float(
        cc.get("max_interop_boundary_engine_switch_overhead_ms"), 0.10
    )
    max_interop_boundary_copy_overhead_ms = _as_float(cc.get("max_interop_boundary_copy_overhead_ms"), 0.20)
    max_interop_boundary_sync_overhead_ms = _as_float(cc.get("max_interop_boundary_sync_overhead_ms"), 0.08)
    min_zero_copy_fallback_reason_coverage_pct = _as_float(
        cc.get("min_zero_copy_fallback_reason_coverage_pct"), 100.0
    )
    min_tf_boundary_reason_coverage_pct = _as_float(cc.get("min_tf_boundary_reason_coverage_pct"), 100.0)
    min_conv_e2e_improvement_pct = _as_float(cc.get("min_conv_e2e_improvement_pct"), 20.0)
    min_attn_e2e_improvement_pct = _as_float(cc.get("min_attn_e2e_improvement_pct"), 20.0)
    min_ffn_e2e_improvement_pct = _as_float(cc.get("min_ffn_e2e_improvement_pct"), 20.0)
    require_conv_e2e_gate = "min_conv_e2e_improvement_pct" in cc
    require_attn_e2e_gate = "min_attn_e2e_improvement_pct" in cc
    require_ffn_e2e_gate = "min_ffn_e2e_improvement_pct" in cc
    min_runner = _as_float(cc.get("min_model_runner_mode_success_rate_pct"), 66.0)

    metrics = {
        "fusion_coverage_pct": _metric("fusion_coverage_pct", fusion_coverage_pct, min_fusion, greater_is_better=True),
        "cost_explain_coverage_pct": _metric(
            "cost_explain_coverage_pct", cost_explain_coverage_pct, min_explain, greater_is_better=True
        ),
        "fusion_reason_code_coverage_pct": _metric(
            "fusion_reason_code_coverage_pct",
            fusion_reason_code_coverage_pct,
            min_fusion_reason_code,
            greater_is_better=True,
        ),
        "host_dispatch_reduction_rate_pct": _metric(
            "host_dispatch_reduction_rate_pct", host_dispatch_reduction_rate_pct, min_dispatch, greater_is_better=True
        ),
        "dispatch_overhead_p95_per_iter": _metric(
            "dispatch_overhead_p95_per_iter",
            dispatch_overhead_p95_per_iter,
            max_dispatch_overhead_p95_per_iter,
            greater_is_better=False,
            applicable=math.isfinite(max_dispatch_overhead_p95_per_iter),
        ),
        "dispatch_overhead_p95_trend_nonincreasing_pct": _metric(
            "dispatch_overhead_p95_trend_nonincreasing_pct",
            dispatch_overhead_p95_trend_nonincreasing_pct,
            100.0,
            greater_is_better=True,
            applicable=bool(require_dispatch_p95_trend and len(dispatch_series) >= 2),
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
        "interop_boundary_max_upload_overhead_ms": _metric(
            "interop_boundary_max_upload_overhead_ms",
            interop_boundary_max_upload_overhead_ms,
            max_interop_boundary_upload_overhead_ms,
            greater_is_better=False,
        ),
        "interop_boundary_max_engine_switch_overhead_ms": _metric(
            "interop_boundary_max_engine_switch_overhead_ms",
            interop_boundary_max_engine_switch_overhead_ms,
            max_interop_boundary_engine_switch_overhead_ms,
            greater_is_better=False,
        ),
        "interop_boundary_max_copy_overhead_ms": _metric(
            "interop_boundary_max_copy_overhead_ms",
            interop_boundary_max_copy_overhead_ms,
            max_interop_boundary_copy_overhead_ms,
            greater_is_better=False,
        ),
        "interop_boundary_max_sync_overhead_ms": _metric(
            "interop_boundary_max_sync_overhead_ms",
            interop_boundary_max_sync_overhead_ms,
            max_interop_boundary_sync_overhead_ms,
            greater_is_better=False,
        ),
        "zero_copy_fallback_reason_coverage_pct": _metric(
            "zero_copy_fallback_reason_coverage_pct",
            zero_copy_fallback_reason_coverage_pct,
            min_zero_copy_fallback_reason_coverage_pct,
            greater_is_better=True,
        ),
        "tf_boundary_reason_coverage_pct": _metric(
            "tf_boundary_reason_coverage_pct",
            tf_boundary_reason_coverage_pct,
            min_tf_boundary_reason_coverage_pct,
            greater_is_better=True,
            applicable=bool(tf_rows),
        ),
        "conv_e2e_improvement_pct": _metric(
            "conv_e2e_improvement_pct",
            conv_e2e_improvement_pct,
            min_conv_e2e_improvement_pct,
            greater_is_better=True,
            applicable=bool(require_conv_e2e_gate and conv_rows and math.isfinite(conv_e2e_improvement_pct)),
        ),
        "attn_e2e_improvement_pct": _metric(
            "attn_e2e_improvement_pct",
            attn_e2e_improvement_pct,
            min_attn_e2e_improvement_pct,
            greater_is_better=True,
            applicable=bool(require_attn_e2e_gate and attn_rows and math.isfinite(attn_e2e_improvement_pct)),
        ),
        "ffn_e2e_improvement_pct": _metric(
            "ffn_e2e_improvement_pct",
            ffn_e2e_improvement_pct,
            min_ffn_e2e_improvement_pct,
            greater_is_better=True,
            applicable=bool(require_ffn_e2e_gate and eager_row and graph_row and math.isfinite(ffn_e2e_improvement_pct)),
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
    artifact_manifest_sha256 = _manifest_sha256(artifact_entries)
    artifact_bundle = {
        "required": bool(args.require_artifacts),
        "artifacts": artifact_entries,
        "artifact_manifest_sha256": artifact_manifest_sha256,
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
            "tf_interop_json": str(args.tf_interop_json) if args.tf_interop_json else "",
            "model_runner_json": str(args.model_runner_json) if args.model_runner_json else "",
            "cost_calibration_json": str(args.cost_calibration_json) if args.cost_calibration_json else "",
            "contract_json": str(args.contract_json),
            "prior_audit_json": [str(x) for x in args.prior_audit_json],
        },
        "trend_context": {
            "dispatch_overhead_p95_series": dispatch_series,
            "dispatch_overhead_p95_series_count": len(dispatch_series),
            "dispatch_overhead_p95_trend_gate_requested": bool(require_dispatch_p95_trend),
            "dispatch_overhead_p95_trend_gate_applicable": bool(require_dispatch_p95_trend and len(dispatch_series) >= 2),
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
