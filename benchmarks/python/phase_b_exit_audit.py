#!/usr/bin/env python3
"""Phase B exit audit and release-candidate evidence bundle generator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _pct(numer: float, denom: float) -> float:
    if denom <= 0:
        return float("nan")
    return (float(numer) / float(denom)) * 100.0


def _fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}%"


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


def _graph_metrics(graph_payload: dict, required_benches: list[str]) -> dict:
    summary = dict(graph_payload.get("summary", {}))
    rows = list(graph_payload.get("rows", []))

    total_cases = _as_int(summary.get("total_cases"), len(rows))
    ok_cases = _as_int(summary.get("ok_cases"), 0)
    allclose_ok_cases = _as_int(summary.get("allclose_ok_cases"), 0)

    dispatch_reduction_rate_pct = _as_float(summary.get("host_dispatch_reduction_rate_pct"))
    chained_latency_reduction_pct = _as_float(summary.get("representative_chain_latency_reduction_pct"))
    chain_gate_applicable = _as_bool(summary.get("chain_latency_gate_applicable", False))

    present_benches = {str(row.get("bench", "")) for row in rows}
    adopted = [bench for bench in required_benches if bench in present_benches]
    graph_pipeline_adoption_rate_pct = _pct(len(adopted), len(required_benches))

    fallback_rows = []
    fallback_reason_rows = 0
    for row in rows:
        status = str(row.get("status", "")).lower()
        graph_fallback_per_iter = _as_float(row.get("graph_fallback_per_iter"), 0.0)
        plan_fallback_groups = _as_int(row.get("graph_plan_fallback_groups"), 0)
        is_fallback_case = (status != "ok") or (graph_fallback_per_iter > 0.0) or (plan_fallback_groups > 0)
        if not is_fallback_case:
            continue
        fallback_rows.append(row)
        reason = str(row.get("graph_plan_fallback_reason_codes", "")).strip().lower()
        if reason and reason not in {"none", "n/a"}:
            fallback_reason_rows += 1

    fallback_rows_count = len(fallback_rows)
    fallback_reason_coverage_pct = 100.0 if fallback_rows_count == 0 else _pct(fallback_reason_rows, fallback_rows_count)

    return {
        "total_cases": total_cases,
        "ok_cases": ok_cases,
        "allclose_ok_cases": allclose_ok_cases,
        "allclose_rate_pct": _pct(allclose_ok_cases, ok_cases) if ok_cases > 0 else float("nan"),
        "dispatch_reduction_rate_pct": dispatch_reduction_rate_pct,
        "chained_latency_reduction_pct": chained_latency_reduction_pct,
        "chain_gate_applicable": chain_gate_applicable,
        "required_benches": required_benches,
        "adopted_benches": adopted,
        "missing_benches": [bench for bench in required_benches if bench not in present_benches],
        "graph_pipeline_adoption_rate_pct": graph_pipeline_adoption_rate_pct,
        "fallback_rows": fallback_rows_count,
        "fallback_reason_coverage_pct": fallback_reason_coverage_pct,
    }


def _artifact_entries(paths: list[Path]) -> tuple[list[dict], list[str]]:
    entries: list[dict] = []
    missing: list[str] = []
    for p in paths:
        exists = p.exists()
        entry = {
            "path": str(p),
            "exists": exists,
            "size_bytes": p.stat().st_size if exists else 0,
            "sha256": _sha256(p) if exists else "",
        }
        entries.append(entry)
        if not exists:
            missing.append(str(p))
    return entries, missing


def _metric_result(name: str, observed: float, *, target_min: float, applicable: bool = True) -> dict:
    passed = applicable and math.isfinite(observed) and observed >= target_min
    if not applicable:
        passed = True
    return {
        "name": name,
        "observed": observed,
        "target_min": target_min,
        "applicable": applicable,
        "pass": passed,
    }


def _render_md(payload: dict) -> str:
    m = payload["metrics"]
    docs = payload["docs_sync"]
    art = payload["artifact_bundle"]

    lines = []
    lines.append("## Phase B Exit Audit")
    lines.append("")
    lines.append(f"- status: `{'pass' if payload['overall_pass'] else 'fail'}`")
    lines.append(f"- generated_at_utc: `{payload['generated_at_utc']}`")
    lines.append(f"- suite source: `{payload['inputs']['graph_json']}`")
    lines.append("")
    lines.append("### Success Metrics (ROADMAP 11.2)")
    lines.append("")
    lines.append("| Metric | Observed | Target | Applicable | Pass |")
    lines.append("| --- | ---: | ---: | --- | --- |")
    for key in [
        "host_dispatch_reduction_rate_pct",
        "chained_latency_reduction_pct",
        "graph_pipeline_adoption_rate_pct",
    ]:
        row = m[key]
        lines.append(
            f"| {row['name']} | {_fmt_pct(_as_float(row['observed']))} | {_fmt_pct(_as_float(row['target_min']))} | {row['applicable']} | {row['pass']} |"
        )

    lines.append("")
    lines.append("### Additional Audit Signals")
    lines.append("")
    lines.append(f"- allclose_rate_ok_only: {_fmt_pct(_as_float(m['allclose_rate_pct']['observed']))}")
    lines.append(f"- fallback_reason_coverage: {_fmt_pct(_as_float(m['fallback_reason_coverage_pct']['observed']))}")
    lines.append(f"- docs_sync(readme+roadmap expected version): `{docs['pass']}`")
    lines.append(f"- missing_required_artifacts: `{len(art['missing_required'])}`")
    lines.append("")
    if art["missing_required"]:
        lines.append("### Missing Required Artifacts")
        lines.append("")
        for item in art["missing_required"]:
            lines.append(f"- `{item}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-json", type=Path, required=True)
    p.add_argument("--contract-json", type=Path, default=Path("docs/phase_b_graph_contract.json"))
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--bundle-json", type=Path, required=True)

    p.add_argument("--required-graph-bench", action="append", default=[])

    p.add_argument("--target-host-dispatch-reduction-rate-pct", type=float, default=float("nan"))
    p.add_argument("--target-chained-latency-reduction-pct", type=float, default=float("nan"))
    p.add_argument("--target-graph-pipeline-adoption-rate-pct", type=float, default=float("nan"))
    p.add_argument("--allow-chain-latency-not-applicable", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--require-artifacts", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--roadmap", type=Path, default=Path("ROADMAP.md"))
    p.add_argument("--expected-version", type=str, default="")
    p.add_argument("--require-doc-sync", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--min-allclose-rate-pct", type=float, default=float("nan"))
    p.add_argument("--min-fallback-reason-coverage-pct", type=float, default=float("nan"))
    p.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=False)

    args = p.parse_args()

    contract = _load_json(args.contract_json)
    graph_payload = _load_json(args.graph_json)

    graph_ab_constants = dict(dict(contract.get("ci_constants", {})).get("graph_eager_ab", {}))
    exit_constants = dict(dict(contract.get("ci_constants", {})).get("phase_b_exit_audit", {}))

    target_host = args.target_host_dispatch_reduction_rate_pct
    if not math.isfinite(target_host):
        target_host = _as_float(graph_ab_constants.get("min_host_dispatch_reduction_rate_pct"), 25.0)

    target_chain = args.target_chained_latency_reduction_pct
    if not math.isfinite(target_chain):
        target_chain = _as_float(graph_ab_constants.get("min_chained_latency_reduction_pct"), 15.0)

    target_adoption = args.target_graph_pipeline_adoption_rate_pct
    if not math.isfinite(target_adoption):
        target_adoption = _as_float(exit_constants.get("min_graph_pipeline_adoption_rate_pct"), 100.0)

    allow_chain_na = args.allow_chain_latency_not_applicable
    if "allow_chain_latency_not_applicable" in exit_constants:
        allow_chain_na = _as_bool(exit_constants.get("allow_chain_latency_not_applicable"))

    required_benches = list(args.required_graph_bench)
    if not required_benches:
        required_benches = list(exit_constants.get("required_graph_benches", []))
    if not required_benches:
        required_benches = [
            "matmul_matrix_sub",
            "conv_attention_torchstrong_nchw",
            "matmul_bias_relu_fusion_path",
        ]

    graph = _graph_metrics(graph_payload, required_benches)

    chain_applicable = bool(graph["chain_gate_applicable"])
    if (not chain_applicable) and allow_chain_na:
        chain_metric = _metric_result(
            "chained_latency_reduction_pct",
            _as_float(graph["chained_latency_reduction_pct"]),
            target_min=target_chain,
            applicable=False,
        )
    else:
        chain_metric = _metric_result(
            "chained_latency_reduction_pct",
            _as_float(graph["chained_latency_reduction_pct"]),
            target_min=target_chain,
            applicable=True,
        )

    metrics = {
        "host_dispatch_reduction_rate_pct": _metric_result(
            "host_dispatch_reduction_rate_pct",
            _as_float(graph["dispatch_reduction_rate_pct"]),
            target_min=target_host,
            applicable=True,
        ),
        "chained_latency_reduction_pct": chain_metric,
        "graph_pipeline_adoption_rate_pct": _metric_result(
            "graph_pipeline_adoption_rate_pct",
            _as_float(graph["graph_pipeline_adoption_rate_pct"]),
            target_min=target_adoption,
            applicable=True,
        ),
        "allclose_rate_pct": {
            "name": "allclose_rate_pct",
            "observed": _as_float(graph["allclose_rate_pct"]),
            "target_min": args.min_allclose_rate_pct,
            "applicable": math.isfinite(args.min_allclose_rate_pct),
            "pass": (not math.isfinite(args.min_allclose_rate_pct))
            or (
                math.isfinite(_as_float(graph["allclose_rate_pct"]))
                and _as_float(graph["allclose_rate_pct"]) >= args.min_allclose_rate_pct
            ),
        },
        "fallback_reason_coverage_pct": {
            "name": "fallback_reason_coverage_pct",
            "observed": _as_float(graph["fallback_reason_coverage_pct"]),
            "target_min": args.min_fallback_reason_coverage_pct,
            "applicable": math.isfinite(args.min_fallback_reason_coverage_pct),
            "pass": (not math.isfinite(args.min_fallback_reason_coverage_pct))
            or (
                math.isfinite(_as_float(graph["fallback_reason_coverage_pct"]))
                and _as_float(graph["fallback_reason_coverage_pct"]) >= args.min_fallback_reason_coverage_pct
            ),
        },
    }

    expected_version = args.expected_version.strip()
    docs_sync = {
        "expected_version": expected_version,
        "readme": str(args.readme),
        "roadmap": str(args.roadmap),
        "readme_contains_expected_version": True,
        "roadmap_contains_expected_version": True,
        "pass": True,
    }
    if expected_version:
        readme_text = args.readme.read_text(encoding="utf-8") if args.readme.exists() else ""
        roadmap_text = args.roadmap.read_text(encoding="utf-8") if args.roadmap.exists() else ""
        readme_ok = expected_version in readme_text
        roadmap_ok = expected_version in roadmap_text
        docs_sync.update(
            {
                "readme_contains_expected_version": readme_ok,
                "roadmap_contains_expected_version": roadmap_ok,
                "pass": bool(readme_ok and roadmap_ok),
            }
        )

    artifact_paths = [Path(a) for a in args.artifact]
    artifact_entries, missing_required = _artifact_entries(artifact_paths)
    artifact_bundle = {
        "required": bool(args.require_artifacts),
        "artifacts": artifact_entries,
        "missing_required": missing_required if args.require_artifacts else [],
        "pass": (len(missing_required) == 0) if args.require_artifacts else True,
    }

    metric_keys_for_phase_b = [
        "host_dispatch_reduction_rate_pct",
        "chained_latency_reduction_pct",
        "graph_pipeline_adoption_rate_pct",
    ]
    phase_b_success_pass = all(bool(metrics[k]["pass"]) for k in metric_keys_for_phase_b)

    extra_pass = (
        bool(metrics["allclose_rate_pct"]["pass"])
        and bool(metrics["fallback_reason_coverage_pct"]["pass"])
        and (docs_sync["pass"] if args.require_doc_sync else True)
        and bool(artifact_bundle["pass"])
    )

    overall_pass = bool(phase_b_success_pass and extra_pass)

    payload = {
        "artifact_kind": "phase_b_exit_audit",
        "phase": "11.2_graph_execution_foundation",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "graph_json": str(args.graph_json),
            "contract_json": str(args.contract_json),
            "cwd": os.getcwd(),
            "python": sys.version,
            "platform": platform.platform(),
        },
        "graph_source": {
            "backend_name": str(graph_payload.get("backend_name", "")),
            "device": str(graph_payload.get("device", "")),
            "summary": graph_payload.get("summary", {}),
            "required_benches": required_benches,
            "adopted_benches": graph["adopted_benches"],
            "missing_benches": graph["missing_benches"],
        },
        "metrics": metrics,
        "docs_sync": docs_sync,
        "artifact_bundle": artifact_bundle,
        "phase_b_success_metrics_pass": phase_b_success_pass,
        "overall_pass": overall_pass,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = _render_md(payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    bundle_payload = {
        "artifact_kind": "phase_b_exit_candidate_bundle",
        "generated_at_utc": payload["generated_at_utc"],
        "overall_pass": overall_pass,
        "phase_b_success_metrics_pass": phase_b_success_pass,
        "audit_json": str(args.out_json),
        "audit_md": str(args.out_md),
        "artifacts": artifact_entries,
        "missing_required": artifact_bundle["missing_required"],
    }
    args.bundle_json.parent.mkdir(parents=True, exist_ok=True)
    args.bundle_json.write_text(json.dumps(bundle_payload, indent=2), encoding="utf-8")

    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")
    print(f"saved: {args.bundle_json}")
    print(
        "phase_b_exit_audit "
        f"overall_pass={overall_pass} "
        f"dispatch={_fmt_pct(_as_float(metrics['host_dispatch_reduction_rate_pct']['observed']))} "
        f"chained={_fmt_pct(_as_float(metrics['chained_latency_reduction_pct']['observed']))} "
        f"adoption={_fmt_pct(_as_float(metrics['graph_pipeline_adoption_rate_pct']['observed']))}"
    )

    if args.require_pass and not overall_pass:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
