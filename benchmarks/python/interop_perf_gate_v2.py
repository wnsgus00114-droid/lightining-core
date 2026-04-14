#!/usr/bin/env python3
"""Interop performance gate v2.

Aggregates interop telemetry from engine split + bridge adapter artifacts into a
single, fixed-format report so release artifacts answer both:
- why fallback happened (reason coverage)
- where overhead came from (boundary decomposition)
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def _load_json(path: Path | None) -> dict:
    if path is None:
        return {}
    p = Path(path)
    if (not str(p).strip()) or (not p.exists()) or (not p.is_file()):
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _as_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.2f}%"


def _fmt_ms(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.6f}"


def _coverage_from_rows(rows: list[dict], key: str) -> float:
    if not rows:
        return 100.0
    ok_rows = [r for r in rows if str(r.get("status", "ok")).lower() == "ok"]
    if not ok_rows:
        return 100.0
    applicable = [r for r in ok_rows if str(r.get(key, "")).strip().lower() != "n/a"]
    if not applicable:
        return 100.0
    covered = [r for r in applicable if str(r.get(key, "")).strip() != ""]
    return (100.0 * float(len(covered))) / float(len(applicable))


def _max_from_rows(rows: list[dict], key: str) -> float:
    vals = [_as_float(r.get(key), float("nan")) for r in rows]
    vals = [v for v in vals if math.isfinite(v)]
    return max(vals) if vals else float("nan")


def _bridge_row(name: str, payload: dict, *, kind: str, threshold_ms: float) -> dict:
    if not payload:
        return {
            "bridge": name,
            "kind": kind,
            "status": "missing",
            "reason_coverage_pct": float("nan"),
            "explain_coverage_pct": float("nan"),
            "max_boundary_overhead_ms": float("nan"),
            "budget_pass_rate_pct": float("nan"),
            "budget_pass": False,
        }

    if name == "engine_split":
        meta = dict(payload.get("meta", {}))
        rows = list(payload.get("rows", []))
        reason_coverage = _as_float(meta.get("interop_boundary_reason_coverage_pct"), _coverage_from_rows(rows, "route_boundary_reason_code"))
        explain_coverage = _coverage_from_rows(rows, "route_boundary_reason_code")
        max_overhead_ms = _as_float(meta.get("interop_boundary_max_overhead_ms"), _max_from_rows(rows, "route_boundary_overhead_est_ms"))
        budget_pass_rate_pct = 100.0 if (math.isfinite(max_overhead_ms) and max_overhead_ms <= threshold_ms + 1.0e-12) else 0.0
        return {
            "bridge": name,
            "kind": kind,
            "status": "ok",
            "reason_coverage_pct": float(reason_coverage),
            "explain_coverage_pct": float(explain_coverage),
            "max_boundary_overhead_ms": float(max_overhead_ms),
            "budget_pass_rate_pct": float(budget_pass_rate_pct),
            "budget_pass": bool(budget_pass_rate_pct >= 100.0),
            "component_max_upload_ms": _as_float(meta.get("interop_boundary_max_upload_overhead_ms"), float("nan")),
            "component_max_engine_switch_ms": _as_float(meta.get("interop_boundary_max_engine_switch_overhead_ms"), float("nan")),
            "component_max_copy_ms": _as_float(meta.get("interop_boundary_max_copy_overhead_ms"), float("nan")),
            "component_max_sync_ms": _as_float(meta.get("interop_boundary_max_sync_overhead_ms"), float("nan")),
        }

    rows = list(payload.get("rows", []))
    reason_coverage = _as_float(payload.get("reason_coverage_pct"), _coverage_from_rows(rows, "boundary_reason_code"))
    explain_coverage = _coverage_from_rows(rows, "boundary_reason_code")
    max_overhead_ms = _max_from_rows(rows, "boundary_overhead_est_ms")
    if not math.isfinite(max_overhead_ms):
        max_overhead_ms = _as_float(payload.get("max_boundary_overhead_ms"), float("nan"))
    budget_pass_rate_pct = _as_float(payload.get("budget_pass_rate_pct"), float("nan"))
    if not math.isfinite(budget_pass_rate_pct):
        budget_pass_rate_pct = 100.0 if (math.isfinite(max_overhead_ms) and max_overhead_ms <= threshold_ms + 1.0e-12) else 0.0

    status = "ok"
    if any(str(r.get("status", "ok")).lower() != "ok" for r in rows):
        status = "mismatch"

    return {
        "bridge": name,
        "kind": kind,
        "status": status,
        "reason_coverage_pct": float(reason_coverage),
        "explain_coverage_pct": float(explain_coverage),
        "max_boundary_overhead_ms": float(max_overhead_ms),
        "budget_pass_rate_pct": float(budget_pass_rate_pct),
        "budget_pass": bool(math.isfinite(budget_pass_rate_pct) and budget_pass_rate_pct >= 100.0),
    }


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _render_md(payload: dict) -> str:
    rows = list(payload.get("bridges", []))
    lines = [
        "## Interop Perf Gate v2",
        "",
        f"- status: `{'pass' if payload.get('overall_pass', False) else 'fail'}`",
        f"- reason coverage (federation): `{_fmt_pct(_as_float(payload.get('federation_reason_coverage_pct')))}`",
        f"- explain coverage (federation): `{_fmt_pct(_as_float(payload.get('federation_explain_coverage_pct')))}`",
        f"- max boundary overhead ms: `{_fmt_ms(_as_float(payload.get('federation_max_boundary_overhead_ms')))}`",
        "",
        "| Bridge | Kind | Status | Reason Coverage | Explain Coverage | Max Boundary Overhead (ms) | Budget Pass Rate |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| "
            f"{r.get('bridge')} | {r.get('kind')} | {r.get('status')} | "
            f"{_fmt_pct(_as_float(r.get('reason_coverage_pct')))} | "
            f"{_fmt_pct(_as_float(r.get('explain_coverage_pct')))} | "
            f"{_fmt_ms(_as_float(r.get('max_boundary_overhead_ms')))} | "
            f"{_fmt_pct(_as_float(r.get('budget_pass_rate_pct')))} |"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine-interop-json", type=Path, required=True)
    p.add_argument("--torch-adapter-json", type=Path, default=Path(""))
    p.add_argument("--tf-runner-json", type=Path, default=Path(""))
    p.add_argument("--coreml-adapter-json", type=Path, default=Path(""))
    p.add_argument("--mlx-adapter-json", type=Path, default=Path(""))
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    p.add_argument("--max-boundary-overhead-ms", type=float, default=6.0)
    p.add_argument("--min-reason-coverage-pct", type=float, default=100.0)
    p.add_argument("--min-perf-explain-coverage-pct", type=float, default=100.0)
    p.add_argument("--require-reason-coverage", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-perf-explain-coverage", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-max-boundary-overhead-ms", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    engine_interop = _load_json(args.engine_interop_json)
    torch_payload = _load_json(args.torch_adapter_json)
    tf_payload = _load_json(args.tf_runner_json)
    coreml_payload = _load_json(args.coreml_adapter_json)
    mlx_payload = _load_json(args.mlx_adapter_json)

    bridges = [
        _bridge_row("engine_split", engine_interop, kind="split", threshold_ms=float(args.max_boundary_overhead_ms)),
        _bridge_row("torch", torch_payload, kind="adapter", threshold_ms=float(args.max_boundary_overhead_ms)),
        _bridge_row("tf", tf_payload, kind="adapter", threshold_ms=float(args.max_boundary_overhead_ms)),
        _bridge_row("coreml", coreml_payload, kind="adapter", threshold_ms=float(args.max_boundary_overhead_ms)),
        _bridge_row("mlx", mlx_payload, kind="adapter", threshold_ms=float(args.max_boundary_overhead_ms)),
    ]

    usable = [r for r in bridges if str(r.get("status", "missing")) != "missing"]
    rc_vals = [_as_float(r.get("reason_coverage_pct"), float("nan")) for r in usable]
    rc_vals = [v for v in rc_vals if math.isfinite(v)]
    ex_vals = [_as_float(r.get("explain_coverage_pct"), float("nan")) for r in usable]
    ex_vals = [v for v in ex_vals if math.isfinite(v)]
    oh_vals = [_as_float(r.get("max_boundary_overhead_ms"), float("nan")) for r in usable]
    oh_vals = [v for v in oh_vals if math.isfinite(v)]

    federation_reason_coverage_pct = float(sum(rc_vals) / float(len(rc_vals))) if rc_vals else float("nan")
    federation_explain_coverage_pct = float(sum(ex_vals) / float(len(ex_vals))) if ex_vals else float("nan")
    federation_max_boundary_overhead_ms = max(oh_vals) if oh_vals else float("nan")

    failures: list[str] = []
    if args.require_reason_coverage and (
        (not math.isfinite(federation_reason_coverage_pct))
        or (federation_reason_coverage_pct + 1.0e-9 < float(args.min_reason_coverage_pct))
    ):
        failures.append("reason_coverage")
    if args.require_perf_explain_coverage and (
        (not math.isfinite(federation_explain_coverage_pct))
        or (federation_explain_coverage_pct + 1.0e-9 < float(args.min_perf_explain_coverage_pct))
    ):
        failures.append("perf_explain_coverage")
    if args.require_max_boundary_overhead_ms and (
        (not math.isfinite(federation_max_boundary_overhead_ms))
        or (federation_max_boundary_overhead_ms > float(args.max_boundary_overhead_ms) + 1.0e-12)
    ):
        failures.append("max_boundary_overhead")
    if any(str(r.get("status", "ok")).lower() == "mismatch" for r in usable):
        failures.append("bridge_mismatch")

    overall_pass = len(failures) == 0

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "interop_perf_gate_v2",
        "bridges": bridges,
        "federation_reason_coverage_pct": float(federation_reason_coverage_pct),
        "federation_explain_coverage_pct": float(federation_explain_coverage_pct),
        "federation_max_boundary_overhead_ms": float(federation_max_boundary_overhead_ms),
        "thresholds": {
            "max_boundary_overhead_ms": float(args.max_boundary_overhead_ms),
            "min_reason_coverage_pct": float(args.min_reason_coverage_pct),
            "min_perf_explain_coverage_pct": float(args.min_perf_explain_coverage_pct),
        },
        "overall_pass": bool(overall_pass),
        "failures": failures,
    }

    _save_json(args.out_json, payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_md(payload), encoding="utf-8")

    print(f"status={'pass' if overall_pass else 'fail'} failures={','.join(failures) if failures else 'none'}")
    print(f"saved: {args.out_json}")
    print(f"saved: {args.out_md}")

    if args.require_pass and (not overall_pass):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
