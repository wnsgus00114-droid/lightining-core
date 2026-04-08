#!/usr/bin/env python3
"""Rule-based graph fusion pilot benchmark (conv+relu v1).

This benchmark compares GraphIR execution with fusion enabled vs disabled,
and emits an explain report for fused/non-fused decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

import lightning_core as lc


def _median_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(median(samples))


def _fmt_ms(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.6f}"


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}x"


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    backend = lc.backend_name().lower()
    if backend not in {"metal", "cuda", "cpu"}:
        return "cpu"
    return backend


def _build_graph(*, multi_consumer: bool) -> tuple[object, dict[str, int]]:
    g = lc.GraphIR()
    ids: dict[str, int] = {}
    ids["x"] = g.add_tensor([1, 3, 8, 8], dtype="float32", name="x", constant=True)
    ids["w"] = g.add_tensor([16, 3, 3, 3], dtype="float32", name="w", constant=True)
    ids["b"] = g.add_tensor([16], dtype="float32", name="b", constant=True)
    ids["mid"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="mid")
    ids["out"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="out")
    g.add_node("conv2d_nchw3x3s1p1", [ids["x"], ids["w"], ids["b"]], [ids["mid"]])
    g.add_node("relu", [ids["mid"]], [ids["out"]])

    if multi_consumer:
        ids["bias2"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="bias2", constant=True)
        ids["side"] = g.add_tensor([1, 16, 8, 8], dtype="float32", name="side")
        g.add_node("vector_add", [ids["mid"], ids["bias2"]], [ids["side"]])

    return g, ids


def _run_case(
    *,
    case_name: str,
    multi_consumer: bool,
    device: str,
    warmup: int,
    iters: int,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    g, ids = _build_graph(multi_consumer=multi_consumer)

    x = rng.random((1, 3, 8, 8), dtype=np.float32)
    w = rng.random((16, 3, 3, 3), dtype=np.float32)
    b = rng.random((16,), dtype=np.float32)
    feeds: dict[int, np.ndarray] = {
        ids["x"]: x,
        ids["w"]: w,
        ids["b"]: b,
    }
    if multi_consumer:
        feeds[ids["bias2"]] = rng.random((1, 16, 8, 8), dtype=np.float32)

    def run_fused_once():
        _ = g.execute_f32(feeds, preferred_device=device, enable_fusion_v1=True)

    def run_unfused_once():
        _ = g.execute_f32(feeds, preferred_device=device, enable_fusion_v1=False)

    fused_ms = _median_ms(run_fused_once, warmup, iters)
    unfused_ms = _median_ms(run_unfused_once, warmup, iters)

    fused_out = np.asarray(
        g.execute_f32(feeds, preferred_device=device, enable_fusion_v1=True)["values"][ids["out"]],
        dtype=np.float32,
    ).reshape(-1)
    unfused_out = np.asarray(
        g.execute_f32(feeds, preferred_device=device, enable_fusion_v1=False)["values"][ids["out"]],
        dtype=np.float32,
    ).reshape(-1)
    abs_diff = np.abs(fused_out - unfused_out)
    rel_diff = abs_diff / (np.abs(unfused_out) + 1.0e-12)

    decisions_enabled = list(g.fusion_report(preferred_device=device, enable_fusion_v1=True))
    decisions_disabled = list(g.fusion_report(preferred_device=device, enable_fusion_v1=False))

    conv_relu_decision_enabled = next((d for d in decisions_enabled if d.get("pattern") == "conv_relu_v1"), None)
    conv_relu_decision_disabled = next((d for d in decisions_disabled if d.get("pattern") == "conv_relu_v1"), None)

    row = {
        "suite": "fusion_pilot",
        "bench": case_name,
        "shape": "conv(n=1,c=3->16,h=8,w=8,k=3)+relu",
        "status": "ok",
        "device": device,
        "fused_ms": fused_ms,
        "unfused_ms": unfused_ms,
        "fused_over_unfused": (fused_ms / unfused_ms) if unfused_ms > 0 else float("nan"),
        "unfused_over_fused": (unfused_ms / fused_ms) if fused_ms > 0 else float("nan"),
        "allclose": bool(np.all(abs_diff <= (1.0e-4 + 1.0e-4 * np.abs(unfused_out)))),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "fusion_applied": bool(conv_relu_decision_enabled and conv_relu_decision_enabled.get("fused", False)),
        "fusion_reason": str(conv_relu_decision_enabled.get("reason", "n/a")) if conv_relu_decision_enabled else "n/a",
        "fusion_disabled_reason": (
            str(conv_relu_decision_disabled.get("reason", "n/a")) if conv_relu_decision_disabled else "n/a"
        ),
        "note": "",
    }

    detail = {
        "row": row,
        "fusion_report_enabled": decisions_enabled,
        "fusion_report_disabled": decisions_disabled,
    }
    return row, detail


def _unsupported_row(bench: str, device: str, reason: str) -> tuple[dict, dict]:
    row = {
        "suite": "fusion_pilot",
        "bench": bench,
        "shape": "conv(n=1,c=3->16,h=8,w=8,k=3)+relu",
        "status": "unsupported",
        "device": device,
        "fused_ms": float("nan"),
        "unfused_ms": float("nan"),
        "fused_over_unfused": float("nan"),
        "unfused_over_fused": float("nan"),
        "allclose": False,
        "max_abs_diff": float("nan"),
        "max_rel_diff": float("nan"),
        "fusion_applied": False,
        "fusion_reason": "n/a",
        "fusion_disabled_reason": "n/a",
        "note": reason,
    }
    return row, {"row": row, "error": reason}


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "suite",
        "bench",
        "shape",
        "status",
        "device",
        "fused_ms",
        "unfused_ms",
        "fused_over_unfused",
        "unfused_over_fused",
        "allclose",
        "max_abs_diff",
        "max_rel_diff",
        "fusion_applied",
        "fusion_reason",
        "fusion_disabled_reason",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _to_markdown(rows: list[dict], summary: dict, device: str, warmup: int, iters: int, perf_threshold: float) -> str:
    lines = []
    lines.append("## Fusion Pilot (conv+relu v1)")
    lines.append("")
    lines.append(f"- device: `{device}`")
    lines.append(f"- config: warmup={warmup}, iters={iters}")
    lines.append(f"- total cases: {summary['total_cases']} (ok={summary['ok_cases']}, unsupported={summary['unsupported_cases']})")
    lines.append(f"- fusion applied cases: {summary['fusion_applied_cases']}/{summary['ok_cases']}")
    lines.append(
        f"- median fused/unfused: {_fmt_ratio(summary['median_fused_over_unfused'])}, "
        f"perf gate threshold: fused/unfused <= {perf_threshold:.3f}x"
    )
    lines.append("")
    lines.append("| bench | status | fused (ms) | unfused (ms) | fused/unfused | fusion_applied | reason |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['bench']} | {r['status']} | {_fmt_ms(float(r['fused_ms']))} | {_fmt_ms(float(r['unfused_ms']))} | "
            f"{_fmt_ratio(float(r['fused_over_unfused']))} | {r['fusion_applied']} | {r['fusion_reason']} |"
        )
    return "\n".join(lines)


def _summary(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    ratios = [float(r["fused_over_unfused"]) for r in ok_rows if math.isfinite(float(r["fused_over_unfused"]))]
    return {
        "total_cases": len(rows),
        "ok_cases": len(ok_rows),
        "unsupported_cases": len(rows) - len(ok_rows),
        "fusion_applied_cases": sum(1 for r in ok_rows if bool(r.get("fusion_applied", False))),
        "allclose_ok_cases": sum(1 for r in ok_rows if bool(r.get("allclose", False))),
        "median_fused_over_unfused": float(median(ratios)) if ratios else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Fusion pilot benchmark for conv+relu v1")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260408)
    p.add_argument("--csv", type=Path, default=Path("benchmarks/reports/ci/fusion_pilot.csv"))
    p.add_argument("--json", type=Path, default=Path("benchmarks/reports/ci/fusion_pilot.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/reports/ci/fusion_pilot.md"))
    p.add_argument(
        "--max-fused-over-unfused",
        type=float,
        default=1.10,
        help="Fail when eligible fused case exceeds this ratio.",
    )
    p.add_argument(
        "--fail-on-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when no case succeeds.",
    )
    args = p.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = _resolve_device(args.device)

    rows: list[dict] = []
    details: list[dict] = []
    for case_name, multi_consumer in [
        ("conv_relu_eligible", False),
        ("conv_relu_multi_consumer", True),
    ]:
        try:
            row, detail = _run_case(
                case_name=case_name,
                multi_consumer=multi_consumer,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                rng=rng,
            )
        except Exception as exc:
            row, detail = _unsupported_row(case_name, device, str(exc))
        rows.append(row)
        details.append(detail)

    summary = _summary(rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "fusion_pilot",
        "backend_name": lc.backend_name(),
        "device": device,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "max_fused_over_unfused": args.max_fused_over_unfused,
        "summary": summary,
        "rows": rows,
        "details": details,
    }

    _write_csv(args.csv, rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(
        _to_markdown(rows, summary, device, args.warmup, args.iters, args.max_fused_over_unfused),
        encoding="utf-8",
    )

    print(f"saved: {args.csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if not ok_rows:
        if args.fail_on_empty:
            raise SystemExit(2)
        return
    if any(not bool(r.get("allclose", False)) for r in ok_rows):
        raise SystemExit(3)

    eligible_rows = [r for r in ok_rows if r["bench"] == "conv_relu_eligible"]
    if eligible_rows:
        for r in eligible_rows:
            ratio = float(r.get("fused_over_unfused", float("nan")))
            if bool(r.get("fusion_applied", False)) and math.isfinite(ratio) and ratio > args.max_fused_over_unfused:
                raise SystemExit(4)


if __name__ == "__main__":
    main()
