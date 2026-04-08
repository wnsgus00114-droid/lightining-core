#!/usr/bin/env python3
"""Graph/Eager A/B benchmark for pipeline-style workloads.

This script compares eager vs graph execution on shipped Lightning Core paths
and publishes host-dispatch/fallback counters from runtime trace events.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
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


def _runtime_trace_available() -> bool:
    required = (
        "runtime_trace_clear",
        "runtime_trace_enable",
        "runtime_trace_events",
        "runtime_trace_timeline",
    )
    return all(hasattr(lc, name) for name in required)


def _collect_trace_metrics(run_once, trace_iters: int) -> dict:
    if trace_iters <= 0:
        raise ValueError("trace_iters must be > 0")

    if not _runtime_trace_available():
        return {
            "events_total": 0,
            "events_per_iter": float("nan"),
            "dispatch_events": 0,
            "dispatch_per_iter": float("nan"),
            "fallback_events": 0,
            "fallback_per_iter": float("nan"),
            "timeline_window_ns": 0,
            "timeline_group_count": 0,
            "top_op_path": "trace_unavailable",
            "top_groups": [],
            "note": "runtime trace API unavailable",
        }

    lc.runtime_trace_clear()
    lc.runtime_trace_enable(True)
    try:
        for _ in range(trace_iters):
            run_once()
    finally:
        lc.runtime_trace_enable(False)

    events = lc.runtime_trace_events()
    timeline = lc.runtime_trace_timeline(
        event_sort_by="timestamp_ns",
        event_descending=False,
        group_by="op_path",
        group_sort_by="total_delta_next_ns",
        group_descending=True,
        hotspot_top_k=8,
    )

    dispatch_events = [ev for ev in events if ev.get("type") == "op_dispatch"]
    fallback_events = [ev for ev in dispatch_events if bool(ev.get("fallback", False))]
    groups = list(timeline.get("groups", []))
    top_op_path = groups[0]["key"] if groups else "n/a"

    return {
        "events_total": len(events),
        "events_per_iter": float(len(events)) / float(trace_iters),
        "dispatch_events": len(dispatch_events),
        "dispatch_per_iter": float(len(dispatch_events)) / float(trace_iters),
        "fallback_events": len(fallback_events),
        "fallback_per_iter": float(len(fallback_events)) / float(trace_iters),
        "timeline_window_ns": int(timeline.get("window_ns", 0)),
        "timeline_group_count": len(groups),
        "top_op_path": top_op_path,
        "top_groups": groups[:6],
        "note": "",
    }


def _fmt_ms(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.6f}"


def _fmt_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}x"


def _fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.2f}%"


def _safe_dispatch_reduction(eager_dispatch: float, graph_dispatch: float) -> tuple[float, float]:
    if not (math.isfinite(eager_dispatch) and math.isfinite(graph_dispatch)):
        return float("nan"), float("nan")
    reduction = eager_dispatch - graph_dispatch
    reduction_pct = (
        (reduction / eager_dispatch) * 100.0
        if eager_dispatch > 0.0
        else float("nan")
    )
    return reduction, reduction_pct


def _run_matmul_matrix_sub_case(
    *,
    m: int,
    k: int,
    n: int,
    device: str,
    warmup: int,
    iters: int,
    trace_iters: int,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)
    bias = rng.random((m, n), dtype=np.float32)

    g = lc.GraphIR()
    ta = g.add_tensor([m, k], dtype="float32", name="a", constant=True)
    tb = g.add_tensor([k, n], dtype="float32", name="b", constant=True)
    tbias = g.add_tensor([m, n], dtype="float32", name="bias", constant=True)
    tmm = g.add_tensor([m, n], dtype="float32", name="mm")
    tout = g.add_tensor([m, n], dtype="float32", name="out")
    g.add_node("matmul", [ta, tb], [tmm])
    g.add_node("matrix_sub", [tmm, tbias], [tout])

    feeds = {ta: a, tb: b, tbias: bias}

    def eager_once():
        mm = lc.matmul2d(a, b, device)
        _ = lc.matrix_sub(mm, bias, m, n, device)

    def graph_once():
        _ = g.execute_f32(feeds, preferred_device=device)

    eager_ms = _median_ms(eager_once, warmup, iters)
    graph_ms = _median_ms(graph_once, warmup, iters)

    eager_out = np.asarray(lc.matrix_sub(lc.matmul2d(a, b, device), bias, m, n, device), dtype=np.float32).reshape(-1)
    graph_out = np.asarray(g.execute_f32(feeds, preferred_device=device)["values"][tout], dtype=np.float32).reshape(-1)
    abs_diff = np.abs(eager_out - graph_out)
    rel_diff = abs_diff / (np.abs(eager_out) + 1.0e-12)

    eager_trace = _collect_trace_metrics(eager_once, trace_iters)
    graph_trace = _collect_trace_metrics(graph_once, trace_iters)

    dispatch_delta = graph_trace["dispatch_per_iter"] - eager_trace["dispatch_per_iter"]
    dispatch_delta_pct = (
        (dispatch_delta / eager_trace["dispatch_per_iter"]) * 100.0
        if math.isfinite(eager_trace["dispatch_per_iter"]) and eager_trace["dispatch_per_iter"] > 0
        else float("nan")
    )
    dispatch_reduction_per_iter, dispatch_reduction_pct = _safe_dispatch_reduction(
        eager_trace["dispatch_per_iter"],
        graph_trace["dispatch_per_iter"],
    )
    fallback_delta = graph_trace["fallback_per_iter"] - eager_trace["fallback_per_iter"]

    row = {
        "suite": "graph_eager_ab",
        "bench": "matmul_matrix_sub",
        "shape": f"m={m},k={k},n={n}",
        "status": "ok",
        "device": device,
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "graph_over_eager": (graph_ms / eager_ms) if eager_ms > 0 else float("nan"),
        "eager_over_graph": (eager_ms / graph_ms) if graph_ms > 0 else float("nan"),
        "allclose": bool(np.all(abs_diff <= (1.0e-4 + 1.0e-4 * np.abs(eager_out)))),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "eager_dispatch_per_iter": eager_trace["dispatch_per_iter"],
        "graph_dispatch_per_iter": graph_trace["dispatch_per_iter"],
        "dispatch_delta_per_iter": dispatch_delta,
        "dispatch_delta_pct": dispatch_delta_pct,
        "dispatch_reduction_per_iter": dispatch_reduction_per_iter,
        "dispatch_reduction_pct": dispatch_reduction_pct,
        "host_dispatch_reduced": bool(
            math.isfinite(dispatch_reduction_per_iter) and dispatch_reduction_per_iter > 0.0
        ),
        "eager_fallback_per_iter": eager_trace["fallback_per_iter"],
        "graph_fallback_per_iter": graph_trace["fallback_per_iter"],
        "fallback_delta_per_iter": fallback_delta,
        "eager_events_per_iter": eager_trace["events_per_iter"],
        "graph_events_per_iter": graph_trace["events_per_iter"],
        "eager_top_op_path": eager_trace["top_op_path"],
        "graph_top_op_path": graph_trace["top_op_path"],
        "note": "",
    }
    detail = {
        "row": row,
        "trace": {
            "eager": eager_trace,
            "graph": graph_trace,
        },
    }
    return row, detail


def _run_conv_attention_case(
    *,
    batch: int,
    in_ch: int,
    h: int,
    w: int,
    out_ch: int,
    seq_len: int,
    head_dim: int,
    device: str,
    warmup: int,
    iters: int,
    trace_iters: int,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    if not hasattr(lc, "api") or not hasattr(lc.api, "conv_attention_torchstrong_nchw"):
        raise RuntimeError("lc.api.conv_attention_torchstrong_nchw is not available in this build")

    x = rng.random((batch, in_ch, h, w), dtype=np.float32)
    w_conv = rng.random((out_ch, in_ch, 3, 3), dtype=np.float32)
    b_conv = rng.random((out_ch,), dtype=np.float32)

    def eager_once():
        _ = lc.api.conv_attention_torchstrong_nchw(
            x,
            w_conv,
            b_conv,
            seq_len,
            head_dim,
            1,
            1,
            1,
            1,
            device,
            "eager",
        )

    def graph_once():
        _ = lc.api.conv_attention_torchstrong_nchw(
            x,
            w_conv,
            b_conv,
            seq_len,
            head_dim,
            1,
            1,
            1,
            1,
            device,
            "graph",
        )

    eager_ms = _median_ms(eager_once, warmup, iters)
    graph_ms = _median_ms(graph_once, warmup, iters)

    eager_out = np.asarray(
        lc.api.conv_attention_torchstrong_nchw(
            x, w_conv, b_conv, seq_len, head_dim, 1, 1, 1, 1, device, "eager"
        ),
        dtype=np.float32,
    ).reshape(-1)
    graph_out = np.asarray(
        lc.api.conv_attention_torchstrong_nchw(
            x, w_conv, b_conv, seq_len, head_dim, 1, 1, 1, 1, device, "graph"
        ),
        dtype=np.float32,
    ).reshape(-1)
    abs_diff = np.abs(eager_out - graph_out)
    rel_diff = abs_diff / (np.abs(eager_out) + 1.0e-12)

    eager_trace = _collect_trace_metrics(eager_once, trace_iters)
    graph_trace = _collect_trace_metrics(graph_once, trace_iters)

    dispatch_delta = graph_trace["dispatch_per_iter"] - eager_trace["dispatch_per_iter"]
    dispatch_delta_pct = (
        (dispatch_delta / eager_trace["dispatch_per_iter"]) * 100.0
        if math.isfinite(eager_trace["dispatch_per_iter"]) and eager_trace["dispatch_per_iter"] > 0
        else float("nan")
    )
    dispatch_reduction_per_iter, dispatch_reduction_pct = _safe_dispatch_reduction(
        eager_trace["dispatch_per_iter"],
        graph_trace["dispatch_per_iter"],
    )
    fallback_delta = graph_trace["fallback_per_iter"] - eager_trace["fallback_per_iter"]

    row = {
        "suite": "graph_eager_ab",
        "bench": "conv_attention_torchstrong_nchw",
        "shape": f"conv(n={batch},c={in_ch}->{out_ch},h={h},w={w},k=3)+attn(seq={seq_len},d={head_dim})",
        "status": "ok",
        "device": device,
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "graph_over_eager": (graph_ms / eager_ms) if eager_ms > 0 else float("nan"),
        "eager_over_graph": (eager_ms / graph_ms) if graph_ms > 0 else float("nan"),
        "allclose": bool(np.all(abs_diff <= (1.0e-4 + 1.0e-4 * np.abs(eager_out)))),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "eager_dispatch_per_iter": eager_trace["dispatch_per_iter"],
        "graph_dispatch_per_iter": graph_trace["dispatch_per_iter"],
        "dispatch_delta_per_iter": dispatch_delta,
        "dispatch_delta_pct": dispatch_delta_pct,
        "dispatch_reduction_per_iter": dispatch_reduction_per_iter,
        "dispatch_reduction_pct": dispatch_reduction_pct,
        "host_dispatch_reduced": bool(
            math.isfinite(dispatch_reduction_per_iter) and dispatch_reduction_per_iter > 0.0
        ),
        "eager_fallback_per_iter": eager_trace["fallback_per_iter"],
        "graph_fallback_per_iter": graph_trace["fallback_per_iter"],
        "fallback_delta_per_iter": fallback_delta,
        "eager_events_per_iter": eager_trace["events_per_iter"],
        "graph_events_per_iter": graph_trace["events_per_iter"],
        "eager_top_op_path": eager_trace["top_op_path"],
        "graph_top_op_path": graph_trace["top_op_path"],
        "note": "",
    }
    detail = {
        "row": row,
        "trace": {
            "eager": eager_trace,
            "graph": graph_trace,
        },
    }
    return row, detail


def _unsupported_row(bench: str, shape: str, device: str, reason: str) -> tuple[dict, dict]:
    row = {
        "suite": "graph_eager_ab",
        "bench": bench,
        "shape": shape,
        "status": "unsupported",
        "device": device,
        "eager_ms": float("nan"),
        "graph_ms": float("nan"),
        "graph_over_eager": float("nan"),
        "eager_over_graph": float("nan"),
        "allclose": False,
        "max_abs_diff": float("nan"),
        "mean_abs_diff": float("nan"),
        "max_rel_diff": float("nan"),
        "eager_dispatch_per_iter": float("nan"),
        "graph_dispatch_per_iter": float("nan"),
        "dispatch_delta_per_iter": float("nan"),
        "dispatch_delta_pct": float("nan"),
        "dispatch_reduction_per_iter": float("nan"),
        "dispatch_reduction_pct": float("nan"),
        "host_dispatch_reduced": False,
        "eager_fallback_per_iter": float("nan"),
        "graph_fallback_per_iter": float("nan"),
        "fallback_delta_per_iter": float("nan"),
        "eager_events_per_iter": float("nan"),
        "graph_events_per_iter": float("nan"),
        "eager_top_op_path": "n/a",
        "graph_top_op_path": "n/a",
        "note": reason,
    }
    detail = {"row": row, "trace": {"eager": {}, "graph": {}}, "error": reason}
    return row, detail


def _summary(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r["status"] == "ok"]
    unsupported_rows = [r for r in rows if r["status"] != "ok"]

    def _finite(values):
        return [v for v in values if math.isfinite(v)]

    ratios = _finite([float(r["graph_over_eager"]) for r in ok_rows])
    dispatch_deltas = _finite([float(r["dispatch_delta_pct"]) for r in ok_rows])
    dispatch_reductions_pct = _finite([float(r["dispatch_reduction_pct"]) for r in ok_rows])
    dispatch_reductions_abs = _finite([float(r["dispatch_reduction_per_iter"]) for r in ok_rows])
    fallback_deltas = _finite([float(r["fallback_delta_per_iter"]) for r in ok_rows])
    graph_wins = sum(1 for r in ok_rows if math.isfinite(float(r["graph_ms"])) and math.isfinite(float(r["eager_ms"])) and float(r["graph_ms"]) < float(r["eager_ms"]))
    eager_wins = sum(1 for r in ok_rows if math.isfinite(float(r["graph_ms"])) and math.isfinite(float(r["eager_ms"])) and float(r["eager_ms"]) < float(r["graph_ms"]))
    ties = max(0, len(ok_rows) - graph_wins - eager_wins)
    host_dispatch_reduction_cases = sum(
        1 for r in ok_rows if bool(r.get("host_dispatch_reduced", False))
    )
    host_dispatch_reduction_rate_pct = (
        (float(host_dispatch_reduction_cases) / float(len(ok_rows))) * 100.0
        if ok_rows
        else 0.0
    )

    return {
        "total_cases": len(rows),
        "ok_cases": len(ok_rows),
        "unsupported_cases": len(unsupported_rows),
        "graph_wins": graph_wins,
        "eager_wins": eager_wins,
        "ties": ties,
        "allclose_ok_cases": sum(1 for r in ok_rows if bool(r["allclose"])),
        "median_graph_over_eager": float(median(ratios)) if ratios else float("nan"),
        "median_dispatch_delta_pct": float(median(dispatch_deltas)) if dispatch_deltas else float("nan"),
        "median_dispatch_reduction_pct": float(median(dispatch_reductions_pct)) if dispatch_reductions_pct else 0.0,
        "mean_dispatch_reduction_per_iter": (
            float(sum(dispatch_reductions_abs) / len(dispatch_reductions_abs))
            if dispatch_reductions_abs
            else 0.0
        ),
        "host_dispatch_reduction_cases": host_dispatch_reduction_cases,
        "host_dispatch_reduction_rate_pct": host_dispatch_reduction_rate_pct,
        "median_fallback_delta_per_iter": float(median(fallback_deltas)) if fallback_deltas else float("nan"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "suite",
        "bench",
        "shape",
        "status",
        "device",
        "eager_ms",
        "graph_ms",
        "graph_over_eager",
        "eager_over_graph",
        "allclose",
        "max_abs_diff",
        "mean_abs_diff",
        "max_rel_diff",
        "eager_dispatch_per_iter",
        "graph_dispatch_per_iter",
        "dispatch_delta_per_iter",
        "dispatch_delta_pct",
        "dispatch_reduction_per_iter",
        "dispatch_reduction_pct",
        "host_dispatch_reduced",
        "eager_fallback_per_iter",
        "graph_fallback_per_iter",
        "fallback_delta_per_iter",
        "eager_events_per_iter",
        "graph_events_per_iter",
        "eager_top_op_path",
        "graph_top_op_path",
        "note",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _to_markdown(rows: list[dict], summary: dict, device: str, warmup: int, iters: int, trace_iters: int) -> str:
    lines = []
    lines.append("## Graph/Eager A/B Benchmark")
    lines.append("")
    lines.append(f"- device: `{device}`")
    lines.append(f"- config: warmup={warmup}, iters={iters}, trace_iters={trace_iters}")
    lines.append(f"- total cases: {summary['total_cases']} (ok={summary['ok_cases']}, unsupported={summary['unsupported_cases']})")
    lines.append(
        f"- winner count: graph={summary['graph_wins']}, eager={summary['eager_wins']}, tie={summary['ties']}"
    )
    lines.append(
        f"- median graph/eager: {_fmt_ratio(summary['median_graph_over_eager'])}, "
        f"median dispatch delta: {_fmt_pct(summary['median_dispatch_delta_pct'])}, "
        f"median fallback delta/iter: {_fmt_ms(summary['median_fallback_delta_per_iter'])}"
    )
    lines.append(
        f"- host dispatch reduction (fixed metric): "
        f"cases={summary['host_dispatch_reduction_cases']}/{summary['ok_cases']}, "
        f"rate={_fmt_pct(summary['host_dispatch_reduction_rate_pct'])}, "
        f"median reduction={_fmt_pct(summary['median_dispatch_reduction_pct'])}, "
        f"mean reduction/iter={_fmt_ms(summary['mean_dispatch_reduction_per_iter'])}"
    )
    lines.append("")
    lines.append("| bench | shape | status | eager (ms) | graph (ms) | graph/eager | dispatch delta (%) | dispatch reduction (%) | fallback delta/iter |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows:
        lines.append(
            f"| {r['bench']} | {r['shape']} | {r['status']} | {_fmt_ms(float(r['eager_ms']))} | "
            f"{_fmt_ms(float(r['graph_ms']))} | {_fmt_ratio(float(r['graph_over_eager']))} | "
            f"{_fmt_pct(float(r['dispatch_delta_pct']))} | {_fmt_pct(float(r['dispatch_reduction_pct']))} | "
            f"{_fmt_ms(float(r['fallback_delta_per_iter']))} |"
        )
    lines.append("")
    unsupported = [r for r in rows if r["status"] != "ok"]
    if unsupported:
        lines.append("### Unsupported Cases")
        for r in unsupported:
            lines.append(f"- {r['bench']} / {r['shape']}: {r['note']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Graph/Eager A/B benchmark with runtime dispatch metrics")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--trace-iters", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260401)
    p.add_argument("--csv", type=Path, default=Path("benchmarks/reports/ci/graph_eager_ab.csv"))
    p.add_argument("--json", type=Path, default=Path("benchmarks/reports/ci/graph_eager_ab.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/reports/ci/graph_eager_ab.md"))
    p.add_argument(
        "--fail-on-empty",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exit non-zero when no case succeeds.",
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.device == "auto":
        device = lc.backend_name().lower()
        if device not in {"metal", "cuda", "cpu"}:
            device = "cpu"
    else:
        device = args.device

    rows: list[dict] = []
    details: list[dict] = []

    matmul_cases = [(128, 128, 128), (256, 256, 256)]
    for m, k, n in matmul_cases:
        shape = f"m={m},k={k},n={n}"
        try:
            row, detail = _run_matmul_matrix_sub_case(
                m=m,
                k=k,
                n=n,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                trace_iters=args.trace_iters,
                rng=rng,
            )
        except Exception as exc:
            row, detail = _unsupported_row("matmul_matrix_sub", shape, device, str(exc))
        rows.append(row)
        details.append(detail)

    conv_attn_cases = [
        (1, 3, 8, 8, 16, 48, 48),
        (1, 3, 8, 8, 16, 96, 48),
        (1, 3, 8, 8, 16, 192, 48),
        (1, 3, 8, 8, 16, 192, 8),
    ]
    for batch, in_ch, h, w, out_ch, seq_len, head_dim in conv_attn_cases:
        shape = f"conv(n={batch},c={in_ch}->{out_ch},h={h},w={w},k=3)+attn(seq={seq_len},d={head_dim})"
        try:
            row, detail = _run_conv_attention_case(
                batch=batch,
                in_ch=in_ch,
                h=h,
                w=w,
                out_ch=out_ch,
                seq_len=seq_len,
                head_dim=head_dim,
                device=device,
                warmup=args.warmup,
                iters=args.iters,
                trace_iters=args.trace_iters,
                rng=rng,
            )
        except Exception as exc:
            row, detail = _unsupported_row("conv_attention_torchstrong_nchw", shape, device, str(exc))
        rows.append(row)
        details.append(detail)

    summary = _summary(rows)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "graph_eager_ab",
        "backend_name": lc.backend_name(),
        "device": device,
        "warmup": args.warmup,
        "iters": args.iters,
        "trace_iters": args.trace_iters,
        "seed": args.seed,
        "summary": summary,
        "rows": rows,
        "details": details,
    }

    _write_csv(args.csv, rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(
        _to_markdown(rows, summary, device, args.warmup, args.iters, args.trace_iters),
        encoding="utf-8",
    )

    print(f"saved: {args.csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")
    print(
        f"ok_cases={summary['ok_cases']} unsupported_cases={summary['unsupported_cases']} "
        f"graph_wins={summary['graph_wins']} eager_wins={summary['eager_wins']} "
        f"median_graph_over_eager={summary['median_graph_over_eager']} "
        f"host_dispatch_reduction_rate_pct={summary['host_dispatch_reduction_rate_pct']}"
    )

    if args.fail_on_empty and summary["ok_cases"] == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
