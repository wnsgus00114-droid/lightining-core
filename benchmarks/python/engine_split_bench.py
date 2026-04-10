#!/usr/bin/env python3
"""Engine-split benchmark for pure-LC vs interop overhead.

Outputs are intentionally separated:
- pure-LC report: runtime direct vs lc.api(lightning)
- interop report: lc.api(lightning) vs lc.api(torch)

This benchmark also records runtime trace timeline hotspots (`group_by=op_path`)
for bottleneck-oriented optimization and release evidence.
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
from typing import Callable

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def _time_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(median(samples))


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0.0 or not math.isfinite(numer) or not math.isfinite(denom):
        return float("nan")
    return numer / denom


def _fmt_ms(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.6f}"


def _fmt_ratio(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.3f}x"


def _torch_available() -> bool:
    return torch is not None


def _runtime_trace_available() -> bool:
    required = (
        "runtime_trace_clear",
        "runtime_trace_enable",
        "runtime_trace_events",
        "runtime_trace_timeline",
    )
    return all(hasattr(lc, name) for name in required)


def _trace_payload_na(note: str) -> dict:
    return {
        "events_per_iter": float("nan"),
        "dispatch_per_iter": float("nan"),
        "fallback_per_iter": float("nan"),
        "top_op_path": "n/a",
        "timeline_window_ns": 0,
        "timeline_group_count": 0,
        "note": note,
    }


def _collect_trace_metrics(run_once: Callable[[], None], trace_iters: int) -> dict:
    if trace_iters <= 0:
        return _trace_payload_na("trace disabled")
    if not _runtime_trace_available():
        return _trace_payload_na("runtime trace API unavailable")

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
        "events_per_iter": float(len(events)) / float(trace_iters),
        "dispatch_per_iter": float(len(dispatch_events)) / float(trace_iters),
        "fallback_per_iter": float(len(fallback_events)) / float(trace_iters),
        "top_op_path": top_op_path,
        "timeline_window_ns": int(timeline.get("window_ns", 0)),
        "timeline_group_count": len(groups),
        "note": "",
    }


def _boundary_defaults() -> dict[str, object]:
    return {
        "route_boundary_switch_count": float("nan"),
        "route_boundary_copy_mode": "n/a",
        "route_boundary_reason_code": "n/a",
        "route_boundary_copy_bytes_estimate": float("nan"),
        "route_boundary_overhead_est_ns": float("nan"),
        "route_boundary_overhead_est_ms": float("nan"),
        "route_zero_copy_eligible": False,
    }


def _apply_boundary_defaults(row: dict) -> dict:
    for k, v in _boundary_defaults().items():
        row.setdefault(k, v)
    return row


def _conv_attn_route_report(
    *,
    x: np.ndarray,
    w: np.ndarray,
    b: np.ndarray,
    seq: int,
    dim: int,
    device: str,
) -> dict[str, object]:
    if not (hasattr(lc, "api") and hasattr(lc.api, "conv_attention_torchstrong_nchw_route_report")):
        return {}
    rep = lc.api.conv_attention_torchstrong_nchw_route_report(
        x,
        w,
        b,
        int(seq),
        int(dim),
        1,
        1,
        1,
        1,
        device,
        "eager",
        None,
    )
    return dict(rep) if isinstance(rep, dict) else {}


def _resolve_device(arg_device: str) -> str:
    if arg_device != "auto":
        return arg_device
    return "metal" if lc.backend_name().lower() == "metal" else "cpu"


def _set_engine(name: str) -> None:
    if hasattr(lc, "api") and hasattr(lc.api, "set_engine"):
        lc.api.set_engine(name)
        return
    lc_api.set_backend(name)


def _get_engine() -> str:
    if hasattr(lc, "api") and hasattr(lc.api, "get_engine"):
        return str(lc.api.get_engine())
    return str(lc_api.get_backend())


def _runtime_conv_relu(x, w, b, device: str):
    if hasattr(lc, "lightning_conv_relu_nchw"):
        return lc.lightning_conv_relu_nchw(x, w, b, 1, 1, 1, 1, device)
    y = lc.conv2d_nchw(x, w, b, 1, 1, 1, 1, device)
    np.maximum(y, 0.0, out=y)
    return y


def _runtime_conv_attention(x, w, b, seq: int, head_dim: int, device: str):
    if hasattr(lc, "lightning_conv_attention_torchstrong_nchw"):
        try:
            return lc.lightning_conv_attention_torchstrong_nchw(
                x, w, b, seq, head_dim, 1, 1, 1, 1, device, "eager"
            )
        except TypeError:
            return lc.lightning_conv_attention_torchstrong_nchw(
                x, w, b, seq, head_dim, 1, 1, 1, 1, device
            )

    if hasattr(lc, "api") and hasattr(lc.api, "conv_attention_torchstrong_nchw_lightning_direct"):
        return lc.api.conv_attention_torchstrong_nchw_lightning_direct(
            x, w, b, seq, head_dim, 1, 1, 1, 1, device, "eager"
        )

    raise RuntimeError("conv_attention_torchstrong_nchw runtime direct path is unavailable")


def _benchmark_case(
    *,
    bench: str,
    shape: str,
    runtime_fn: Callable[[], None],
    api_fn: Callable[[], None],
    interop_fn: Callable[[], None] | None,
    device: str,
    warmup: int,
    iters: int,
    trace_iters: int,
) -> tuple[dict, dict]:
    runtime_ms = _time_ms(runtime_fn, warmup, iters)
    runtime_trace = _collect_trace_metrics(runtime_fn, trace_iters)

    _set_engine("lightning")
    api_lightning_ms = _time_ms(api_fn, warmup, iters)
    api_lightning_trace = _collect_trace_metrics(api_fn, trace_iters)

    pure_row = {
        "suite": "engine_split_pure_lc",
        "bench": bench,
        "shape": shape,
        "device": device,
        "status": "ok",
        "lc_runtime_ms": runtime_ms,
        "lc_api_lightning_ms": api_lightning_ms,
        "api_over_runtime": _ratio(api_lightning_ms, runtime_ms),
        "runtime_over_api": _ratio(runtime_ms, api_lightning_ms),
        "runtime_top_op_path": runtime_trace["top_op_path"],
        "api_lightning_top_op_path": api_lightning_trace["top_op_path"],
        "runtime_dispatch_per_iter": runtime_trace["dispatch_per_iter"],
        "api_lightning_dispatch_per_iter": api_lightning_trace["dispatch_per_iter"],
        "runtime_fallback_per_iter": runtime_trace["fallback_per_iter"],
        "api_lightning_fallback_per_iter": api_lightning_trace["fallback_per_iter"],
        "note": runtime_trace["note"] or api_lightning_trace["note"],
    }
    _apply_boundary_defaults(pure_row)

    if interop_fn is None:
        interop_row = {
            "suite": "engine_split_interop",
            "bench": bench,
            "shape": shape,
            "device": device,
            "status": "unsupported",
            "lc_api_lightning_ms": api_lightning_ms,
            "lc_api_torch_ms": float("nan"),
            "interop_over_pure": float("nan"),
            "pure_over_interop": float("nan"),
            "api_lightning_top_op_path": api_lightning_trace["top_op_path"],
            "api_torch_top_op_path": "n/a",
            "api_lightning_dispatch_per_iter": api_lightning_trace["dispatch_per_iter"],
            "api_torch_dispatch_per_iter": float("nan"),
            "api_lightning_fallback_per_iter": api_lightning_trace["fallback_per_iter"],
            "api_torch_fallback_per_iter": float("nan"),
            "note": "torch backend unavailable",
        }
        _apply_boundary_defaults(interop_row)
        return pure_row, interop_row

    _set_engine("torch")
    api_torch_ms = _time_ms(interop_fn, warmup, iters)
    api_torch_trace = _collect_trace_metrics(interop_fn, trace_iters)
    _set_engine("lightning")

    interop_row = {
        "suite": "engine_split_interop",
        "bench": bench,
        "shape": shape,
        "device": device,
        "status": "ok",
        "lc_api_lightning_ms": api_lightning_ms,
        "lc_api_torch_ms": api_torch_ms,
        "interop_over_pure": _ratio(api_torch_ms, api_lightning_ms),
        "pure_over_interop": _ratio(api_lightning_ms, api_torch_ms),
        "api_lightning_top_op_path": api_lightning_trace["top_op_path"],
        "api_torch_top_op_path": api_torch_trace["top_op_path"],
        "api_lightning_dispatch_per_iter": api_lightning_trace["dispatch_per_iter"],
        "api_torch_dispatch_per_iter": api_torch_trace["dispatch_per_iter"],
        "api_lightning_fallback_per_iter": api_lightning_trace["fallback_per_iter"],
        "api_torch_fallback_per_iter": api_torch_trace["fallback_per_iter"],
        "note": api_torch_trace["note"],
    }
    _apply_boundary_defaults(interop_row)
    return pure_row, interop_row


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure-LC vs interop engine split benchmark")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=120)
    parser.add_argument("--trace-iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "metal", "cpu", "cuda"],
        help="Execution device; auto selects metal when available, otherwise cpu.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--pure-csv", type=str, default="engine_split_pure_lc.csv")
    parser.add_argument("--pure-json", type=str, default="engine_split_pure_lc.json")
    parser.add_argument("--interop-csv", type=str, default="engine_split_interop.csv")
    parser.add_argument("--interop-json", type=str, default="engine_split_interop.json")
    parser.add_argument(
        "--require-shared-bench-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when pure/interop reports do not share identical bench keys.",
    )
    parser.add_argument(
        "--require-torch-interop-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when torch is available but no interop row is status=ok.",
    )
    parser.add_argument(
        "--require-max-boundary-overhead-ms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when estimated boundary overhead exceeds max budget in conv->attn rows.",
    )
    parser.add_argument(
        "--max-boundary-overhead-ms",
        type=float,
        default=0.35,
        help="Maximum allowed estimated boundary overhead (ms) for conv->attn rows.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if _torch_available():
        try:
            torch.manual_seed(args.seed)  # type: ignore[union-attr]
        except Exception:
            pass

    device = _resolve_device(args.device)
    _set_engine("lightning")

    pure_rows: list[dict] = []
    interop_rows: list[dict] = []

    # matmul
    a = np.random.rand(512, 512).astype(np.float32)
    b = np.random.rand(512, 512).astype(np.float32)
    pure, interop = _benchmark_case(
        bench="matmul",
        shape="m=512,k=512,n=512",
        runtime_fn=lambda: lc.matmul2d(a, b, device),
        api_fn=lambda: lc_api.lightning_matmul(a, b, device=device),
        interop_fn=(lambda: lc_api.lightning_matmul(a, b, device=device)) if _torch_available() else None,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        trace_iters=args.trace_iters,
    )
    pure_rows.append(pure)
    interop_rows.append(interop)

    # attention
    q = np.random.rand(256, 64).astype(np.float32)
    k = np.random.rand(256, 64).astype(np.float32)
    v = np.random.rand(256, 64).astype(np.float32)
    pure, interop = _benchmark_case(
        bench="attention",
        shape="seq=256,head_dim=64",
        runtime_fn=lambda: lc.attention2d(q, k, v, False, device),
        api_fn=lambda: lc.api.attention(q, k, v, 256, 64, False, device),
        interop_fn=(lambda: lc.api.attention(q, k, v, 256, 64, False, device)) if _torch_available() else None,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
        trace_iters=args.trace_iters,
    )
    pure_rows.append(pure)
    interop_rows.append(interop)

    # conv+relu targeted tiny/baseline cases
    conv_cases = [
        (1, 3, 16, 16, 16, 3),
        (1, 3, 24, 24, 16, 3),
        (1, 3, 32, 32, 16, 3),
    ]
    for n, c, h, w_h, oc, ksz in conv_cases:
        x = np.random.rand(n, c, h, w_h).astype(np.float32)
        w_conv = np.random.rand(oc, c, ksz, ksz).astype(np.float32)
        b_conv = np.random.rand(oc).astype(np.float32)
        pure, interop = _benchmark_case(
            bench="conv_relu_nchw",
            shape=f"batch={n},in_ch={c},h={h},w={w_h},out_ch={oc},k={ksz}",
            runtime_fn=lambda x=x, w_conv=w_conv, b_conv=b_conv: _runtime_conv_relu(x, w_conv, b_conv, device),
            api_fn=lambda x=x, w_conv=w_conv, b_conv=b_conv: lc.api.conv_relu_nchw(
                x, w_conv, b_conv, 1, 1, 1, 1, device
            ),
            interop_fn=(
                lambda x=x, w_conv=w_conv, b_conv=b_conv: lc.api.conv_relu_nchw(
                    x, w_conv, b_conv, 1, 1, 1, 1, device
                )
            )
            if _torch_available()
            else None,
            device=device,
            warmup=args.warmup,
            iters=args.iters,
            trace_iters=args.trace_iters,
        )
        pure_rows.append(pure)
        interop_rows.append(interop)

    # conv->attn hotspot shape set
    x_pipe = np.random.rand(1, 3, 8, 8).astype(np.float32)
    w_pipe = np.random.rand(16, 3, 3, 3).astype(np.float32)
    b_pipe = np.random.rand(16).astype(np.float32)
    conv_attn_cases = [(48, 48), (96, 48), (192, 48), (192, 8)]
    for seq, dim in conv_attn_cases:
        pure, interop = _benchmark_case(
            bench="conv_attention_torchstrong_nchw",
            shape=f"conv(n=1,c=3->16,h=8,w=8,k=3)+attn(seq={seq},d={dim})",
            runtime_fn=lambda seq=seq, dim=dim: _runtime_conv_attention(x_pipe, w_pipe, b_pipe, seq, dim, device),
            api_fn=lambda seq=seq, dim=dim: lc.api.conv_attention_torchstrong_nchw(
                x_pipe, w_pipe, b_pipe, seq, dim, 1, 1, 1, 1, device, "eager"
            ),
            interop_fn=(
                lambda seq=seq, dim=dim: lc.api.conv_attention_torchstrong_nchw(
                    x_pipe, w_pipe, b_pipe, seq, dim, 1, 1, 1, 1, device, "eager"
                )
            )
            if _torch_available()
            else None,
            device=device,
            warmup=max(8, args.warmup // 2),
            iters=max(60, args.iters // 2),
            trace_iters=max(4, args.trace_iters // 2),
        )
        _set_engine("lightning")
        light_route = _conv_attn_route_report(x=x_pipe, w=w_pipe, b=b_pipe, seq=seq, dim=dim, device=device)
        pure["route_boundary_switch_count"] = int(light_route.get("boundary_switch_count", 0))
        pure["route_boundary_copy_mode"] = str(light_route.get("boundary_copy_mode", "n/a"))
        pure["route_boundary_reason_code"] = str(light_route.get("boundary_reason_code", "n/a"))
        pure["route_boundary_copy_bytes_estimate"] = float(light_route.get("boundary_copy_bytes_estimate", 0))
        pure["route_boundary_overhead_est_ns"] = float(light_route.get("boundary_overhead_est_ns", 0))
        pure["route_boundary_overhead_est_ms"] = float(pure["route_boundary_overhead_est_ns"]) / 1e6
        pure["route_zero_copy_eligible"] = bool(light_route.get("zero_copy_eligible", False))
        if _torch_available():
            _set_engine("torch")
            torch_route = _conv_attn_route_report(x=x_pipe, w=w_pipe, b=b_pipe, seq=seq, dim=dim, device=device)
            _set_engine("lightning")
            interop["route_boundary_switch_count"] = int(torch_route.get("boundary_switch_count", 0))
            interop["route_boundary_copy_mode"] = str(torch_route.get("boundary_copy_mode", "n/a"))
            interop["route_boundary_reason_code"] = str(torch_route.get("boundary_reason_code", "n/a"))
            interop["route_boundary_copy_bytes_estimate"] = float(torch_route.get("boundary_copy_bytes_estimate", 0))
            interop["route_boundary_overhead_est_ns"] = float(torch_route.get("boundary_overhead_est_ns", 0))
            interop["route_boundary_overhead_est_ms"] = float(interop["route_boundary_overhead_est_ns"]) / 1e6
            interop["route_zero_copy_eligible"] = bool(torch_route.get("zero_copy_eligible", False))
        pure_rows.append(pure)
        interop_rows.append(interop)

    pure_fields = [
        "suite",
        "bench",
        "shape",
        "device",
        "status",
        "lc_runtime_ms",
        "lc_api_lightning_ms",
        "api_over_runtime",
        "runtime_over_api",
        "runtime_top_op_path",
        "api_lightning_top_op_path",
        "runtime_dispatch_per_iter",
        "api_lightning_dispatch_per_iter",
        "runtime_fallback_per_iter",
        "api_lightning_fallback_per_iter",
        "note",
        "route_boundary_switch_count",
        "route_boundary_copy_mode",
        "route_boundary_reason_code",
        "route_boundary_copy_bytes_estimate",
        "route_boundary_overhead_est_ns",
        "route_boundary_overhead_est_ms",
        "route_zero_copy_eligible",
    ]
    interop_fields = [
        "suite",
        "bench",
        "shape",
        "device",
        "status",
        "lc_api_lightning_ms",
        "lc_api_torch_ms",
        "interop_over_pure",
        "pure_over_interop",
        "api_lightning_top_op_path",
        "api_torch_top_op_path",
        "api_lightning_dispatch_per_iter",
        "api_torch_dispatch_per_iter",
        "api_lightning_fallback_per_iter",
        "api_torch_fallback_per_iter",
        "note",
        "route_boundary_switch_count",
        "route_boundary_copy_mode",
        "route_boundary_reason_code",
        "route_boundary_copy_bytes_estimate",
        "route_boundary_overhead_est_ns",
        "route_boundary_overhead_est_ms",
        "route_zero_copy_eligible",
    ]

    pure_csv_path = args.out_dir / args.pure_csv
    pure_json_path = args.out_dir / args.pure_json
    interop_csv_path = args.out_dir / args.interop_csv
    interop_json_path = args.out_dir / args.interop_json

    _save_csv(pure_csv_path, pure_rows, pure_fields)
    _save_csv(interop_csv_path, interop_rows, interop_fields)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lc_backend": lc.backend_name(),
        "device": device,
        "engine_api": {
            "has_lc_api_set_engine": bool(hasattr(lc, "api") and hasattr(lc.api, "set_engine")),
            "resolved_engine_after_run": _get_engine(),
        },
        "trace_available": _runtime_trace_available(),
        "trace_iters": args.trace_iters,
        "torch_available": _torch_available(),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    _save_json(pure_json_path, {"meta": meta, "rows": pure_rows})
    _save_json(interop_json_path, {"meta": meta, "rows": interop_rows})

    print(
        f"backend={meta['lc_backend']} device={device} torch_available={meta['torch_available']} "
        f"engine_api_setter={meta['engine_api']['has_lc_api_set_engine']} trace_available={meta['trace_available']}"
    )
    print(f"saved: {pure_csv_path}")
    print(f"saved: {pure_json_path}")
    print(f"saved: {interop_csv_path}")
    print(f"saved: {interop_json_path}")

    print("\n=== Pure-LC Split (runtime vs lc.api(lightning)) ===")
    for row in pure_rows:
        print(
            f"[{row['bench']}] {row['shape']} | runtime={_fmt_ms(row['lc_runtime_ms'])}ms "
            f"api(lightning)={_fmt_ms(row['lc_api_lightning_ms'])}ms api/runtime={_fmt_ratio(row['api_over_runtime'])} "
            f"hotspot={row['api_lightning_top_op_path']}"
        )

    print("\n=== Interop Split (lc.api(torch) vs lc.api(lightning)) ===")
    for row in interop_rows:
        print(
            f"[{row['bench']}] {row['shape']} | api(lightning)={_fmt_ms(row['lc_api_lightning_ms'])}ms "
            f"api(torch)={_fmt_ms(row['lc_api_torch_ms'])}ms interop/pure={_fmt_ratio(row['interop_over_pure'])} "
            f"hotspot(lightning)={row['api_lightning_top_op_path']} status={row['status']}"
        )

    if args.require_shared_bench_coverage:
        pure_keys = {f"{r['bench']}|{r['shape']}" for r in pure_rows}
        interop_keys = {f"{r['bench']}|{r['shape']}" for r in interop_rows}
        if pure_keys != interop_keys:
            raise SystemExit(5)

    if args.require_torch_interop_rows and _torch_available():
        interop_ok = [r for r in interop_rows if r.get("status") == "ok"]
        if not interop_ok:
            raise SystemExit(6)

    if args.require_max_boundary_overhead_ms:
        offenders = []
        for row in interop_rows:
            if str(row.get("bench", "")) != "conv_attention_torchstrong_nchw":
                continue
            value = float(row.get("route_boundary_overhead_est_ms", float("nan")))
            if math.isfinite(value) and value > float(args.max_boundary_overhead_ms):
                offenders.append((str(row.get("shape", "")), value))
        if offenders:
            for shape, value in offenders:
                print(f"boundary-overhead-budget-fail: shape={shape} estimated_overhead_ms={value:.6f}")
            raise SystemExit(7)


if __name__ == "__main__":
    main()
