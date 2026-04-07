#!/usr/bin/env python3
"""Engine-split benchmark for pure-LC vs interop overhead.

Outputs are intentionally separated:
- pure-LC report: runtime direct vs lc.api(lightning)
- interop report: lc.api(lightning) vs lc.api(torch)
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
) -> tuple[dict, dict]:
    runtime_ms = _time_ms(runtime_fn, warmup, iters)
    _set_engine("lightning")
    api_lightning_ms = _time_ms(api_fn, warmup, iters)

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
        "note": "",
    }

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
            "note": "torch backend unavailable",
        }
        return pure_row, interop_row

    _set_engine("torch")
    api_torch_ms = _time_ms(interop_fn, warmup, iters)
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
        "note": "",
    }
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
    )
    pure_rows.append(pure)
    interop_rows.append(interop)

    # conv+relu
    x = np.random.rand(1, 3, 32, 32).astype(np.float32)
    w_conv = np.random.rand(16, 3, 3, 3).astype(np.float32)
    b_conv = np.random.rand(16).astype(np.float32)
    pure, interop = _benchmark_case(
        bench="conv_relu_nchw",
        shape="batch=1,in_ch=3,h=32,w=32,out_ch=16,k=3",
        runtime_fn=lambda: _runtime_conv_relu(x, w_conv, b_conv, device),
        api_fn=lambda: lc.api.conv_relu_nchw(x, w_conv, b_conv, 1, 1, 1, 1, device),
        interop_fn=(lambda: lc.api.conv_relu_nchw(x, w_conv, b_conv, 1, 1, 1, 1, device)) if _torch_available() else None,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    pure_rows.append(pure)
    interop_rows.append(interop)

    # conv->attn pipeline
    x_pipe = np.random.rand(1, 3, 8, 8).astype(np.float32)
    w_pipe = np.random.rand(16, 3, 3, 3).astype(np.float32)
    b_pipe = np.random.rand(16).astype(np.float32)
    pure, interop = _benchmark_case(
        bench="conv_attention_torchstrong_nchw",
        shape="conv(n=1,c=3->16,h=8,w=8,k=3)+attn(seq=96,d=48)",
        runtime_fn=lambda: _runtime_conv_attention(x_pipe, w_pipe, b_pipe, 96, 48, device),
        api_fn=lambda: lc.api.conv_attention_torchstrong_nchw(x_pipe, w_pipe, b_pipe, 96, 48, 1, 1, 1, 1, device, "eager"),
        interop_fn=(
            lambda: lc.api.conv_attention_torchstrong_nchw(x_pipe, w_pipe, b_pipe, 96, 48, 1, 1, 1, 1, device, "eager")
        )
        if _torch_available()
        else None,
        device=device,
        warmup=max(8, args.warmup // 2),
        iters=max(60, args.iters // 2),
    )
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
        "note",
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
        "note",
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
        "torch_available": _torch_available(),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
    }

    _save_json(pure_json_path, {"meta": meta, "rows": pure_rows})
    _save_json(interop_json_path, {"meta": meta, "rows": interop_rows})

    print(
        f"backend={meta['lc_backend']} device={device} torch_available={meta['torch_available']} "
        f"engine_api_setter={meta['engine_api']['has_lc_api_set_engine']}"
    )
    print(f"saved: {pure_csv_path}")
    print(f"saved: {pure_json_path}")
    print(f"saved: {interop_csv_path}")
    print(f"saved: {interop_json_path}")

    print("\n=== Pure-LC Split (runtime vs lc.api(lightning)) ===")
    for row in pure_rows:
        print(
            f"[{row['bench']}] {row['shape']} | runtime={_fmt_ms(row['lc_runtime_ms'])}ms "
            f"api(lightning)={_fmt_ms(row['lc_api_lightning_ms'])}ms api/runtime={_fmt_ratio(row['api_over_runtime'])}"
        )

    print("\n=== Interop Split (lc.api(torch) vs lc.api(lightning)) ===")
    for row in interop_rows:
        print(
            f"[{row['bench']}] {row['shape']} | api(lightning)={_fmt_ms(row['lc_api_lightning_ms'])}ms "
            f"api(torch)={_fmt_ms(row['lc_api_torch_ms'])}ms interop/pure={_fmt_ratio(row['interop_over_pure'])} "
            f"status={row['status']}"
        )


if __name__ == "__main__":
    main()
