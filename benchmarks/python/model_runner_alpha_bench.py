#!/usr/bin/env python3
"""Model Runner Alpha benchmark (graph/eager/interop mode switch)."""

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
import lightning_core_integrated_api as lc_api


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


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0.0 or (not math.isfinite(numer)) or (not math.isfinite(denom)):
        return float("nan")
    return numer / denom


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown(payload: dict) -> str:
    rows = list(payload.get("rows", []))
    lines = []
    lines.append("## Model Runner Alpha Benchmark")
    lines.append("")
    lines.append(f"- backend: `{payload.get('backend_name', 'unknown')}`")
    lines.append(f"- device: `{payload.get('device', 'unknown')}`")
    lines.append("")
    lines.append("| mode | status | latency_ms | vs_eager | allclose_vs_eager |")
    lines.append("| --- | --- | ---: | ---: | --- |")
    for row in rows:
        lat = row.get("latency_ms", float("nan"))
        lat_s = "n/a" if not math.isfinite(float(lat)) else f"{float(lat):.6f}"
        ratio = row.get("mode_over_eager", float("nan"))
        ratio_s = "n/a" if not math.isfinite(float(ratio)) else f"{float(ratio):.3f}x"
        lines.append(
            f"| {row.get('mode')} | {row.get('status')} | {lat_s} | {ratio_s} | {bool(row.get('allclose_vs_eager', False))} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260410)
    p.add_argument("--csv", type=Path, default=Path("benchmarks/reports/ci/model_runner_alpha.csv"))
    p.add_argument("--json", type=Path, default=Path("benchmarks/reports/ci/model_runner_alpha.json"))
    p.add_argument("--md", type=Path, default=Path("benchmarks/reports/ci/model_runner_alpha.md"))
    p.add_argument("--require-interop", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    np.random.seed(args.seed)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, seed=args.seed)
    x = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)

    mode_rows: list[dict] = []
    modes = ["eager", "graph", "interop"]
    eager_ref: np.ndarray | None = None
    eager_ms = float("nan")
    for mode in modes:
        try:
            fn = lambda m=mode: runner.run(x, mode=m, device=args.device)
            lat = _median_ms(fn, args.warmup, args.iters)
            out = np.asarray(fn(), dtype=np.float32)
            if mode == "eager":
                eager_ref = out
                eager_ms = lat
            allclose = bool(np.allclose(out, eager_ref, atol=1.0e-4, rtol=1.0e-4)) if eager_ref is not None else True
            mode_rows.append(
                {
                    "suite": "model_runner_alpha",
                    "mode": mode,
                    "status": "ok",
                    "device": args.device,
                    "latency_ms": float(lat),
                    "mode_over_eager": _ratio(float(lat), float(eager_ms)),
                    "allclose_vs_eager": allclose,
                    "note": "",
                }
            )
        except Exception as exc:
            mode_rows.append(
                {
                    "suite": "model_runner_alpha",
                    "mode": mode,
                    "status": "unsupported",
                    "device": args.device,
                    "latency_ms": float("nan"),
                    "mode_over_eager": float("nan"),
                    "allclose_vs_eager": False,
                    "note": str(exc),
                }
            )

    if args.require_interop:
        interop_ok = [r for r in mode_rows if r["mode"] == "interop" and r["status"] == "ok"]
        if not interop_ok:
            raise SystemExit(3)

    fields = [
        "suite",
        "mode",
        "status",
        "device",
        "latency_ms",
        "mode_over_eager",
        "allclose_vs_eager",
        "note",
    ]
    _save_csv(args.csv, mode_rows, fields)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "model_runner_alpha",
        "backend_name": str(lc.backend_name()),
        "device": args.device,
        "warmup": args.warmup,
        "iters": args.iters,
        "seed": args.seed,
        "rows": mode_rows,
    }
    _save_json(args.json, payload)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(_markdown(payload), encoding="utf-8")
    print(f"saved: {args.csv}")
    print(f"saved: {args.json}")
    print(f"saved: {args.md}")


if __name__ == "__main__":
    main()

