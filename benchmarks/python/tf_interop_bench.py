#!/usr/bin/env python3
"""TensorFlow bridge prototype benchmark (v0.3.3).

Emits pure-LC vs TF-wrapper timing in the same CSV/JSON artifact style used by
engine split reports, while staying runnable when TensorFlow is not installed.
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
import lightning_core_integrated_api as lc_api


def _median_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(median(samples))


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0.0 or (not math.isfinite(numer)) or (not math.isfinite(denom)):
        return float("nan")
    return float(numer / denom)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "numpy") and callable(getattr(value, "numpy")):
        return np.asarray(value.numpy(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--mode", type=str, default="eager", choices=["eager", "graph", "interop"])
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--iters", type=int, default=24)
    p.add_argument("--seed", type=int, default=20260411)
    p.add_argument(
        "--force-graph-policy-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force deterministic graph fallback code by setting route_policy.graph=torch.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/reports/ci"))
    p.add_argument("--csv", type=str, default="tf_interop.csv")
    p.add_argument("--json", type=str, default="tf_interop.json")
    p.add_argument(
        "--require-reason-coverage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail when TF wrapper boundary reason-code coverage is below threshold.",
    )
    p.add_argument("--min-reason-coverage-pct", type=float, default=100.0)
    args = p.parse_args()

    np.random.seed(args.seed)
    x = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, seed=args.seed)

    route_policy = {"conv": "auto", "attention": "auto", "graph": "auto"}
    if args.force_graph_policy_fallback:
        route_policy["graph"] = "torch"

    def run_pure_once():
        y = runner.run(x, mode=args.mode, device=args.device, route_policy=route_policy)
        return np.asarray(y, dtype=np.float32)

    wrapper = lc_api.create_tf_keras_layer_wrapper(
        runner,
        mode=args.mode,
        device=args.device,
        route_policy=route_policy,
        prefer_tensorflow_runtime=True,
        allow_missing_runtime=True,
    )

    def run_tf_once():
        y = wrapper(x)
        return _to_numpy(y)

    pure_ms = _median_ms(run_pure_once, args.warmup, args.iters)
    tf_ms = _median_ms(run_tf_once, args.warmup, args.iters)
    y_pure = run_pure_once()
    y_tf = run_tf_once()
    telem = lc_api.get_tf_wrapper_telemetry(wrapper)
    boundary_reason = str(telem.get("boundary_reason_code", "")).strip()
    reason_covered = boundary_reason.lower() not in {"", "n/a", "none"}
    reason_coverage_pct = 100.0 if reason_covered else 0.0

    row = {
        "suite": "tf_interop",
        "bench": "tiny_transformer_runner_tf_bridge",
        "shape": "seq=48,d_model=48",
        "device": args.device,
        "status": "ok",
        "lc_api_lightning_ms": float(pure_ms),
        "lc_api_tf_ms": float(tf_ms),
        "interop_over_pure": _ratio(tf_ms, pure_ms),
        "pure_over_interop": _ratio(pure_ms, tf_ms),
        "allclose_vs_pure": bool(np.allclose(y_tf, y_pure, atol=3.0e-3, rtol=3.0e-3)),
        "requested_mode": str(args.mode),
        "resolved_mode": str(telem.get("resolved_mode", args.mode)),
        "resolved_engine": str(telem.get("resolved_engine", "unknown")),
        "runner_fallback_reason_code": str(telem.get("runner_fallback_reason_code", "none")),
        "route_policy_json": json.dumps(route_policy, sort_keys=True, ensure_ascii=False, separators=(",", ":")),
        "route_boundary_reason_code": str(telem.get("boundary_reason_code", "n/a")),
        "route_boundary_copy_mode": str(telem.get("boundary_copy_mode", "n/a")),
        "route_boundary_copy_bytes_estimate": float(telem.get("boundary_copy_bytes_estimate", 0.0)),
        "route_boundary_overhead_est_ns": float(telem.get("boundary_overhead_est_ns", 0.0)),
        "route_boundary_overhead_est_ms": float(telem.get("boundary_overhead_est_ms", 0.0)),
        "route_zero_copy_eligible": bool(telem.get("zero_copy_eligible", False)),
        "tf_runtime": str(telem.get("runtime", "unknown")),
        "tf_runtime_available": bool(str(telem.get("runtime", "")) == "tensorflow"),
        "note": "",
    }

    rows = [row]
    fields = [
        "suite",
        "bench",
        "shape",
        "device",
        "status",
        "lc_api_lightning_ms",
        "lc_api_tf_ms",
        "interop_over_pure",
        "pure_over_interop",
        "allclose_vs_pure",
        "requested_mode",
        "resolved_mode",
        "resolved_engine",
        "runner_fallback_reason_code",
        "route_policy_json",
        "route_boundary_reason_code",
        "route_boundary_copy_mode",
        "route_boundary_copy_bytes_estimate",
        "route_boundary_overhead_est_ns",
        "route_boundary_overhead_est_ms",
        "route_zero_copy_eligible",
        "tf_runtime",
        "tf_runtime_available",
        "note",
    ]

    csv_path = args.out_dir / args.csv
    json_path = args.out_dir / args.json
    _save_csv(csv_path, rows, fields)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "tf_interop",
        "artifact_schema_version": "tf_interop_artifact_v1",
        "backend_name": str(lc.backend_name()),
        "device": args.device,
        "mode": args.mode,
        "seed": int(args.seed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "tf_runtime_available": bool(row["tf_runtime_available"]),
        "tf_runtime": row["tf_runtime"],
        "reason_coverage_pct": float(reason_coverage_pct),
        "rows": rows,
    }
    _save_json(json_path, payload)

    print(
        f"tf_runtime={payload['tf_runtime']} tf_runtime_available={payload['tf_runtime_available']} "
        f"reason_coverage_pct={payload['reason_coverage_pct']:.2f}"
    )
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")

    if args.require_reason_coverage:
        observed = float(reason_coverage_pct)
        target = float(args.min_reason_coverage_pct)
        if observed + 1.0e-9 < target:
            raise SystemExit(2)
    if not bool(row["allclose_vs_pure"]):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
