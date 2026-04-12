#!/usr/bin/env python3
"""TensorFlow runner adapter beta benchmark.

Covers tensorflow-runtime path and missing-runtime (numpy shim) path with a
fixed artifact schema so CI can enforce regressions deterministically.
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
from typing import Any

import numpy as np

import lightning_core as lc
import lightning_core_integrated_api as lc_api


class _FakeLayer:
    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs, *args, **kwargs)

    def call(self, inputs, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _FakeTF:
    class keras:  # noqa: N801
        class layers:  # noqa: N801
            Layer = _FakeLayer

    @staticmethod
    def convert_to_tensor(value, dtype=None):
        return np.asarray(value, dtype=dtype if dtype is not None else np.float32)


def _median_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, int(warmup))):
        fn()
    samples: list[float] = []
    for _ in range(max(1, int(iters))):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        samples.append((t1 - t0) / 1e6)
    return float(median(samples))


def _ratio(numer: float, denom: float) -> float:
    if denom <= 0.0 or (not math.isfinite(numer)) or (not math.isfinite(denom)):
        return float("nan")
    return float(numer / denom)


def _to_numpy(value: Any) -> np.ndarray:
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


def _render_md(payload: dict) -> str:
    rows = list(payload.get("rows", []))
    lines = [
        "## TF Runner Adapter Beta",
        "",
        f"- backend: `{payload.get('backend_name', 'unknown')}`",
        f"- schema: `{payload.get('artifact_schema_version', 'unknown')}`",
        f"- reason coverage: `{payload.get('reason_coverage_pct', float('nan')):.2f}%`",
        f"- schema pass: `{bool(payload.get('artifact_schema_pass', False))}`",
        "",
        "| runtime | mode | input_kind | status | latency_ms | vs_eager | reason_code |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            "| "
            f"{r.get('runtime')} | {r.get('mode')} | {r.get('input_kind')} | {r.get('status')} | "
            f"{float(r.get('latency_ms', float('nan'))):.6f} | {float(r.get('mode_over_eager', float('nan'))):.3f}x | "
            f"{r.get('boundary_reason_code')} |"
        )
    return "\n".join(lines)


def _required_row_fields() -> list[str]:
    return [
        "suite",
        "runtime",
        "mode",
        "input_kind",
        "status",
        "device",
        "latency_ms",
        "mode_over_eager",
        "allclose_vs_runner",
        "resolved_mode",
        "resolved_engine",
        "runner_fallback_reason_code",
        "boundary_reason_code",
        "boundary_copy_mode",
        "boundary_copy_bytes_estimate",
        "boundary_overhead_est_ns",
        "boundary_overhead_est_ms",
        "boundary_overhead_budget_ms",
        "boundary_overhead_budget_pass",
        "reason_code_covered",
        "note",
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260411)
    p.add_argument("--max-boundary-overhead-ms", type=float, default=5.0)
    p.add_argument("--require-reason-coverage", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-reason-coverage-pct", type=float, default=100.0)
    p.add_argument("--require-artifact-schema", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/reports/ci"))
    p.add_argument("--csv", type=str, default="tf_runner_adapter.csv")
    p.add_argument("--json", type=str, default="tf_runner_adapter.json")
    p.add_argument("--md", type=str, default="tf_runner_adapter.md")
    args = p.parse_args()

    np.random.seed(args.seed)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=args.seed)

    rows: list[dict] = []
    runtime_modes = ["tensorflow", "numpy_shim"]
    test_modes = ["eager", "graph"]
    input_kinds = ["token_ids", "embedding_features"]
    eager_by_runtime_input: dict[tuple[str, str], float] = {}

    for runtime in runtime_modes:
        for input_kind in input_kinds:
            if input_kind == "token_ids":
                x_np = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)
            else:
                x_np = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)

            for mode in test_modes:
                route_policy = {"conv": "auto", "attention": "auto", "graph": "auto"}
                if runtime == "numpy_shim" and mode == "graph":
                    route_policy["graph"] = "torch"

                adapter = lc_api.create_tf_model_runner_adapter(
                    runner,
                    mode=mode,
                    device=args.device,
                    route_policy=route_policy,
                    prefer_tensorflow_runtime=False,
                    allow_missing_runtime=(runtime == "numpy_shim"),
                    overhead_budget_ms=float(args.max_boundary_overhead_ms),
                    tensorflow_module=(_FakeTF if runtime == "tensorflow" else None),
                )

                fn = lambda: adapter.infer(x_np)
                lat = _median_ms(fn, args.warmup, args.iters)
                y = _to_numpy(adapter.infer(x_np))
                ref = np.asarray(runner.run(x_np, mode=mode, device=args.device, route_policy=route_policy), dtype=np.float32)
                allclose = bool(np.allclose(y, ref, atol=3.0e-3, rtol=3.0e-3))
                telem = adapter.last_telemetry()

                key = (runtime, input_kind)
                if mode == "eager":
                    eager_by_runtime_input[key] = lat
                eager_lat = float(eager_by_runtime_input.get(key, lat))

                rows.append(
                    {
                        "suite": "tf_runner_adapter",
                        "runtime": runtime,
                        "mode": mode,
                        "input_kind": input_kind,
                        "status": "ok" if allclose else "mismatch",
                        "device": str(args.device),
                        "latency_ms": float(lat),
                        "mode_over_eager": _ratio(lat, eager_lat),
                        "allclose_vs_runner": bool(allclose),
                        "resolved_mode": str(telem.get("resolved_mode", mode)),
                        "resolved_engine": str(telem.get("resolved_engine", "unknown")),
                        "runner_fallback_reason_code": str(telem.get("runner_fallback_reason_code", "none")),
                        "boundary_reason_code": str(telem.get("boundary_reason_code", "")),
                        "boundary_copy_mode": str(telem.get("boundary_copy_mode", "n/a")),
                        "boundary_copy_bytes_estimate": float(telem.get("boundary_copy_bytes_estimate", 0.0)),
                        "boundary_overhead_est_ns": float(telem.get("boundary_overhead_est_ns", 0.0)),
                        "boundary_overhead_est_ms": float(telem.get("boundary_overhead_est_ms", 0.0)),
                        "boundary_overhead_budget_ms": float(telem.get("boundary_overhead_budget_ms", args.max_boundary_overhead_ms)),
                        "boundary_overhead_budget_pass": bool(telem.get("boundary_overhead_budget_pass", False)),
                        "reason_code_covered": bool(telem.get("reason_code_covered", False)),
                        "note": "",
                    }
                )

    row_fields = _required_row_fields()
    schema_pass = all(all(f in r for f in row_fields) for r in rows)

    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    reason_cov_rows = [r for r in ok_rows if bool(r.get("reason_code_covered", False)) and str(r.get("boundary_reason_code", "")).strip()]
    reason_coverage_pct = 100.0 if not ok_rows else (100.0 * float(len(reason_cov_rows)) / float(len(ok_rows)))

    runtimes_observed = {str(r.get("runtime", "")) for r in rows}
    both_runtime_paths_ok = bool({"tensorflow", "numpy_shim"}.issubset(runtimes_observed))

    csv_path = args.out_dir / args.csv
    json_path = args.out_dir / args.json
    md_path = args.out_dir / args.md
    _save_csv(csv_path, rows, row_fields)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "tf_runner_adapter",
        "artifact_schema_version": "tf_runner_adapter_artifact_v1",
        "backend_name": str(lc.backend_name()),
        "device": str(args.device),
        "seed": int(args.seed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "adapter_schema": lc_api.tf_runner_adapter_schema() if hasattr(lc_api, "tf_runner_adapter_schema") else {},
        "required_row_fields": row_fields,
        "artifact_schema_pass": bool(schema_pass),
        "reason_coverage_pct": float(reason_coverage_pct),
        "both_runtime_paths_ok": bool(both_runtime_paths_ok),
        "rows": rows,
    }
    _save_json(json_path, payload)
    md_path.write_text(_render_md(payload), encoding="utf-8")

    print(
        f"reason_coverage_pct={reason_coverage_pct:.2f} schema_pass={schema_pass} both_runtime_paths_ok={both_runtime_paths_ok}"
    )
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")

    if args.require_reason_coverage and reason_coverage_pct + 1.0e-9 < float(args.min_reason_coverage_pct):
        raise SystemExit(2)
    if args.require_artifact_schema and (not schema_pass or not both_runtime_paths_ok):
        raise SystemExit(3)
    if any(str(r.get("status", "")).lower() != "ok" for r in rows):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
