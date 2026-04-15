#!/usr/bin/env python3
"""Torch runner adapter GA benchmark.

Evaluates nn.Module-style runner wrapper telemetry coverage and boundary overhead budget.
Works even without torch by using a tiny fake torch shim for contract testing.
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


class _FakeDevice:
    def __init__(self, name: str):
        self.type = str(name)


class _FakeTensor:
    def __init__(self, arr: np.ndarray, *, dtype: str, device: str = "cpu"):
        self._arr = np.asarray(arr)
        self.dtype = dtype
        self.device = _FakeDevice(device)

    def detach(self):
        return self

    def to(self, device: str):
        return _FakeTensor(self._arr, dtype=self.dtype, device=device)

    def contiguous(self):
        return self

    def numpy(self):
        return np.asarray(self._arr)

    def is_contiguous(self):
        return True


class _FakeModule:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _FakeTorch:
    float32 = "float32"
    int64 = "int64"

    class nn:  # noqa: N801
        Module = _FakeModule

    @staticmethod
    def as_tensor(value: Any, dtype=None, device: str = "cpu"):
        arr = np.asarray(value)
        if dtype == _FakeTorch.float32:
            arr = np.asarray(arr, dtype=np.float32)
        elif dtype == _FakeTorch.int64:
            arr = np.asarray(arr, dtype=np.int64)
        return _FakeTensor(arr, dtype=dtype if dtype is not None else _FakeTorch.float32, device=device)


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


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
    if hasattr(value, "detach") and callable(getattr(value, "detach")):
        value = value.detach()
    if hasattr(value, "cpu") and callable(getattr(value, "cpu")):
        value = value.cpu()
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
        "## Torch Runner Adapter GA",
        "",
        f"- runtime: `{payload.get('runtime', 'unknown')}`",
        f"- backend: `{payload.get('backend_name', 'unknown')}`",
        f"- schema: `{payload.get('artifact_schema_version', 'unknown')}`",
        f"- reason coverage: `{payload.get('reason_coverage_pct', float('nan')):.2f}%`",
        f"- budget pass rate: `{payload.get('budget_pass_rate_pct', float('nan')):.2f}%`",
        "",
        "| mode | input_kind | status | latency_ms | vs_eager | reason_code | budget_pass | overhead_ms |",
        "| --- | --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for r in rows:
        lat = float(r.get("latency_ms", float("nan")))
        ratio = float(r.get("mode_over_eager", float("nan")))
        lines.append(
            "| "
            f"{r.get('mode')} | {r.get('input_kind')} | {r.get('status')} | "
            f"{lat:.6f} | {ratio:.3f}x | {r.get('boundary_reason_code')} | "
            f"{bool(r.get('boundary_overhead_budget_pass', False))} | {float(r.get('boundary_overhead_est_ms', 0.0)):.6f} |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "metal", "cpu", "cuda"])
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260411)
    p.add_argument("--max-boundary-overhead-ms", type=float, default=5.0)
    p.add_argument("--allow-fake-torch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require-reason-coverage", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-reason-coverage-pct", type=float, default=100.0)
    p.add_argument("--require-budget-gate", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out-dir", type=Path, default=Path("benchmarks/reports/ci"))
    p.add_argument("--csv", type=str, default="torch_runner_adapter.csv")
    p.add_argument("--json", type=str, default="torch_runner_adapter.json")
    p.add_argument("--md", type=str, default="torch_runner_adapter.md")
    args = p.parse_args()

    np.random.seed(args.seed)
    runner = lc_api.TinyTransformerRunner(seq_len=48, d_model=48, d_ff=128, vocab_size=256, seed=args.seed)

    torch_mod = None
    runtime = "torch"
    if _torch_available():
        import torch as _torch

        torch_mod = _torch
    elif args.allow_fake_torch:
        torch_mod = _FakeTorch
        runtime = "fake_torch"
    else:
        raise RuntimeError("torch is unavailable and --no-allow-fake-torch was requested")

    rows: list[dict] = []
    modes = ["eager", "graph", "interop"]
    input_kinds = ["token_ids", "embedding_features"]

    eager_latency_by_input: dict[str, float] = {}
    for input_kind in input_kinds:
        if input_kind == "token_ids":
            x_np = np.asarray(np.random.randint(0, 256, size=(48,)), dtype=np.int64)
            dtype = getattr(torch_mod, "int64", None)
        else:
            x_np = (np.random.standard_normal((48, 48)) * 0.2).astype(np.float32)
            dtype = getattr(torch_mod, "float32", None)
        x_t = torch_mod.as_tensor(x_np, dtype=dtype, device="cpu")

        for mode in modes:
            wrapper = lc_api.create_torch_module_wrapper(
                runner,
                mode=mode,
                device=args.device,
                route_policy={"conv": "auto", "attention": "auto", "graph": "auto"},
                overhead_budget_ms=float(args.max_boundary_overhead_ms),
                torch_module=torch_mod,
            )

            fn = lambda: wrapper(x_t)
            lat = _median_ms(fn, args.warmup, args.iters)
            out_t = wrapper(x_t)
            telem = lc_api.get_torch_wrapper_telemetry(wrapper)
            out_np = _to_numpy(out_t)
            ref_np = np.asarray(runner.run(x_np, mode=mode, device=args.device), dtype=np.float32)
            allclose = bool(np.allclose(out_np, ref_np, atol=3.0e-3, rtol=3.0e-3))

            if mode == "eager":
                eager_latency_by_input[input_kind] = lat
            eager_lat = float(eager_latency_by_input.get(input_kind, lat))

            boundary_reason = str(telem.get("boundary_reason_code", ""))
            runner_reason = str(telem.get("runner_fallback_reason_code", "none"))
            reason_covered = bool(boundary_reason.strip()) and bool(telem.get("reason_code_covered", False))
            budget_pass = bool(telem.get("boundary_overhead_budget_pass", False))

            rows.append(
                {
                    "suite": "torch_runner_adapter",
                    "mode": str(mode),
                    "input_kind": str(input_kind),
                    "status": "ok" if allclose else "mismatch",
                    "device": str(args.device),
                    "runtime": runtime,
                    "latency_ms": float(lat),
                    "mode_over_eager": _ratio(lat, eager_lat),
                    "allclose_vs_runner": bool(allclose),
                    "resolved_mode": str(telem.get("resolved_mode", mode)),
                    "resolved_engine": str(telem.get("resolved_engine", "unknown")),
                    "runner_fallback_reason_code": runner_reason,
                    "boundary_reason_code": boundary_reason,
                    "boundary_copy_mode": str(telem.get("boundary_copy_mode", "n/a")),
                    "boundary_bridge_path_in": str(telem.get("boundary_bridge_path_in", "n/a")),
                    "boundary_bridge_path_out": str(telem.get("boundary_bridge_path_out", "n/a")),
                    "boundary_copy_bytes_estimate": float(telem.get("boundary_copy_bytes_estimate", 0.0)),
                    "boundary_overhead_est_ns": float(telem.get("boundary_overhead_est_ns", 0.0)),
                    "boundary_overhead_est_ms": float(telem.get("boundary_overhead_est_ms", 0.0)),
                    "boundary_overhead_budget_ms": float(telem.get("boundary_overhead_budget_ms", args.max_boundary_overhead_ms)),
                    "boundary_overhead_budget_pass": bool(budget_pass),
                    "reason_code_covered": bool(reason_covered),
                    "zero_copy_eligible": bool(telem.get("zero_copy_eligible", False)),
                    "zero_copy_effective": bool(telem.get("zero_copy_effective", False)),
                    "note": "",
                }
            )

    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    reason_cov_rows = [r for r in ok_rows if bool(r.get("reason_code_covered", False))]
    budget_ok_rows = [r for r in ok_rows if bool(r.get("boundary_overhead_budget_pass", False))]

    reason_coverage_pct = 100.0 if not ok_rows else (100.0 * float(len(reason_cov_rows)) / float(len(ok_rows)))
    budget_pass_rate_pct = 100.0 if not ok_rows else (100.0 * float(len(budget_ok_rows)) / float(len(ok_rows)))

    fields = [
        "suite",
        "mode",
        "input_kind",
        "status",
        "device",
        "runtime",
        "latency_ms",
        "mode_over_eager",
        "allclose_vs_runner",
        "resolved_mode",
        "resolved_engine",
        "runner_fallback_reason_code",
        "boundary_reason_code",
        "boundary_copy_mode",
        "boundary_bridge_path_in",
        "boundary_bridge_path_out",
        "boundary_copy_bytes_estimate",
        "boundary_overhead_est_ns",
        "boundary_overhead_est_ms",
        "boundary_overhead_budget_ms",
        "boundary_overhead_budget_pass",
        "reason_code_covered",
        "zero_copy_eligible",
        "zero_copy_effective",
        "note",
    ]

    csv_path = args.out_dir / args.csv
    json_path = args.out_dir / args.json
    md_path = args.out_dir / args.md
    _save_csv(csv_path, rows, fields)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "suite": "torch_runner_adapter",
        "artifact_schema_version": "torch_runner_adapter_artifact_v1",
        "backend_name": str(lc.backend_name()),
        "runtime": runtime,
        "device": str(args.device),
        "seed": int(args.seed),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "max_boundary_overhead_ms": float(args.max_boundary_overhead_ms),
        "reason_coverage_pct": float(reason_coverage_pct),
        "budget_pass_rate_pct": float(budget_pass_rate_pct),
        "adapter_schema": lc_api.torch_runner_adapter_schema() if hasattr(lc_api, "torch_runner_adapter_schema") else {},
        "rows": rows,
    }
    _save_json(json_path, payload)
    md_path.write_text(_render_md(payload), encoding="utf-8")

    print(
        f"runtime={runtime} reason_coverage_pct={reason_coverage_pct:.2f} "
        f"budget_pass_rate_pct={budget_pass_rate_pct:.2f}"
    )
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")

    if args.require_reason_coverage and reason_coverage_pct + 1.0e-9 < float(args.min_reason_coverage_pct):
        raise SystemExit(2)
    if args.require_budget_gate and budget_pass_rate_pct + 1.0e-9 < 100.0:
        raise SystemExit(3)
    if any(str(r.get("status", "")).lower() != "ok" for r in rows):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
